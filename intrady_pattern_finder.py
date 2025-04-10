import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    layout="wide",
    page_title="Crypto Pattern Finder",
    page_icon="🔍",
)

# Title and description
st.title("Crypto Pattern Finder")
st.markdown("Identifies the most similar historical trading patterns to today's price action")

def fetch_crypto_data(symbol, period="60d", interval="15m"):
    """Fetch crypto data from Yahoo Finance"""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    
    if hist.empty:
        return None
        
    # Convert timestamps to EST
    hist.index = hist.index.tz_convert('US/Eastern')
    
    return hist

def calculate_daily_pattern(data, num_days=60):
    """Calculate intraday pattern using complete days"""
    # Get today's date to exclude it from historical patterns
    today = pd.Timestamp.now(tz='US/Eastern').date()
    
    # Get the most recent days
    end_date = data.index.max()
    start_date = end_date - pd.Timedelta(days=num_days)
    filtered_data = data[data.index >= start_date].copy()
    
    # Create empty DataFrame for patterns
    all_patterns = pd.DataFrame()
    
    # Get unique dates
    unique_dates = pd.Series(filtered_data.index.date).drop_duplicates()
    
    # Determine expected points per day based on interval
    interval = pd.Timedelta(filtered_data.index.freq or filtered_data.index[1] - filtered_data.index[0])
    expected_points = int(pd.Timedelta('1D') / interval)
    
    # Process each day
    for date in unique_dates:
        # Skip today for historical patterns
        if date == today:
            continue
            
        # Get data for this day
        day_data = filtered_data[filtered_data.index.date == date].copy()
        
        # Only process days with expected number of points
        if len(day_data) >= expected_points * 0.9:  # Allow for small variations
            # Calculate percentage change from day's open
            day_open = day_data['Close'].iloc[0]
            day_pattern = ((day_data['Close'] - day_open) / day_open) * 100
            
            # Store pattern using time as index
            try:
                # Only convert to time if timestamps are available
                if all(isinstance(idx, pd.Timestamp) for idx in day_data.index):
                    day_pattern.index = day_data.index.time
                # Otherwise keep the original index
                all_patterns[date] = day_pattern
            except Exception as e:
                # If time extraction fails, skip this day
                continue
    
    # We're removing the average pattern calculation
    # Calculate average pattern
    # avg_pattern = all_patterns.mean(axis=1)
    
    return None, all_patterns  # Return None instead of avg_pattern

def find_top_similar_days(today_data, historical_patterns, num_matches=10, recent_bias=False, min_overlap=5):
    """Find the top N historical days that most closely match today's pattern"""
    if today_data.empty or len(today_data) < min_overlap:
        return []
    
    # Calculate today's pattern
    today_open = today_data['Close'].iloc[0]
    today_pattern = ((today_data['Close'] - today_open) / today_open) * 100
    
    try:
        # Make sure index consists of timestamps before extracting time
        if all(isinstance(idx, pd.Timestamp) for idx in today_data.index):
            today_pattern.index = today_data.index.time
        # If not timestamps, keep original index
    except Exception as e:
        # If we can't extract time, return empty list - this pattern can't be matched
        return []
    
    # Store similarity scores for all days
    similarity_scores = []
    
    for date, pattern in historical_patterns.items():
        # Skip if patterns don't have enough overlap
        common_times = set(pattern.index).intersection(set(today_pattern.index))
        if len(common_times) < min_overlap:
            continue
            
        # Calculate similarity (mean squared error) for common time points
        squared_diffs = []
        times_list = []
        
        for time in common_times:
            if time in pattern.index and time in today_pattern.index:
                diff = today_pattern[time] - pattern[time]
                squared_diffs.append(diff * diff)
                times_list.append(time.hour * 60 + time.minute)  # Convert to minutes for weighting
        
        if squared_diffs:
            # Calculate weighted MSE - more weight to recent points if requested
            if recent_bias:
                max_time = max(times_list)
                normalized_times = [t/max_time for t in times_list]
                weights = np.array([0.5 + 0.5 * t for t in normalized_times])
            else:
                weights = np.ones(len(squared_diffs))
                
            weighted_mse = sum(w * d for w, d in zip(weights, squared_diffs)) / sum(weights)
            
            # Add end-of-day value for trend analysis
            end_value = pattern.iloc[-1] if not pattern.empty else 0
            
            similarity_scores.append({
                'date': date,
                'similarity': weighted_mse,
                'pattern': pattern,
                'timestamp': pd.Timestamp(date),
                'end_value': end_value,
                'trend': 'up' if end_value > 0 else 'down'
            })
    
    # Sort by similarity (lower is better)
    similarity_scores.sort(key=lambda x: x['similarity'])
    
    # Return top N matches
    return similarity_scores[:num_matches]

def group_similar_patterns(patterns, threshold=0.8):
    """Group patterns that are similar to each other"""
    if not patterns:
        return []
        
    grouped_patterns = []
    used_indices = set()
    
    for i, pattern1 in enumerate(patterns):
        if i in used_indices:
            continue
            
        group = [pattern1]
        used_indices.add(i)
        
        for j, pattern2 in enumerate(patterns):
            if j in used_indices or i == j:
                continue
                
            # Calculate similarity between patterns
            p1 = pattern1['pattern']
            p2 = pattern2['pattern']
            common_times = set(p1.index).intersection(set(p2.index))
            
            if len(common_times) < 5:
                continue
                
            squared_diffs = []
            for time in common_times:
                if time in p1.index and time in p2.index:
                    diff = p1[time] - p2[time]
                    squared_diffs.append(diff * diff)
            
            if squared_diffs:
                similarity = sum(squared_diffs) / len(squared_diffs)
                if similarity < threshold:
                    group.append(pattern2)
                    used_indices.add(j)
        
        # Add the group
        if len(group) > 1:
            # Use the best matching pattern as the representative
            best_pattern = min(group, key=lambda x: x['similarity'])
            best_pattern['count'] = len(group)
            best_pattern['similar_dates'] = [p['date'] for p in group if p['date'] != best_pattern['date']]
            grouped_patterns.append(best_pattern)
        else:
            pattern1['count'] = 1
            pattern1['similar_dates'] = []
            grouped_patterns.append(pattern1)
    
    # Sort by similarity again
    grouped_patterns.sort(key=lambda x: x['similarity'])
    
    return grouped_patterns

def plot_pattern_matches(data, symbol, top_matches, show_scores=True, max_patterns=10):
    """Create plot showing pattern matches with today's data"""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Today's date for finding most similar pattern
    today = pd.Timestamp.now(tz='US/Eastern').date()
    today_data = data[data.index.date == today]
    
    # Colors for different patterns (extend for more patterns)
    colors = {
        'today': 'rgb(255, 255, 255)',    # White
        'match1': 'rgb(255, 99, 132)',    # Red
        'match2': 'rgb(66, 135, 245)',    # Blue
        'match3': 'rgb(52, 191, 73)',     # Green
        'match4': 'rgb(242, 184, 64)',    # Yellow
        'match5': 'rgb(153, 102, 255)',   # Purple
        'match6': 'rgb(0, 204, 204)',     # Teal
        'match7': 'rgb(255, 51, 153)',    # Pink
        'match8': 'rgb(102, 255, 102)',   # Light Green
        'match9': 'rgb(255, 204, 0)',     # Gold
        'match10': 'rgb(204, 102, 255)',  # Lavender
    }
    
            # Add today's price
    if not today_data.empty:
        today_open = today_data['Close'].iloc[0]
        today_changes = ((today_data['Close'] - today_open) / today_open) * 100
        
        # Safely extract time strings
        time_strings = []
        for idx in today_data.index:
            try:
                if isinstance(idx, pd.Timestamp):
                    time_strings.append(idx.strftime('%H:%M'))
                else:
                    # If not a timestamp, convert to string in some reasonable way
                    time_strings.append(str(idx))
            except:
                time_strings.append('00:00')  # Fallback time
        
        fig.add_trace(
            go.Scatter(
                x=time_strings,
                y=today_changes.values,
                mode='lines',
                name="Today's Price",
                line=dict(color=colors['today'], width=3),
            ),
            row=1, col=1
        )
    
    # Removed yesterday's price trace
    
    # Add top matching patterns (limit to max_patterns)
    for i, match in enumerate(top_matches[:max_patterns]):
        if i < len(colors) - 1:  # Ensure we have a color for this match
            match_color = colors[f'match{i+1}']
            date_str = pd.Timestamp(match['date']).strftime('%Y-%m-%d')
            
            # Create name with or without score
            if show_scores:
                name = f"Match #{i+1}: {date_str} (Score: {match['similarity']:.4f})"
            else:
                name = f"Match #{i+1}: {date_str}"
                
            # Add group count if available
            if 'count' in match and match['count'] > 1:
                name += f" (+{match['count']-1} similar)"
            
            # Safely extract time strings for pattern data
            pattern_times = []
            for t in match['pattern'].index:
                try:
                    if hasattr(t, 'strftime'):
                        pattern_times.append(t.strftime('%H:%M'))
                    else:
                        # If not a time object, convert to string
                        pattern_times.append(str(t))
                except:
                    pattern_times.append('00:00')  # Fallback
            
            fig.add_trace(
                go.Scatter(
                    x=pattern_times,
                    y=match['pattern'].values,
                    mode='lines',
                    name=name,
                    line=dict(color=match_color, width=2),
                ),
                row=1, col=1
            )
    
    # Removed composite pattern trace
    
    # Process volume data
    if 'Volume' in data.columns:
        # Get historical volume data (excluding today)
        historical_data = data[data.index.date != today]
        historical_data['time_bucket'] = historical_data.index.strftime('%H:%M')
        historical_volume = historical_data.groupby('time_bucket')['Volume'].mean()
        
        # Today's volume data
        if not today_data.empty:
            today_data['time_bucket'] = today_data.index.strftime('%H:%M')
            today_volume = today_data.groupby('time_bucket')['Volume'].mean()
            
            # Removed yesterday's volume data section
            
            # Find max volume for normalization
            max_vol = max(historical_volume.max(), today_volume.max())
            
            # Create legend groups
            legend_groups = {
                'historical': False,
                'today': False
            }
            
            # Plot historical volume
            norm_hist_vol = historical_volume / max_vol
            for time_bucket, volume in norm_hist_vol.items():
                fig.add_trace(
                    go.Bar(
                        x=[time_bucket],
                        y=[volume],
                        name='Average Volume',
                        marker_color='rgba(128, 128, 128, 0.3)',
                        showlegend=legend_groups['historical'] is False,
                        legendgroup='historical'
                    ),
                    row=2, col=1
                )
                legend_groups['historical'] = True
            
            # Plot today's volume
            norm_today_vol = today_volume / max_vol
            for time_bucket, volume in norm_today_vol.items():
                fig.add_trace(
                    go.Bar(
                        x=[time_bucket],
                        y=[volume],
                        name="Today's Volume",
                        marker_color='rgba(255, 99, 132, 0.5)',
                        showlegend=legend_groups['today'] is False,
                        legendgroup='today'
                    ),
                    row=2, col=1
                )
                legend_groups['today'] = True
            
            # Removed yesterday's volume plotting

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Pattern Matches',
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=700,
        template='plotly_dark',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Common x-axis settings
    hours = list(range(0, 24))
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        ticktext=[f'{h:02d}:00' for h in hours],
        tickvals=[f'{h:02d}:00' for h in hours],
        tickangle=45,
        zeroline=True,
        zerolinecolor='rgba(128,128,128,0.2)',
        row=1, col=1
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        ticktext=[f'{h:02d}:00' for h in hours],
        tickvals=[f'{h:02d}:00' for h in hours],
        tickangle=45,
        title_text="Time of Day (EST)",
        zeroline=True,
        zerolinecolor='rgba(128,128,128,0.2)',
        row=2, col=1
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="Price Change from Open (%)",
        gridcolor='rgba(128,128,128,0.2)',
        gridwidth=1,
        showgrid=True,
        zeroline=True,
        zerolinecolor='rgba(255,255,255,0.4)',
        zerolinewidth=2,
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Relative Volume",
        gridcolor='rgba(128,128,128,0.2)',
        gridwidth=1,
        showgrid=True,
        range=[0, 1],
        row=2, col=1
    )
    
    return fig

def create_match_details_card(match, data):
    """Create details card for a matching pattern"""
    date_str = pd.Timestamp(match['date']).strftime('%Y-%m-%d')
    match_data = data[data.index.date == match['date']]
    
    if match_data.empty:
        return None
    
    # Calculate day stats
    day_open = match_data['Open'].iloc[0]
    day_close = match_data['Close'].iloc[-1]
    day_return = (day_close - day_open) / day_open * 100
    
    high_point = match_data['High'].max()
    high_return = (high_point - day_open) / day_open * 100
    high_time = match_data.loc[match_data['High'] == high_point].index[0].strftime('%H:%M')
    
    low_point = match_data['Low'].min()
    low_return = (low_point - day_open) / day_open * 100
    low_time = match_data.loc[match_data['Low'] == low_point].index[0].strftime('%H:%M')
    
    # Calculate future performance
    next_day = pd.Timestamp(match['date']) + pd.Timedelta(days=1)
    next_day_data = data[data.index.date == next_day.date()]
    
    if not next_day_data.empty:
        next_day_open = next_day_data['Open'].iloc[0]
        next_day_close = next_day_data['Close'].iloc[-1]
        next_day_return = (next_day_close - next_day_open) / next_day_open * 100
    else:
        next_day_return = None
    
    # Create the card
    expanded = match.get('count', 1) > 1  # Auto-expand if it represents a group
    with st.expander(f"Details for {date_str} (Similarity: {match['similarity']:.4f})", expanded=expanded):
        # Show similar patterns if this is a group
        if 'similar_dates' in match and match['similar_dates']:
            similar_dates_str = ', '.join([pd.Timestamp(d).strftime('%Y-%m-%d') for d in match['similar_dates']])
            st.info(f"This pattern is similar to {len(match['similar_dates'])} other days: {similar_dates_str}")
            
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Day Close", f"{day_return:.2f}%")
            st.markdown(f"**Volume**: {match_data['Volume'].sum():,.0f}")
        
        with col2:
            st.metric("Day High", f"{high_return:.2f}% at {high_time}")
            if next_day_return is not None:
                st.metric("Next Day", f"{next_day_return:.2f}%")
        
        with col3:
            st.metric("Day Low", f"{low_return:.2f}% at {low_time}")
            
            # Calculate when the pattern started diverging from today (if applicable)
            today = pd.Timestamp.now(tz='US/Eastern').date()
            today_data = data[data.index.date == today]
            
            if not today_data.empty and len(today_data) >= 5:
                today_open = today_data['Close'].iloc[0]
                today_pattern = ((today_data['Close'] - today_open) / today_open) * 100
                today_pattern.index = today_data.index.time
                
                # Find time of maximum divergence
                max_diff = 0
                max_diff_time = None
                
                try:
                    # Ensure we're working with compatible index types
                    pattern_times = set(match['pattern'].index)
                    today_times = set(today_pattern.index)
                    
                    common_times = pattern_times.intersection(today_times)
                    
                    for time in common_times:
                        diff = abs(today_pattern[time] - match['pattern'][time])
                        if diff > max_diff:
                            max_diff = diff
                            max_diff_time = time
                    
                    if max_diff_time:
                        # Format the time string appropriately
                        if hasattr(max_diff_time, 'strftime'):
                            time_str = max_diff_time.strftime('%H:%M')
                        else:
                            time_str = str(max_diff_time)
                        st.markdown(f"**Max Divergence**: {max_diff:.2f}% at {time_str}")
                except Exception as e:
                    st.info("Could not calculate maximum divergence")

def analyze_pattern_distribution(top_matches):
    """Analyze the distribution of patterns - up vs down trends"""
    if not top_matches:
        return None
        
    # Count trends
    up_count = sum(1 for m in top_matches if m['end_value'] > 0)
    down_count = sum(1 for m in top_matches if m['end_value'] <= 0)
    
    # Calculate average ending values by trend
    up_values = [m['end_value'] for m in top_matches if m['end_value'] > 0]
    down_values = [m['end_value'] for m in top_matches if m['end_value'] <= 0]
    
    avg_up = sum(up_values) / len(up_values) if up_values else 0
    avg_down = sum(down_values) / len(down_values) if down_values else 0
    
    # Create distribution chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Upward', 'Downward'],
        y=[up_count, down_count],
        marker_color=['green', 'red'],
        text=[f"{up_count} patterns<br>Avg: +{avg_up:.2f}%", 
              f"{down_count} patterns<br>Avg: {avg_down:.2f}%"],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Pattern Distribution",
        xaxis_title="End of Day Direction",
        yaxis_title="Number of Patterns",
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    return fig

# Main UI layout
st.sidebar.header("Pattern Finder Settings")
symbol = st.sidebar.text_input("Enter Crypto Symbol:", value="BTC-USD").upper()

# Quick select buttons
st.sidebar.subheader("Quick Select")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("BTC-USD"): symbol = "BTC-USD"
    if st.button("SOL-USD"): symbol = "SOL-USD"
with col2:
    if st.button("ETH-USD"): symbol = "ETH-USD"
    if st.button("DOGE-USD"): symbol = "DOGE-USD"

# Interval selection
interval = st.sidebar.select_slider(
    "Select Interval:",
    options=["5m", "15m", "30m", "1h"],
    value="15m"
)

# Number of matches to display
all_matches = st.sidebar.slider(
    "Calculate Top Matches:",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
    help="Total number of matches to find (you can display fewer)"
)

display_matches = st.sidebar.slider(
    "Display on Chart:",
    min_value=1,
    max_value=min(10, all_matches),
    value=min(4, all_matches),
    step=1,
    help="Number of matches to show on the chart"
)

# Display options
st.sidebar.subheader("Display Options")
# Removed the show_composite checkbox
recent_bias = st.sidebar.checkbox("Bias Toward Recent Patterns", value=False)
show_scores = st.sidebar.checkbox("Show Similarity Scores", value=True)

# Group similar patterns
group_patterns = st.sidebar.checkbox("Group Similar Patterns", value=False, 
                                   help="Group patterns that are very similar to reduce redundancy")

# If user wants to group similar patterns, add threshold slider
similarity_threshold = 0.8
if group_patterns:
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold:",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Patterns with similarity below this threshold will be grouped together"
    )

# Main app logic
try:
    # Fetch data
    with st.spinner(f"Fetching data for {symbol}..."):
        data = fetch_crypto_data(symbol, interval=interval)
    
    if data is not None:
        # Calculate patterns
        avg_pattern, all_patterns = calculate_daily_pattern(data, num_days=60)
        
        # Find top matching patterns
        today = pd.Timestamp.now(tz='US/Eastern').date()
        today_data = data[data.index.date == today]
        
        # Find more matches than we'll display to allow for grouping
        matches_to_find = max(all_matches, 5)
        all_top_matches = find_top_similar_days(today_data, all_patterns, 
                                             num_matches=matches_to_find,
                                             recent_bias=recent_bias)
        
        # Group similar patterns if requested
        if group_patterns and all_top_matches:
            top_matches = group_similar_patterns(all_top_matches, threshold=similarity_threshold)
            # Limit to requested number after grouping
            top_matches = top_matches[:all_matches]
        else:
            top_matches = all_top_matches[:all_matches]
        
        if top_matches:
            # Display the chart - removed the composite pattern parameter
            fig = plot_pattern_matches(
                data, 
                symbol, 
                top_matches,
                show_scores=show_scores,
                max_patterns=display_matches
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show pattern distribution
            if len(top_matches) >= 3:
                dist_fig = analyze_pattern_distribution(top_matches)
                if dist_fig:
                    st.plotly_chart(dist_fig, use_container_width=True)
            
            # Display match details
            st.subheader("Pattern Match Details")
            
            # Create a card for each match
            for match in top_matches[:display_matches]:
                create_match_details_card(match, data)
                
            # Display prediction based on patterns
            st.subheader("Pattern-Based Forecast")
            
            # Calculate average ending for matching patterns
            avg_ending = 0
            count = 0
            
            for match in top_matches:
                match_data = data[data.index.date == match['date']]
                if not match_data.empty:
                    day_open = match_data['Open'].iloc[0]
                    day_close = match_data['Close'].iloc[-1]
                    day_return = (day_close - day_open) / day_open * 100
                    avg_ending += day_return * match.get('count', 1)  # Weight by group size
                    count += match.get('count', 1)
            
            if count > 0:
                avg_ending /= count
                
                # Simple direction prediction
                direction = "upward" if avg_ending > 0 else "downward"
                
                # Calculate confidence based on pattern agreement
                up_patterns = sum(1 for m in top_matches if m['end_value'] > 0)
                down_patterns = sum(1 for m in top_matches if m['end_value'] <= 0)
                total_patterns = up_patterns + down_patterns
                
                if total_patterns > 0:
                    if up_patterns > down_patterns:
                        confidence = (up_patterns / total_patterns) * 100
                    else:
                        confidence = (down_patterns / total_patterns) * 100
                else:
                    confidence = 50
                
                st.markdown(f"""
                Based on the {len(top_matches)} most similar historical patterns (representing {count} trading days):
                
                - The average day ended **{avg_ending:.2f}%** from the open
                - **{up_patterns}** patterns ended positive, **{down_patterns}** ended negative
                - This suggests a potential **{direction}** movement with **{confidence:.0f}%** pattern agreement
                - Strength of signal: **{"Strong" if confidence > 75 else "Moderate" if confidence > 60 else "Weak"}**
                
                *Note: This is a pattern-based observation, not financial advice.*
                """)
                
                # Display forecast horizon
                hours_left = 0
                if not today_data.empty:
                    try:
                        # Make sure we're working with a datetime object
                        if isinstance(today_data.index[-1], pd.Timestamp):
                            last_time = today_data.index[-1]
                            end_of_day = pd.Timestamp.combine(today, datetime.time(23, 59))
                            hours_left = (end_of_day - last_time).total_seconds() / 3600
                            
                            if hours_left > 1:
                                st.info(f"Approximately {hours_left:.1f} hours remaining in today's trading day.")
                    except Exception as e:
                        st.error(f"Could not calculate remaining time: {str(e)}")
        else:
            st.warning("Not enough data to find pattern matches. Try a different interval or check back later in the day.")
            
    else:
        st.error("No data available for the selected symbol.")
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}")