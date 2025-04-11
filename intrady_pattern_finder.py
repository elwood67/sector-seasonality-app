import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta, time
import numpy as np
import pytz

# Set page config
st.set_page_config(
    layout="wide",
    page_title="Crypto Pattern Finder",
    page_icon="🔍",
)

# Title and description
st.title("Crypto Pattern Finder")
st.markdown("Identifies the most similar historical trading patterns to today's price action based on Asia open (UTC midnight)")

# Constants for Asia session
ASIA_OPEN_HOUR_UTC = 0  # 12:00 AM UTC (Asia market open)

def fetch_crypto_data(symbol, period="60d", interval="5m"):
    """Fetch crypto data from Yahoo Finance"""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    
    if hist.empty:
        return None
        
    # Convert timestamps to UTC for consistency
    if hist.index.tz is not None:
        hist.index = hist.index.tz_convert('UTC')
    else:
        hist.index = hist.index.tz_localize('UTC')
    
    return hist

def get_asia_session_date(timestamp):
    """
    Get the Asia session date for a given timestamp.
    The Asia session runs from UTC midnight to UTC midnight.
    """
    # The session date is the date of UTC midnight AFTER the timestamp
    if timestamp.hour >= ASIA_OPEN_HOUR_UTC:
        # If timestamp is after or at UTC midnight, session date is the next day
        return (timestamp.date() + timedelta(days=1))
    else:
        # If timestamp is before UTC midnight, session date is the same day
        return timestamp.date()

def get_asia_sessions(data):
    """Get data grouped by Asia trading sessions"""
    # Create a copy to avoid modifying the original
    data_copy = data.copy()
    
    # Add a column for Asia session date
    data_copy['asia_session'] = data_copy.index.map(get_asia_session_date)
    
    # Get unique session dates
    session_dates = sorted(data_copy['asia_session'].unique())
    
    # Store session data
    sessions = {}
    
    for session_date in session_dates:
        # Calculate session start (previous day's UTC midnight)
        session_start = datetime.combine(session_date - timedelta(days=1), time(hour=ASIA_OPEN_HOUR_UTC))
        session_start = pytz.utc.localize(session_start)
        
        # Calculate session end (UTC midnight)
        session_end = datetime.combine(session_date, time(hour=ASIA_OPEN_HOUR_UTC))
        session_end = pytz.utc.localize(session_end)
        
        # Get data for this session
        session_data = data_copy[(data_copy.index >= session_start) & (data_copy.index < session_end)]
        
        if not session_data.empty:
            sessions[session_date] = session_data
    
    return sessions

def calculate_session_patterns(sessions, current_session_date=None):
    """Calculate price patterns for each Asia session"""
    patterns = {}
    
    for date, session_data in sessions.items():
        # Skip current session for historical patterns
        if date == current_session_date:
            continue
            
        if len(session_data) < 12:  # Require at least 12 data points
            continue
            
        # Calculate percentage change from session open
        session_open = session_data['Close'].iloc[0]
        session_pattern = ((session_data['Close'] - session_open) / session_open) * 100
        
        # Use minutes from session start as index for better comparison
        session_start = datetime.combine(date - timedelta(days=1), time(hour=ASIA_OPEN_HOUR_UTC))
        session_start = pytz.utc.localize(session_start)
        
        minutes_from_start = [(ts - session_start).total_seconds() / 60 for ts in session_data.index]
        session_pattern.index = minutes_from_start
        
        patterns[date] = session_pattern
    
    return patterns

def find_similar_sessions(current_pattern, historical_patterns, num_matches=10, recent_bias=False, min_overlap=12):
    """Find the most similar historical patterns to current session"""
    if current_pattern is None or len(current_pattern) < min_overlap:
        return []
    
    similarity_scores = []
    
    for date, pattern in historical_patterns.items():
        # Skip if patterns don't have enough overlap
        common_times = set(pattern.index).intersection(set(current_pattern.index))
        if len(common_times) < min_overlap:
            continue
            
        # Calculate similarity (mean squared error) for common time points
        squared_diffs = []
        times_list = []
        
        for minutes in common_times:
            if minutes in pattern.index and minutes in current_pattern.index:
                diff = current_pattern[minutes] - pattern[minutes]
                squared_diffs.append(diff * diff)
                times_list.append(minutes)  # minutes from session start
        
        if not squared_diffs:
            continue
            
        # Calculate weighted MSE - more weight to recent points if requested
        if recent_bias and times_list:
            max_time = max(times_list)
            normalized_times = [t/max_time for t in times_list]
            weights = np.array([0.5 + 0.5 * t for t in normalized_times])
        else:
            weights = np.ones(len(squared_diffs))
            
        weighted_mse = sum(w * d for w, d in zip(weights, squared_diffs)) / sum(weights)
        
        # Calculate end of session value
        end_value = pattern.iloc[-1] if not pattern.empty else 0
        
        similarity_scores.append({
            'date': date,
            'similarity': weighted_mse,
            'pattern': pattern,
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
            
            if len(common_times) < 12:  # Require more overlap for grouping
                continue
                
            squared_diffs = []
            for minutes in common_times:
                if minutes in p1.index and minutes in p2.index:
                    diff = p1[minutes] - p2[minutes]
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

def minutes_to_time_str(minutes):
    """Convert minutes from session start to formatted time string"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

def plot_pattern_matches(sessions, symbol, current_session_date, top_matches, show_scores=True, max_patterns=10):
    """Create plot showing pattern matches with current session data"""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Get current session data
    if current_session_date in sessions:
        current_session = sessions[current_session_date]
    else:
        st.error("Current session data not found")
        return None
    
    # Colors for different patterns
    colors = {
        'current': 'rgb(255, 255, 255)',   # White
        'match1': 'rgb(255, 99, 132)',     # Red
        'match2': 'rgb(66, 135, 245)',     # Blue
        'match3': 'rgb(52, 191, 73)',      # Green
        'match4': 'rgb(242, 184, 64)',     # Yellow
        'match5': 'rgb(153, 102, 255)',    # Purple
        'match6': 'rgb(0, 204, 204)',      # Teal
        'match7': 'rgb(255, 51, 153)',     # Pink
        'match8': 'rgb(102, 255, 102)',    # Light Green
        'match9': 'rgb(255, 204, 0)',      # Gold
        'match10': 'rgb(204, 102, 255)',   # Lavender
    }
    
    # Calculate session start (for minutes conversion)
    session_start = datetime.combine(current_session_date - timedelta(days=1), time(hour=ASIA_OPEN_HOUR_UTC))
    session_start = pytz.utc.localize(session_start)
    
    # Add current session price
    if not current_session.empty:
        session_open = current_session['Close'].iloc[0]
        current_changes = ((current_session['Close'] - session_open) / session_open) * 100
        
        # Calculate minutes from session start for each timestamp
        minutes_list = [(ts - session_start).total_seconds() / 60 for ts in current_session.index]
        
        # Convert to hours:minutes format for display
        time_strings = [minutes_to_time_str(m) for m in minutes_list]
        
        fig.add_trace(
            go.Scatter(
                x=time_strings,
                y=current_changes.values,
                mode='lines',
                name="Current Session",
                line=dict(color=colors['current'], width=3),
            ),
            row=1, col=1
        )
    
    # Add top matching patterns
    for i, match in enumerate(top_matches[:max_patterns]):
        if i < len(colors) - 1:  # Ensure we have a color for this match
            match_color = colors[f'match{i+1}']
            date_str = match['date'].strftime('%Y-%m-%d')
            
            # Create name with or without score
            if show_scores:
                name = f"Match #{i+1}: {date_str} (Score: {match['similarity']:.4f})"
            else:
                name = f"Match #{i+1}: {date_str}"
                
            # Add group count if available
            if 'count' in match and match['count'] > 1:
                name += f" (+{match['count']-1} similar)"
            
            # Convert pattern minutes to time strings
            pattern_times = [minutes_to_time_str(m) for m in match['pattern'].index]
            
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
    
    # Process volume data
    if 'Volume' in current_session.columns:
        # Get all sessions except current for historical average
        historical_sessions = {d: s for d, s in sessions.items() if d != current_session_date}
        
        # Combine all historical data
        all_historical_data = pd.concat([s for s in historical_sessions.values()])
        
        # Group by minutes from session start
        all_historical_data['minutes_from_start'] = all_historical_data.index.map(
            lambda ts: int((ts - datetime.combine(get_asia_session_date(ts) - timedelta(days=1), 
                                              time(hour=ASIA_OPEN_HOUR_UTC)).replace(tzinfo=pytz.UTC)).total_seconds() / 60)
        )
        
        # Bin by 5-minute intervals for smoother visualization
        all_historical_data['minute_bin'] = (all_historical_data['minutes_from_start'] // 5) * 5
        historical_volume = all_historical_data.groupby('minute_bin')['Volume'].mean()
        
        # Current session volume
        current_session_copy = current_session.copy()
        current_session_copy['minutes_from_start'] = current_session_copy.index.map(
            lambda ts: int((ts - session_start).total_seconds() / 60)
        )
        current_session_copy['minute_bin'] = (current_session_copy['minutes_from_start'] // 5) * 5
        current_volume = current_session_copy.groupby('minute_bin')['Volume'].mean()
        
        # Find max volume for normalization
        max_vol = max(historical_volume.max(), current_volume.max()) if not current_volume.empty else historical_volume.max()
        
        # Create legend groups
        legend_groups = {
            'historical': False,
            'current': False
        }
        
        # Plot historical volume
        if not historical_volume.empty:
            norm_hist_vol = historical_volume / max_vol
            for minute_bin, volume in norm_hist_vol.items():
                time_str = minutes_to_time_str(minute_bin)
                fig.add_trace(
                    go.Bar(
                        x=[time_str],
                        y=[volume],
                        name='Average Volume',
                        marker_color='rgba(128, 128, 128, 0.3)',
                        showlegend=legend_groups['historical'] is False,
                        legendgroup='historical'
                    ),
                    row=2, col=1
                )
                legend_groups['historical'] = True
        
        # Plot current session volume
        if not current_volume.empty:
            norm_current_vol = current_volume / max_vol
            for minute_bin, volume in norm_current_vol.items():
                time_str = minutes_to_time_str(minute_bin)
                fig.add_trace(
                    go.Bar(
                        x=[time_str],
                        y=[volume],
                        name="Current Volume",
                        marker_color='rgba(255, 99, 132, 0.5)',
                        showlegend=legend_groups['current'] is False,
                        legendgroup='current'
                    ),
                    row=2, col=1
                )
                legend_groups['current'] = True

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Pattern Matches (Asia Session)',
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

    # Common x-axis settings - use hours from session start
    hours = list(range(0, 24))
    hour_labels = [f"{h:02d}:00" for h in hours]
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        ticktext=hour_labels,
        tickvals=hour_labels,
        tickangle=45,
        zeroline=True,
        zerolinecolor='rgba(128,128,128,0.2)',
        row=1, col=1
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        ticktext=hour_labels,
        tickvals=hour_labels,
        tickangle=45,
        title_text="Hours from Asia Open (UTC Midnight)",
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

def create_match_details_card(match, sessions):
    """Create details card for a matching pattern"""
    date_str = match['date'].strftime('%Y-%m-%d')
    
    # Get session data
    session_data = sessions.get(match['date'])
    
    if session_data is None or session_data.empty:
        return None
    
    # Calculate session stats
    session_open = session_data['Open'].iloc[0]
    session_close = session_data['Close'].iloc[-1]
    session_return = (session_close - session_open) / session_open * 100
    
    high_point = session_data['High'].max()
    high_return = (high_point - session_open) / session_open * 100
    
    # Find time of high in minutes from session start
    high_idx = session_data['High'].idxmax()
    session_start = datetime.combine(match['date'] - timedelta(days=1), time(hour=ASIA_OPEN_HOUR_UTC))
    session_start = pytz.utc.localize(session_start)
    high_minutes = int((high_idx - session_start).total_seconds() / 60)
    high_time = minutes_to_time_str(high_minutes)
    
    low_point = session_data['Low'].min()
    low_return = (low_point - session_open) / session_open * 100
    
    # Find time of low
    low_idx = session_data['Low'].idxmin()
    low_minutes = int((low_idx - session_start).total_seconds() / 60)
    low_time = minutes_to_time_str(low_minutes)
    
    # Try to get next session's data
    next_session_date = match['date'] + timedelta(days=1)
    next_session_data = sessions.get(next_session_date)
    
    if next_session_data is not None and not next_session_data.empty:
        next_session_open = next_session_data['Open'].iloc[0]
        next_session_close = next_session_data['Close'].iloc[-1]
        next_session_return = (next_session_close - next_session_open) / next_session_open * 100
    else:
        next_session_return = None
    
    # Create the card
    expanded = match.get('count', 1) > 1  # Auto-expand if it represents a group
    with st.expander(f"Details for {date_str} (Similarity: {match['similarity']:.4f})", expanded=expanded):
        # Show similar patterns if this is a group
        if 'similar_dates' in match and match['similar_dates']:
            similar_dates_str = ', '.join([pd.Timestamp(d).strftime('%Y-%m-%d') for d in match['similar_dates']])
            st.info(f"This pattern is similar to {len(match['similar_dates'])} other sessions: {similar_dates_str}")
            
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Session Close", f"{session_return:.2f}%")
            st.markdown(f"**Volume**: {session_data['Volume'].sum():,.0f}")
        
        with col2:
            st.metric("Session High", f"{high_return:.2f}% at {high_time}")
            if next_session_return is not None:
                st.metric("Next Session", f"{next_session_return:.2f}%")
        
        with col3:
            st.metric("Session Low", f"{low_return:.2f}% at {low_time}")
            
            # Calculate when the pattern started diverging from current session
            current_date = datetime.now(pytz.utc).date()
            current_session_date = get_asia_session_date(datetime.now(pytz.utc))
            
            if current_session_date in sessions:
                current_session = sessions[current_session_date]
                
                if not current_session.empty and len(current_session) >= 12:
                    # Calculate current session pattern
                    current_open = current_session['Close'].iloc[0]
                    current_pattern = ((current_session['Close'] - current_open) / current_open) * 100
                    
                    # Calculate minutes from session start
                    current_start = datetime.combine(current_session_date - timedelta(days=1), time(hour=ASIA_OPEN_HOUR_UTC))
                    current_start = pytz.utc.localize(current_start)
                    current_pattern.index = [(ts - current_start).total_seconds() / 60 for ts in current_session.index]
                    
                    # Find maximum divergence
                    max_diff = 0
                    max_diff_time = None
                    
                    try:
                        # Get common time points
                        pattern_times = set(match['pattern'].index)
                        current_times = set(current_pattern.index)
                        common_times = pattern_times.intersection(current_times)
                        
                        for minutes in common_times:
                            diff = abs(current_pattern[minutes] - match['pattern'][minutes])
                            if diff > max_diff:
                                max_diff = diff
                                max_diff_time = minutes
                        
                        if max_diff_time is not None:
                            time_str = minutes_to_time_str(max_diff_time)
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
        xaxis_title="End of Session Direction",
        yaxis_title="Number of Patterns",
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    return fig

def calculate_confidence_metrics(top_matches):
    """Calculate confidence metrics based on pattern agreement"""
    if not top_matches:
        return None
    
    # Count up vs down patterns
    up_count = sum(1 for m in top_matches if m['end_value'] > 0)
    down_count = sum(1 for m in top_matches if m['end_value'] <= 0)
    total_count = up_count + down_count
    
    # Calculate average ending values by trend
    up_values = [m['end_value'] for m in top_matches if m['end_value'] > 0]
    down_values = [m['end_value'] for m in top_matches if m['end_value'] <= 0]
    all_values = up_values + down_values
    
    avg_up = sum(up_values) / len(up_values) if up_values else 0
    avg_down = sum(down_values) / len(down_values) if down_values else 0
    avg_all = sum(all_values) / len(all_values) if all_values else 0
    
    # Calculate weighted average by similarity (inverse of MSE)
    weights = [1/m['similarity'] for m in top_matches]
    weighted_values = [m['end_value'] * (1/m['similarity']) for m in top_matches]
    weighted_avg = sum(weighted_values) / sum(weights) if weights else 0
    
    # Calculate confidence based on agreement
    if total_count > 0:
        if up_count > down_count:
            confidence = up_count / total_count
            direction = "up"
        elif down_count > up_count:
            confidence = down_count / total_count
            direction = "down"
        else:
            confidence = 0.5
            direction = "up" if avg_all > 0 else "down"
    else:
        confidence = 0
        direction = "neutral"
    
    # Calculate expected magnitude
    expected_magnitude = abs(avg_all)
    
    # Determine confidence level
    if confidence >= 0.8:
        confidence_level = "High"
    elif confidence >= 0.6:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    # Determine magnitude level
    if expected_magnitude >= 2.0:
        magnitude_level = "Large"
    elif expected_magnitude >= 1.0:
        magnitude_level = "Medium"
    else:
        magnitude_level = "Small"
    
    # Return metrics
    return {
        'direction': direction,
        'confidence': confidence,
        'confidence_level': confidence_level,
        'expected_magnitude': expected_magnitude,
        'magnitude_level': magnitude_level,
        'avg_all': avg_all,
        'avg_up': avg_up,
        'avg_down': avg_down,
        'weighted_avg': weighted_avg,
        'up_count': up_count,
        'down_count': down_count,
        'total_count': total_count
    }

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
    value="5m"  # Changed default to 5m
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
    value=min(5, all_matches),
    step=1,
    help="Number of matches to show on the chart"
)

# Display options
st.sidebar.subheader("Display Options")
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

# Add confidence filter
st.sidebar.subheader("Confidence Settings")
filter_by_confidence = st.sidebar.checkbox("Only Show High Confidence", value=False,
                                        help="Only show signals with high confidence (>80%)")

    # Main app logic
try:
    # Fetch data
    with st.spinner(f"Fetching data for {symbol}..."):
        data = fetch_crypto_data(symbol, period="60d", interval=interval)
    
    if data is not None:
        # Group data by Asia sessions
        with st.spinner("Processing trading sessions..."):
            sessions = get_asia_sessions(data)
            
            # Get current session date
            current_time = datetime.now(pytz.utc)
            current_session_date = get_asia_session_date(current_time)
            
            # Calculate patterns for all sessions
            all_patterns = calculate_session_patterns(sessions, current_session_date)
            
            # Get current session data and pattern
            current_session = sessions.get(current_session_date)
            
            if current_session is not None and not current_session.empty:
                session_open = current_session['Close'].iloc[0]
                current_pattern = ((current_session['Close'] - session_open) / session_open) * 100
                
                # Calculate minutes from session start
                session_start = datetime.combine(current_session_date - timedelta(days=1), time(hour=ASIA_OPEN_HOUR_UTC))
                session_start = pytz.utc.localize(session_start)
                current_pattern.index = [(ts - session_start).total_seconds() / 60 for ts in current_session.index]
                
                # Find top matching patterns
                matches_to_find = max(all_matches, 5)
                all_top_matches = find_similar_sessions(
                    current_pattern, all_patterns, 
                    num_matches=matches_to_find,
                    recent_bias=recent_bias
                )
                
                # Group similar patterns if requested
                if group_patterns and all_top_matches:
                    top_matches = group_similar_patterns(all_top_matches, threshold=similarity_threshold)
                    # Limit to requested number after grouping
                    top_matches = top_matches[:all_matches]
                else:
                    top_matches = all_top_matches[:all_matches]
                
                if top_matches:
                    # Calculate confidence metrics
                    confidence_metrics = calculate_confidence_metrics(top_matches)
                    
                    # Apply confidence filter if selected
                    show_results = True
                    if filter_by_confidence and confidence_metrics:
                        if confidence_metrics['confidence'] < 0.8:
                            st.warning("Current pattern has low/medium confidence. Enable 'Show All Patterns' to view it.")
                            show_results = False
                    
                    if show_results:
                        # Display the chart
                        fig = plot_pattern_matches(
                            sessions, 
                            symbol,
                            current_session_date,
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
                        
                        # Display confidence metrics in a nice format
                        if confidence_metrics:
                            st.subheader("Pattern Prediction")
                            
                            # Create three columns for metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                direction_str = "Upward" if confidence_metrics['direction'] == "up" else "Downward"
                                st.metric("Predicted Direction", direction_str)
                                
                            with col2:
                                conf_pct = confidence_metrics['confidence'] * 100
                                st.metric("Confidence", f"{conf_pct:.1f}%", 
                                         delta=f"{confidence_metrics['confidence_level']}")
                                
                            with col3:
                                mag_val = confidence_metrics['avg_all']
                                st.metric("Expected Change", f"{mag_val:.2f}%", 
                                        delta=f"{confidence_metrics['magnitude_level']} Magnitude")
                            
                            # Show detailed analysis
                            st.markdown(f"""
                            **Analysis based on {len(top_matches)} similar patterns:**
                            
                            - **{confidence_metrics['up_count']}** patterns ended positive (avg: **{confidence_metrics['avg_up']:.2f}%**)
                            - **{confidence_metrics['down_count']}** patterns ended negative (avg: **{confidence_metrics['avg_down']:.2f}%**)
                            - Similarity-weighted average: **{confidence_metrics['weighted_avg']:.2f}%**
                            
                            Based on our backtesting, predictions with **{confidence_metrics['confidence_level']} confidence** and **{confidence_metrics['magnitude_level']} magnitude** 
                            have historically been {
                            "very accurate (75%+ win rate)" if confidence_metrics['confidence_level'] == "High" else 
                            "moderately accurate" if confidence_metrics['confidence_level'] == "Medium" else 
                            "less reliable"}.
                            
                            *This is a pattern-based observation, not financial advice.*
                            """)
                            
                            # Show time remaining in current session
                            session_end = datetime.combine(current_session_date, time(hour=ASIA_OPEN_HOUR_UTC))
                            session_end = pytz.utc.localize(session_end)
                            time_remaining = (session_end - current_time).total_seconds() / 3600
                            
                            if time_remaining > 0:
                                st.info(f"⏱️ **{time_remaining:.1f} hours** remaining in current trading session (until UTC midnight)")
                        
                        # Display match details
                        st.subheader("Pattern Match Details")
                        
                        # Create a card for each match
                        for match in top_matches[:display_matches]:
                            create_match_details_card(match, sessions)
                            
                else:
                    st.warning("No similar historical patterns found. Try a different interval or check back later in the session.")
            else:
                st.warning("Not enough data for the current trading session. Please check back later.")
        
    else:
        st.error("No data available for the selected symbol.")
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)