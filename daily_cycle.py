import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# Set page config
st.set_page_config(
    layout="wide",
    page_title="Crypto Intraday Patterns",
    page_icon="📈"
)

# Title and description
st.title("Crypto Intraday Pattern Analysis")
st.markdown("Analyze multi-timeframe patterns and volume profiles for crypto assets")

def fetch_crypto_data(symbol, period="60d", interval="15m"):
    """Fetch crypto data from Yahoo Finance"""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    
    if hist.empty:
        return None
        
    # Convert timestamps to EST
    hist.index = hist.index.tz_convert('US/Eastern')
    
    return hist

def calculate_daily_pattern(data, num_days=30):
    """Calculate intraday pattern using only complete days"""
    # Get today's date to exclude it from historical patterns
    today = pd.Timestamp.now(tz='US/Eastern').date()
    
    # Get the most recent days
    end_date = data.index.max()
    start_date = end_date - pd.Timedelta(days=num_days)
    filtered_data = data[data.index >= start_date].copy()
    
    # Create empty DataFrame for patterns
    all_patterns = pd.DataFrame()
    
    # Get unique dates using pandas proper
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
            day_pattern.index = day_data.index.time
            all_patterns[date] = day_pattern
    
    # Calculate average pattern
    avg_pattern = all_patterns.mean(axis=1)
    
    return avg_pattern, all_patterns

def find_most_similar_day(today_data, historical_patterns):
    """Find the historical day that most closely matches today's pattern"""
    if today_data.empty or len(today_data) < 5:
        return None, float('inf')
    
    # Calculate today's pattern
    today_open = today_data['Close'].iloc[0]
    today_pattern = ((today_data['Close'] - today_open) / today_open) * 100
    today_pattern.index = today_data.index.time
    
    # Find the most similar historical day
    best_similarity = float('inf')
    most_similar_date = None
    
    for date, pattern in historical_patterns.items():
        # Skip if patterns don't have enough overlap
        common_times = set(pattern.index).intersection(set(today_pattern.index))
        if len(common_times) < 5:
            continue
            
        # Calculate similarity (mean squared error) for common time points
        similarity = 0
        count = 0
        
        for time in common_times:
            if time in pattern.index and time in today_pattern.index:
                diff = today_pattern[time] - pattern[time]
                similarity += diff * diff
                count += 1
            
        if count > 0:
            avg_similarity = similarity / count
            
            if avg_similarity < best_similarity:
                best_similarity = avg_similarity
                most_similar_date = date
    
    return most_similar_date, best_similarity

def plot_multi_timeframe_patterns(data, symbol, timeframes, show_patterns, custom_date=None, enable_custom_date=False):
    """Create plot showing patterns across multiple timeframes with volume"""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Colors for different timeframes
    colors = {
        5: 'rgb(255, 99, 132)',    # Red
        10: 'rgb(66, 135, 245)',   # Blue
        20: 'rgb(52, 191, 73)',    # Green
        30: 'rgb(242, 184, 64)',   # Yellow
        60: 'rgb(153, 102, 255)',  # Purple
        'custom': 'rgb(0, 255, 255)'  # Cyan
    }
    
    # Today's date for finding most similar pattern
    today = pd.Timestamp.now(tz='US/Eastern').date()
    today_data = data[data.index.date == today]
    
    # Store pattern data for similarity comparison
    all_timeframe_patterns = {}
    all_pattern_collections = {}
    
    # Calculate and plot patterns for each timeframe
    for days in timeframes:
        if show_patterns[days]:
            avg_pattern, all_patterns = calculate_daily_pattern(data, num_days=days)
            all_timeframe_patterns[days] = avg_pattern
            all_pattern_collections[days] = all_patterns
            
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in avg_pattern.index],
                    y=avg_pattern.values,
                    mode='lines+markers',
                    name=f'{days}-Day Pattern',
                    line=dict(color=colors[days], width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # Plot individual day patterns with low opacity
            for date, pattern in all_patterns.items():
                date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
                fig.add_trace(
                    go.Scatter(
                        x=[t.strftime('%H:%M') for t in pattern.index],
                        y=pattern.values,
                        mode='lines',
                        name=f'{days}-Day - {date_str}',
                        line=dict(color=colors[days], width=1),
                        opacity=0.15,  # Low opacity
                        showlegend=False
                    ),
                    row=1, col=1
                )

    # Calculate and add composite pattern if enabled
    if show_patterns['composite'] and all_timeframe_patterns:
        composite_times = sorted(set().union(*[pattern.index for pattern in all_timeframe_patterns.values()]))
        composite_values = []

        for time in composite_times:
            values = [pattern[time] for pattern in all_timeframe_patterns.values() if time in pattern.index]
            composite_values.append(sum(values) / len(values))

        fig.add_trace(
            go.Scatter(
                x=[t.strftime('%H:%M') for t in composite_times],
                y=composite_values,
                mode='lines',
                name='Composite Pattern',
                line=dict(color='rgb(255, 255, 255)', width=4),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Find most similar historical day across all timeframes
    most_similar_date = None
    best_similarity = float('inf')
    best_timeframe = None
    
    # Only try to find similar pattern if we have today's data
    if not today_data.empty and len(today_data) >= 5:
        for days, patterns in all_pattern_collections.items():
            similar_date, similarity = find_most_similar_day(today_data, patterns)
            if similar_date and similarity < best_similarity:
                most_similar_date = similar_date
                best_similarity = similarity
                best_timeframe = days
        
        # Plot most similar historical day if found
        if most_similar_date and best_timeframe:
            pattern = all_pattern_collections[best_timeframe][most_similar_date]
            similar_date_str = pd.Timestamp(most_similar_date).strftime('%Y-%m-%d')
            
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in pattern.index],
                    y=pattern.values,
                    mode='lines',
                    name=f'Most Similar: {similar_date_str}',
                    line=dict(
                        color='rgb(255, 255, 0)',  # Yellow
                        width=2
                    ),
                ),
                row=1, col=1
            )
    
    # Add today's price
    if not today_data.empty:
        today_open = today_data['Close'].iloc[0]
        today_changes = ((today_data['Close'] - today_open) / today_open) * 100
        
        fig.add_trace(
            go.Scatter(
                x=[t.strftime('%H:%M') for t in today_data.index.time],
                y=today_changes.values,
                mode='lines',
                name="Today's Price",
                line=dict(color='white', width=2, dash='dash'),
            ),
            row=1, col=1
        )
    
    # Add custom date pattern if enabled
    if enable_custom_date and custom_date:
        custom_date_data = data[data.index.date == custom_date]
        
        if not custom_date_data.empty:
            custom_open = custom_date_data['Close'].iloc[0]
            custom_changes = ((custom_date_data['Close'] - custom_open) / custom_open) * 100
            
            date_str = custom_date.strftime('%Y-%m-%d')
            
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in custom_date_data.index.time],
                    y=custom_changes.values,
                    mode='lines',
                    name=f"Custom Date: {date_str}",
                    line=dict(color=colors['custom'], width=2),
                ),
                row=1, col=1
            )
    
    # Add yesterday's price
    yesterday = today - pd.Timedelta(days=1)
    yesterday_data = data[data.index.date == yesterday]
    
    # Display yesterday's price if available
    if not yesterday_data.empty:
        yesterday_open = yesterday_data['Close'].iloc[0]
        yesterday_changes = ((yesterday_data['Close'] - yesterday_open) / yesterday_open) * 100
        
        fig.add_trace(
            go.Scatter(
                x=[t.strftime('%H:%M') for t in yesterday_data.index.time],
                y=yesterday_changes.values,
                mode='lines',
                name="Yesterday's Price",
                line=dict(color='rgba(255, 165, 0, 0.9)', width=2, dash='dot'),
            ),
            row=1, col=1
        )
    
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
            
            # Yesterday's volume data
            yesterday_volume = None
            if not yesterday_data.empty:
                yesterday_data['time_bucket'] = yesterday_data.index.strftime('%H:%M')
                yesterday_volume = yesterday_data.groupby('time_bucket')['Volume'].mean()
            
            # Find max volume for normalization
            max_vol_values = [historical_volume.max(), today_volume.max()]
            if yesterday_volume is not None:
                max_vol_values.append(yesterday_volume.max())
                
            max_vol = max(max_vol_values)
            
            # Create legend groups to avoid duplicate entries
            legend_groups = {
                'historical': False,
                'today': False,
                'yesterday': False
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
            
            # Add yesterday's volume if available
            if yesterday_volume is not None:
                norm_yesterday_vol = yesterday_volume / max_vol
                for time_bucket, volume in norm_yesterday_vol.items():
                    fig.add_trace(
                        go.Bar(
                            x=[time_bucket],
                            y=[volume],
                            name="Yesterday's Volume",
                            marker_color='rgba(255, 165, 0, 0.5)',
                            showlegend=legend_groups['yesterday'] is False,
                            legendgroup='yesterday'
                        ),
                        row=2, col=1
                    )
                    legend_groups['yesterday'] = True

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Multi-Timeframe Patterns with Volume',
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=800,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            groupclick="toggleitem",
            itemsizing="constant"
        ),
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
    
    return fig, most_similar_date, best_timeframe, best_similarity, all_pattern_collections

# Sidebar controls
with st.sidebar:
    # Symbol selection
    symbol = st.text_input("Enter Crypto Symbol:", value="BTC-USD").upper()
    
    # Quick select buttons
    st.subheader("Quick Select")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("BTC-USD"): symbol = "BTC-USD"
        if st.button("SOL-USD"): symbol = "SOL-USD"
    with col2:
        if st.button("ETH-USD"): symbol = "ETH-USD"
        if st.button("DOGE-USD"): symbol = "DOGE-USD"
    
    # Interval selection
    interval = st.select_slider(
        "Select Interval:",
        options=["5m", "15m", "30m", "1h"],
        value="15m"
    )
    
    # Pattern visibility
    st.subheader("Show Patterns")
    show_patterns = {
        5: st.checkbox("5-Day Pattern", value=True),
        10: st.checkbox("10-Day Pattern", value=True),
        20: st.checkbox("20-Day Pattern", value=True),
        30: st.checkbox("30-Day Pattern", value=True),
        60: st.checkbox("60-Day Pattern", value=True),
        'composite': st.checkbox("Composite Pattern", value=True)
    }
    
    # Custom date pattern selector
    st.subheader("Custom Date Pattern")
    enable_custom_date = st.checkbox("Show Custom Date", value=False)
    
    min_date = pd.Timestamp.now(tz='US/Eastern').date() - pd.Timedelta(days=60)
    max_date = pd.Timestamp.now(tz='US/Eastern').date() - pd.Timedelta(days=1)
    
    custom_date = st.date_input(
        "Select Date:",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        disabled=not enable_custom_date
    )

# Main app logic
try:
    # Fetch data
    with st.spinner(f"Fetching data for {symbol}..."):
        data = fetch_crypto_data(symbol, interval=interval)
    
    if data is not None:
        # Create and display plot
        timeframes = [days for days in [5, 10, 20, 30, 60] if show_patterns[days]]
        fig, most_similar_date, best_timeframe, similarity_score, all_pattern_collections = plot_multi_timeframe_patterns(
            data, 
            symbol, 
            timeframes, 
            show_patterns, 
            custom_date=custom_date, 
            enable_custom_date=enable_custom_date
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display some stats
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Volume Statistics")
            st.write(f"Total Trading Volume: {data['Volume'].sum():,.0f}")
            st.write(f"Average Daily Volume: {data.groupby(data.index.date)['Volume'].sum().mean():,.0f}")
        
        with col2:
            st.subheader("Pattern Statistics")
            st.write(f"Data Points: {len(data)}")
            st.write(f"Time Range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        
        # Display pattern match information if available
        if most_similar_date:
            st.subheader("Pattern Match Analysis")
            
            similar_date_str = pd.Timestamp(most_similar_date).strftime('%Y-%m-%d')
            st.write(f"Today's pattern most closely matches: **{similar_date_str}** (from {best_timeframe}-day timeframe)")
            st.write(f"Similarity score: {similarity_score:.4f} (lower is better)")
            
            # Get the end-of-day return for the most similar day
            similar_day_data = data[data.index.date == most_similar_date]
            if not similar_day_data.empty:
                day_open = similar_day_data['Open'].iloc[0]
                day_close = similar_day_data['Close'].iloc[-1]
                day_return = (day_close - day_open) / day_open * 100
                
                st.write(f"On that day, the {symbol} price ended {day_return:.2f}% from the open")
                
                # Calculate how much of that day has passed
                high_point = similar_day_data['High'].max()
                high_return = (high_point - day_open) / day_open * 100
                
                low_point = similar_day_data['Low'].min()
                low_return = (low_point - day_open) / day_open * 100
                
                st.write(f"High of day: {high_return:.2f}%, Low of day: {low_return:.2f}%")
                
                # Add a note about prediction
                st.info(f"If today continues to follow this pattern, {symbol} might end the day with similar performance.")
        
        # Display custom date statistics if enabled
        if enable_custom_date and custom_date:
            custom_date_data = data[data.index.date == custom_date]
            
            if not custom_date_data.empty:
                st.subheader(f"Custom Date Analysis: {custom_date.strftime('%Y-%m-%d')}")
                
                # Calculate day's performance
                day_open = custom_date_data['Open'].iloc[0]
                day_close = custom_date_data['Close'].iloc[-1]
                day_return = (day_close - day_open) / day_open * 100
                
                high_point = custom_date_data['High'].max()
                high_return = (high_point - day_open) / day_open * 100
                
                low_point = custom_date_data['Low'].min()
                low_return = (low_point - day_open) / day_open * 100
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Day Return", f"{day_return:.2f}%")
                with col2:
                    st.metric("Day High", f"{high_return:.2f}%")
                with col3:
                    st.metric("Day Low", f"{low_return:.2f}%")
                
                # Calculate volume
                day_volume = custom_date_data['Volume'].sum()
                avg_day_volume = data.groupby(data.index.date)['Volume'].sum().mean()
                volume_vs_avg = (day_volume / avg_day_volume - 1) * 100
                
                st.write(f"Total Volume: {day_volume:,.0f} ({volume_vs_avg:+.1f}% vs average)")
                
                # Find most similar day to the custom date
                similar_to_custom_date, custom_similarity = find_most_similar_day(custom_date_data, 
                                                                                 {d: p for d, p in all_pattern_collections[best_timeframe].items() 
                                                                                  if d != custom_date})
                
                if similar_to_custom_date:
                    similar_date_str = pd.Timestamp(similar_to_custom_date).strftime('%Y-%m-%d')
                    st.write(f"This date was most similar to: **{similar_date_str}**")
                    st.write(f"Similarity score: {custom_similarity:.4f} (lower is better)")
            
    else:
        st.error("No data available for the selected symbol.")
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}")