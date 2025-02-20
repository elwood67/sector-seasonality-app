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

def plot_multi_timeframe_patterns(data, symbol, timeframes, show_patterns):
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
        60: 'rgb(153, 102, 255)'   # Purple
    }
    
    # Calculate and plot patterns for each timeframe
    all_timeframe_patterns = {}
    for days in timeframes:
        if show_patterns[days]:
            avg_pattern, all_patterns = calculate_daily_pattern(data, num_days=days)
            all_timeframe_patterns[days] = avg_pattern
            
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
    
    # Add today's price
    today = pd.Timestamp.now(tz='US/Eastern').date()
    today_data = data[data.index.date == today]
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
    
    # Process volume data
    if 'Volume' in data.columns:
        data['time_bucket'] = data.index.strftime('%H:%M')
        historical_volume = data.groupby('time_bucket')['Volume'].mean()
        
        if not today_data.empty:
            today_volume = today_data.groupby(today_data.index.strftime('%H:%M'))['Volume'].mean()
            
            max_vol = max(historical_volume.max(), today_volume.max())
            norm_hist_vol = historical_volume / max_vol
            norm_today_vol = today_volume / max_vol
            
            fig.add_trace(
                go.Bar(
                    x=historical_volume.index,
                    y=norm_hist_vol.values,
                    name='Average Volume',
                    marker_color='rgba(128, 128, 128, 0.3)',
                    showlegend=True
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=today_volume.index,
                    y=norm_today_vol.values,
                    name="Today's Volume",
                    marker_color='rgba(255, 99, 132, 0.5)',
                    showlegend=True
                ),
                row=2, col=1
            )

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

# Main app logic
try:
    # Fetch data
    with st.spinner(f"Fetching data for {symbol}..."):
        data = fetch_crypto_data(symbol, interval=interval)
    
    if data is not None:
        # Create and display plot
        timeframes = [days for days in [5, 10, 20, 30, 60] if show_patterns[days]]
        fig = plot_multi_timeframe_patterns(data, symbol, timeframes, show_patterns)
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
            
    else:
        st.error("No data available for the selected symbol.")
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}")