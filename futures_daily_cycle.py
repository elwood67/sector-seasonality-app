import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import numpy as np

# Set page config
st.set_page_config(layout="wide", page_title="Futures Pattern Analysis")

# Futures market definitions
FUTURES_MARKETS = {
    'NQ': {
        'symbol': 'NQ=F',
        'name': 'E-mini NASDAQ',
        'rth_start': '09:30',
        'rth_end': '16:00',
        'eth_start': '18:00',
        'eth_end': '17:00',
        'tick_size': 0.25
    },
    'ES': {
        'symbol': 'ES=F',
        'name': 'E-mini S&P',
        'rth_start': '09:30',
        'rth_end': '16:00',
        'eth_start': '18:00',
        'eth_end': '17:00',
        'tick_size': 0.25
    },
    'CL': {
        'symbol': 'CL=F',
        'name': 'Crude Oil',
        'rth_start': '09:30',
        'rth_end': '16:00',
        'eth_start': '18:00',
        'eth_end': '17:00',
        'tick_size': 0.01
    },
    'GC': {
        'symbol': 'GC=F',
        'name': 'Gold',
        'rth_start': '09:30',
        'rth_end': '16:00',
        'eth_start': '18:00',
        'eth_end': '17:00',
        'tick_size': 0.10
    }
}

# Basic utility functions
def is_market_open(time_str, market_info):
    """Check if given time is during market hours"""
    time = pd.to_datetime(time_str).time()
    rth_start = pd.to_datetime(market_info['rth_start']).time()
    rth_end = pd.to_datetime(market_info['rth_end']).time()
    eth_start = pd.to_datetime(market_info['eth_start']).time()
    eth_end = pd.to_datetime(market_info['eth_end']).time()
    
    # Check RTH
    if rth_start <= time <= rth_end:
        return 'RTH'
    # Check ETH
    elif (eth_start <= time) or (time <= eth_end):
        return 'ETH'
    return None

def fetch_futures_data(symbol, period="60d", interval="15m"):
    """Fetch futures data and handle continuous contract"""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    
    if hist.empty:
        return None
        
    # Convert timestamps to EST
    hist.index = hist.index.tz_convert('US/Eastern')
    
    # Add session markers
    market_info = next((info for sym, info in FUTURES_MARKETS.items() 
                       if info['symbol'] == symbol), None)
    if market_info:
        hist['session'] = hist.index.map(
            lambda x: is_market_open(x.strftime('%H:%M'), market_info)
        )
    
    return hist

def calculate_daily_pattern(data, num_days=30, session_filter=None):
    """Calculate intraday pattern with session awareness"""
    # Get the most recent days
    end_date = data.index.max()
    start_date = end_date - pd.Timedelta(days=num_days)
    filtered_data = data[data.index >= start_date].copy()
    
    # Create empty DataFrame for patterns
    all_patterns = pd.DataFrame()
    
    # Get unique trading days (excluding weekends)
    unique_dates = filtered_data.index.date.unique()
    unique_dates = [d for d in unique_dates if d.weekday() < 5]  # Exclude weekends
    
    # Process each trading day
    for date in unique_dates:
        # Get data for this day
        day_data = filtered_data[filtered_data.index.date == date].copy()
        
        # Apply session filter if specified
        if session_filter and session_filter != "All Sessions":
            session = 'RTH' if session_filter == "RTH Only" else 'ETH'
            day_data = day_data[day_data['session'] == session]
        
        if not day_data.empty:
            # Calculate percentage change from session open
            day_open = day_data['Close'].iloc[0]
            day_pattern = ((day_data['Close'] - day_open) / day_open) * 100
            
            # Store pattern using time as index
            day_pattern.index = day_data.index.time
            all_patterns[date] = day_pattern
    
    # Calculate average pattern
    avg_pattern = all_patterns.mean(axis=1)
    
    return avg_pattern, all_patterns

def create_session_backgrounds(fig, market_info):
    """Add session background highlighting"""
    # RTH background (lighter)
    fig.add_vrect(
        x0=market_info['rth_start'],
        x1=market_info['rth_end'],
        fillcolor='rgba(255,255,255,0.05)',
        layer="below",
        line_width=0,
        name="RTH"
    )
    
    # ETH background (darker)
    eth_start_time = pd.to_datetime(market_info['eth_start']).time()
    eth_end_time = pd.to_datetime(market_info['eth_end']).time()
    
    if eth_start_time > eth_end_time:  # Overnight session
        fig.add_vrect(
            x0=market_info['eth_start'],
            x1="23:59",
            fillcolor='rgba(100,100,100,0.05)',
            layer="below",
            line_width=0,
            name="ETH"
        )
        fig.add_vrect(
            x0="00:00",
            x1=market_info['eth_end'],
            fillcolor='rgba(100,100,100,0.05)',
            layer="below",
            line_width=0,
            showlegend=False
        )
    else:  # Same day session
        fig.add_vrect(
            x0=market_info['eth_start'],
            x1=market_info['eth_end'],
            fillcolor='rgba(100,100,100,0.05)',
            layer="below",
            line_width=0,
            name="ETH"
        )

def add_market_events(fig, market_info):
    """Add vertical lines for significant market events"""
    events = {
        'US Open': market_info['rth_start'],
        'US Close': market_info['rth_end'],
        'London Open': '03:00',
        'Asia Open': '20:00'
    }
    
    for event, time in events.items():
        fig.add_vline(
            x=time,
            line_dash="dash",
            line_color="rgba(255,255,255,0.2)",
            annotation_text=event,
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color="rgba(255,255,255,0.5)"
        )

def handle_contract_rollover(data, market_info):
    """Handle futures contract rollover"""
    # Detect potential rollovers
    daily_ranges = data.groupby(data.index.date).agg({
        'High': 'max',
        'Low': 'min'
    })
    
    # Look for abnormal price jumps that might indicate rollover
    daily_ranges['gap'] = daily_ranges['Low'].shift(-1) - daily_ranges['High']
    threshold = daily_ranges['High'].std() * 3
    rollover_dates = daily_ranges[abs(daily_ranges['gap']) > threshold].index
    
    if len(rollover_dates) > 0:
        st.warning(f"Detected potential contract rollover on {rollover_dates[0]}")
    
    return data

def plot_futures_patterns(data, symbol, timeframes, show_patterns, market_info):
    """Create plot showing patterns across multiple timeframes with volume and sessions"""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Add session backgrounds and market events
    create_session_backgrounds(fig, market_info)
    add_market_events(fig, market_info)
    
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
    session_filter = market_info.get('session_filter', 'All Sessions')
    
    for days in timeframes:
        if show_patterns[days]:
            avg_pattern, all_patterns = calculate_daily_pattern(data, num_days=days, session_filter=session_filter)
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

    # Add composite pattern if enabled
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
                line=dict(color='white', width=4),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Add today's price if during trading hours
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
    
    # Add volume analysis
    if 'Volume' in data.columns:
        data['time_bucket'] = data.index.strftime('%H:%M')
        historical_volume = data.groupby('time_bucket')['Volume'].mean()
        
        if not today_data.empty:
            today_volume = today_data.groupby(today_data.index.strftime('%H:%M'))['Volume'].mean()
            
            max_vol = max(historical_volume.max(), today_volume.max() if not today_volume.empty else 0)
            norm_hist_vol = historical_volume / max_vol
            norm_today_vol = today_volume / max_vol if not today_volume.empty else pd.Series()
            
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
    
    # Update axes
    fig.update_xaxes(
        title_text=None,
        tickangle=45,
        tickformat='%H:%M',
        gridcolor='rgba(128,128,128,0.2)',
        gridwidth=1,
        showgrid=True,
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="Time of Day (EST)",
        tickangle=45,
        tickformat='%H:%M',
        gridcolor='rgba(128,128,128,0.2)',
        gridwidth=1,
        showgrid=True,
        row=2, col=1
    )
    
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

# Main Streamlit app
st.title("Futures Intraday Pattern Analysis")

# Sidebar controls
with st.sidebar:
    # Symbol selection
    selected_market = st.selectbox(
        "Select Futures Market:",
        options=list(FUTURES_MARKETS.keys()),
        format_func=lambda x: FUTURES_MARKETS[x]['name']
    )
    
    # Quick select buttons
    st.subheader("Quick Select")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("NQ"): selected_market = "NQ"
        if st.button("CL"): selected_market = "CL"
    with col2:
        if st.button("GC"): selected_market = "GC"
        if st.button("ES"): selected_market = "ES"
    
    # Session selection
    session_filter = st.radio(
        "Session Filter:",
        options=["All Sessions", "RTH Only", "ETH Only"],
        index=0
    )
    
    # Time interval selection
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
    market_info = FUTURES_MARKETS[selected_market].copy()
    market_info['session_filter'] = session_filter
    
    with st.spinner(f"Fetching data for {market_info['name']}..."):
        data = fetch_futures_data(market_info['symbol'], interval=interval)
        
        if data is not None:
            # Handle contract rollover
            data = handle_contract_rollover(data, market_info)
            
            # Calculate and display patterns
            timeframes = [days for days in [5, 10, 20, 30, 60] if show_patterns[days]]
            fig = plot_futures_patterns(data, selected_market, timeframes, show_patterns, market_info)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Volume Statistics")
                display_volume_stats(data, session_filter)
            
            with col2:
                st.subheader("Pattern Statistics")
                display_pattern_stats(data, market_info)
                
        else:
            st.error(f"No data available for {market_info['name']}")
            
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please try another symbol or check your internet connection.")            