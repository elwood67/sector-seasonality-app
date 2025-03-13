import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import numpy as np

# Set page config
st.set_page_config(
    layout="wide",
    page_title="Multi-Asset Intraday Pattern Analysis",
    page_icon="📈"
)

# Initialize session state
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'
if 'asset_type' not in st.session_state:
    st.session_state.asset_type = 'crypto'
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_interval' not in st.session_state:
    st.session_state.current_interval = '15m'

# Title and description
st.title("Multi-Asset Intraday Pattern Analysis")
st.markdown("Analyze intraday patterns and volume profiles for crypto and traditional financial assets")

def get_asset_type(symbol):
    """Determine if a symbol is crypto, stock, index, etc."""
    if "-USD" in symbol:
        return "crypto"
    elif symbol.startswith("^"):
        return "index"
    elif "=" in symbol:
        return "forex_commodity"
    else:
        return "stock"

def get_asset_trading_hours(symbol):
    """
    Get the appropriate trading hours for different asset types
    This helps properly display and process the data
    """
    asset_type = get_asset_type(symbol)
    
    if asset_type == 'crypto':
        # Crypto trades 24/7
        return {
            'is_24h': True,
            'regular_start': None,
            'regular_end': None,
            'x_range': None,  # Full 24 hours
            'hours_per_day': 24,
            'expected_points_per_day': {'5m': 288, '15m': 96, '30m': 48, '1h': 24}
        }
    elif asset_type == 'stock' or asset_type == 'index':
        # Standard US market hours: 9:30 AM to 4:00 PM Eastern
        return {
            'is_24h': False,
            'regular_start': '09:30',
            'regular_end': '16:00',
            'x_range': ['09:00', '16:30'],  # Pad a bit for display
            'hours_per_day': 6.5,  # 9:30 to 16:00 = 6.5 hours
            'expected_points_per_day': {'5m': 78, '15m': 26, '30m': 13, '1h': 7}
        }
    elif 'GC=F' in symbol:  # Gold futures specific handling
        # Gold futures trade almost 24 hours with a short break
        return {
            'is_24h': False,
            'regular_start': '18:00',  # Can vary, this is approximate
            'regular_end': '17:00',    # Next day
            'x_range': None,           # Show full day
            'hours_per_day': 23,       # Nearly 24 hours
            'expected_points_per_day': {'5m': 276, '15m': 92, '30m': 46, '1h': 23}
        }
    else:  # Other commodities and forex
        # Most futures/commodities trade in extended sessions
        return {
            'is_24h': False,
            'regular_start': '08:00',  # Varies by product
            'regular_end': '17:00',
            'x_range': ['08:00', '17:30'],
            'hours_per_day': 9,
            'expected_points_per_day': {'5m': 108, '15m': 36, '30m': 18, '1h': 9}
        }

def fetch_market_data(symbol, interval="15m"):
    """
    Fetch market data with improved handling for different asset types
    """
    try:
        ticker = yf.Ticker(symbol)
        asset_type = get_asset_type(symbol)
        
        # Determine appropriate parameters based on asset type
        if asset_type == 'crypto':
            # For crypto, use period parameter as it's 24/7 trading
            hist = ticker.history(period="60d", interval=interval)
        else:
            # For traditional assets, use start/end date to ensure we get enough trading days
            # We need to go back further since traditional markets are only open weekdays
            end_date = datetime.now()
            
            # For traditional markets, we need more calendar days to get enough trading days
            # 60 trading days ≈ 84 calendar days (accounting for weekends)
            calendar_days = 60  # Respecting the 60-day limit
            start_date = end_date - timedelta(days=calendar_days)
            
            hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                  end=end_date.strftime('%Y-%m-%d'), 
                                  interval=interval)
        
        if hist.empty:
            return None
            
        # Convert timestamps to EST for consistency
        try:
            hist.index = hist.index.tz_convert('US/Eastern')
        except:
            # If timezone conversion fails, localize to UTC then convert
            if not isinstance(hist.index, pd.DatetimeIndex):
                hist.index = pd.to_datetime(hist.index)
            
            if hist.index.tz is None:
                hist.index = hist.index.tz_localize('UTC').tz_convert('US/Eastern')
        
        return hist
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_daily_pattern(data, symbol, num_days=30):
    """
    Calculate intraday pattern with proper handling of trading hours
    """
    # Get the most recent days
    end_date = data.index.max()
    start_date = end_date - pd.Timedelta(days=num_days)
    filtered_data = data[data.index >= start_date].copy()
    
    # Create empty DataFrame for patterns
    all_patterns = pd.DataFrame()
    
    # Get unique dates
    unique_dates = pd.Series(filtered_data.index.date).drop_duplicates()
    
    # Get trading hours info
    trading_hours = get_asset_trading_hours(symbol)
    
    # For traditional markets, filter for regular trading hours
    if not trading_hours['is_24h']:
        if trading_hours['regular_start'] and trading_hours['regular_end']:
            # Filter data to include only regular trading hours
            market_filter = (
                (filtered_data.index.strftime('%H:%M') >= trading_hours['regular_start']) & 
                (filtered_data.index.strftime('%H:%M') <= trading_hours['regular_end'])
            )
            filtered_data = filtered_data[market_filter]
    
    # Determine expected points per day based on interval and asset type
    if len(filtered_data) > 1:
        interval = filtered_data.index[1] - filtered_data.index[0]
        interval_str = str(interval)
        
        if "0 days 00:05:00" in interval_str:
            expected_interval = "5m"
        elif "0 days 00:15:00" in interval_str:
            expected_interval = "15m"
        elif "0 days 00:30:00" in interval_str:
            expected_interval = "30m"
        elif "0 days 01:00:00" in interval_str:
            expected_interval = "1h"
        else:
            expected_interval = "15m"  # Default
    else:
        expected_interval = "15m"  # Default if we can't determine
    
    expected_points = trading_hours['expected_points_per_day'].get(expected_interval, 0)
    
    # Adjust minimum points requirement based on asset type
    # For crypto we expect more complete data
    min_points_factor = 0.7 if trading_hours['is_24h'] else 0.5
    
    # Process each day
    for date in unique_dates:
        # Get data for this day
        day_data = filtered_data[filtered_data.index.date == date].copy()
        
        # Only process days with enough data points
        if len(day_data) >= expected_points * min_points_factor:
            if len(day_data) > 0:  # Ensure we have data for this day
                # Use the first point of the day as the "open" reference
                day_open = day_data['Close'].iloc[0]
                if day_open > 0:  # Avoid division by zero
                    day_pattern = ((day_data['Close'] - day_open) / day_open) * 100
                    
                    # Store pattern using time as index
                    day_pattern.index = day_data.index.time
                    all_patterns[date] = day_pattern
    
    # Calculate average pattern if we have data
    if not all_patterns.empty:
        avg_pattern = all_patterns.mean(axis=1)
        return avg_pattern, all_patterns
    else:
        # Return empty series with proper index for error handling
        return pd.Series(), pd.DataFrame()

def fill_time_gaps_improved(time_series, value_series, symbol, interval_minutes=15):
    """
    Improved gap filling that respects trading hours
    """
    # Ensure inputs are valid
    if len(time_series) == 0 or len(value_series) == 0:
        return [], []
    
    # Create a dictionary of existing values
    value_dict = {t: v for t, v in zip(time_series, value_series)}
    
    # Get trading hours info
    trading_hours = get_asset_trading_hours(symbol)
    
    # Determine appropriate start and end times
    if trading_hours['is_24h']:
        # For crypto, use 24-hour range
        start_time = datetime.strptime('00:00', '%H:%M').time()
        end_time = datetime.strptime('23:59', '%H:%M').time()
    else:
        # For traditional markets, use market hours
        if trading_hours['regular_start'] and trading_hours['regular_end']:
            start_time = datetime.strptime(trading_hours['regular_start'], '%H:%M').time()
            end_time = datetime.strptime(trading_hours['regular_end'], '%H:%M').time()
        else:
            # Default times if none specified
            start_time = datetime.strptime('09:30', '%H:%M').time()
            end_time = datetime.strptime('16:00', '%H:%M').time()
    
    # Create list of all possible time points at the specified interval
    current_time = datetime.combine(datetime.today(), start_time)
    end_datetime = datetime.combine(datetime.today(), end_time)
    interval = timedelta(minutes=interval_minutes)
    
    all_times = []
    all_values = []
    
    while current_time <= end_datetime:
        t = current_time.time()
        all_times.append(t)
        
        # If we have a value for this time, use it, otherwise interpolate
        if t in value_dict:
            all_values.append(value_dict[t])
        else:
            # Find nearest previous and next times with values
            prev_times = [time for time in value_dict.keys() if time < t]
            next_times = [time for time in value_dict.keys() if time > t]
            
            if prev_times and next_times:
                # Interpolate between previous and next values
                prev_time = max(prev_times)
                next_time = min(next_times)
                
                prev_val = value_dict[prev_time]
                next_val = value_dict[next_time]
                
                # Convert times to seconds for ratio calculation
                prev_secs = prev_time.hour * 3600 + prev_time.minute * 60 + prev_time.second
                current_secs = t.hour * 3600 + t.minute * 60 + t.second
                next_secs = next_time.hour * 3600 + next_time.minute * 60 + next_time.second
                
                # Calculate the interpolation ratio
                if next_secs > prev_secs:  # Avoid division by zero
                    ratio = (current_secs - prev_secs) / (next_secs - prev_secs)
                    interp_val = prev_val + ratio * (next_val - prev_val)
                    all_values.append(interp_val)
                else:
                    all_values.append(None)
            elif prev_times:
                # If only previous values exist, use the most recent
                all_values.append(value_dict[max(prev_times)])
            elif next_times:
                # If only future values exist, use the closest
                all_values.append(value_dict[min(next_times)])
            else:
                # No values available
                all_values.append(None)
        
        current_time += interval
    
    return all_times, all_values

def process_and_display_volume(fig, data, symbol, today_data, yesterday_data, interval_minutes):
    """
    Process and display volume data with proper handling of trading hours
    """
    # Get trading hours info
    trading_hours = get_asset_trading_hours(symbol)
    
    # Create time buckets aligned with the interval
    time_buckets = {}
    
    # For each data point, create a time bucket at the interval boundary
    for idx, row in data.iterrows():
        # Format as HH:MM and ensure alignment to interval boundaries
        hour = idx.hour
        minute = (idx.minute // interval_minutes) * interval_minutes
        time_key = f"{hour:02d}:{minute:02d}"
        
        if time_key not in time_buckets:
            time_buckets[time_key] = []
        
        # Add volume to the bucket
        if 'Volume' in row and row['Volume'] > 0:
            time_buckets[time_key].append(row['Volume'])
    
    # Calculate average volume for each time bucket
    avg_volumes = {k: sum(v)/len(v) for k, v in time_buckets.items() if v}
    
    # Extract today's and yesterday's volumes
    today_volumes = {}
    yesterday_volumes = {}
    
    if not today_data.empty:
        for idx, row in today_data.iterrows():
            hour = idx.hour
            minute = (idx.minute // interval_minutes) * interval_minutes
            time_key = f"{hour:02d}:{minute:02d}"
            if 'Volume' in row and row['Volume'] > 0:
                today_volumes[time_key] = row['Volume']
                
    if not yesterday_data.empty:
        for idx, row in yesterday_data.iterrows():
            hour = idx.hour
            minute = (idx.minute // interval_minutes) * interval_minutes
            time_key = f"{hour:02d}:{minute:02d}"
            if 'Volume' in row and row['Volume'] > 0:
                yesterday_volumes[time_key] = row['Volume']
    
    # Create sorted lists of times
    # For traditional markets, only include times within trading hours
    if not trading_hours['is_24h'] and trading_hours['regular_start'] and trading_hours['regular_end']:
        start_time = datetime.strptime(trading_hours['regular_start'], '%H:%M')
        end_time = datetime.strptime(trading_hours['regular_end'], '%H:%M')
        
        # Filter times to include only those within trading hours
        valid_times = []
        for time_str in avg_volumes.keys():
            time_obj = datetime.strptime(time_str, '%H:%M')
            if start_time <= time_obj <= end_time:
                valid_times.append(time_str)
        
        display_times = sorted(valid_times)
    else:
        # For 24h markets, include all times
        display_times = sorted(avg_volumes.keys())
    
    # Create volume arrays for display
    avg_display = [avg_volumes.get(t, 0) for t in display_times]
    today_display = [today_volumes.get(t, 0) for t in display_times]
    yesterday_display = [yesterday_volumes.get(t, 0) for t in display_times]
    
    # Normalize volumes for display
    all_volumes = [v for v in avg_display + today_display + yesterday_display if v > 0]
    max_vol = max(all_volumes) if all_volumes else 1
    
    norm_avg = [v/max_vol if v > 0 else 0 for v in avg_display]
    norm_today = [v/max_vol if v > 0 else 0 for v in today_display]
    norm_yesterday = [v/max_vol if v > 0 else 0 for v in yesterday_display]
    
    # Add to chart
    fig.add_trace(
        go.Bar(
            x=display_times,
            y=norm_avg,
            name='Average Volume',
            marker_color='rgba(128, 128, 128, 0.3)',
            showlegend=True
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=display_times,
            y=norm_today,
            name="Today's Volume",
            marker_color='rgba(255, 99, 132, 0.5)',
            showlegend=True
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=display_times,
            y=norm_yesterday,
            name="Yesterday's Volume",
            marker_color='rgba(255, 165, 0, 0.5)',
            showlegend=True
        ),
        row=2, col=1
    )

def update_chart_layout(fig, symbol, trading_hours):
    """
    Update chart layout based on asset type
    """
    fig.update_layout(
        title=dict(
            text=f'{symbol} Multi-Timeframe Intraday Patterns',
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

    # Determine appropriate x-axis range and ticks
    if trading_hours['is_24h']:
        # For crypto, show full 24 hour cycle
        hours = list(range(0, 24))
        x_range = None
    else:
        # For traditional assets, focus on market hours with appropriate padding
        if trading_hours['x_range']:
            x_range = trading_hours['x_range']
            
            # Generate ticks based on range
            start_hour = int(x_range[0].split(':')[0])
            end_hour = int(x_range[1].split(':')[0])
            hours = list(range(start_hour, end_hour + 1))
        else:
            # Default to standard market hours if no specific range
            hours = list(range(9, 17))
            x_range = ['09:00', '16:30']
    
    # Common x-axis settings
    tick_vals = [f'{h:02d}:00' for h in hours]
    tick_text = [f'{h:02d}:00' for h in hours]
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=45,
        zeroline=True,
        zerolinecolor='rgba(128,128,128,0.2)',
        range=x_range,
        row=1, col=1
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=45,
        title_text="Time of Day (EST)",
        zeroline=True,
        zerolinecolor='rgba(128,128,128,0.2)',
        range=x_range,
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

def plot_multi_timeframe_patterns(data, symbol, timeframes, show_patterns):
    """
    Create plot with improved handling of different asset types
    """
    # Get trading hours info
    trading_hours = get_asset_trading_hours(symbol)
    
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
    
    # Determine interval minutes for filling gaps
    interval_map = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}
    current_interval = st.session_state.get('current_interval', '15m')
    interval_minutes = interval_map.get(current_interval, 15)
    
    # Calculate and plot patterns for each timeframe
    all_timeframe_patterns = {}
    
    composite_times = []
    composite_values = []
    has_patterns = False
    
    for days in timeframes:
        if show_patterns[days]:
            try:
                avg_pattern, all_patterns = calculate_daily_pattern(data, symbol, num_days=days)
                if not avg_pattern.empty:
                    has_patterns = True
                    all_timeframe_patterns[days] = avg_pattern
                    
                    # Fill gaps in time series for better visualization
                    times, values = fill_time_gaps_improved(
                        list(avg_pattern.index), 
                        list(avg_pattern.values),
                        symbol,
                        interval_minutes
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[t.strftime('%H:%M') for t in times],
                            y=values,
                            mode='lines',
                            name=f'{days}-Day Pattern',
                            line=dict(color=colors[days], width=2),
                        ),
                        row=1, col=1
                    )
            except Exception as e:
                st.warning(f"Could not calculate {days}-day pattern: {str(e)}")

    # Calculate and add composite pattern if enabled
    if show_patterns['composite'] and all_timeframe_patterns:
        # Collect all times from all patterns
        all_times = set()
        for pattern in all_timeframe_patterns.values():
            all_times.update(pattern.index)
        
        # Sort times for consistent display
        all_times = sorted(all_times)
        
        # For each time point, average available pattern values
        for time_point in all_times:
            values = [pattern[time_point] for pattern in all_timeframe_patterns.values() 
                     if time_point in pattern.index and not pd.isna(pattern[time_point])]
            
            if values:
                composite_times.append(time_point)
                composite_values.append(sum(values) / len(values))
        
        # Fill gaps in composite pattern
        if composite_times:
            comp_times, comp_values = fill_time_gaps_improved(
                composite_times, 
                composite_values,
                symbol,
                interval_minutes
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in comp_times],
                    y=comp_values,
                    mode='lines',
                    name='Composite Pattern',
                    line=dict(color='rgb(255, 255, 255)', width=4),
                    opacity=0.8
                ),
                row=1, col=1
            )
    
    # Add today's and yesterday's price lines
    today = pd.Timestamp.now(tz=pytz.timezone('US/Eastern')).date()
    today_data = data[data.index.date == today]
    
    yesterday = today - pd.Timedelta(days=1)
    
    # If yesterday is a weekend or holiday for traditional assets, go back to find the last trading day
    if not trading_hours['is_24h']:
        days_back = 1
        yesterday_data = data[data.index.date == yesterday]
        
        while yesterday_data.empty and days_back < 5:
            days_back += 1
            yesterday = today - pd.Timedelta(days=days_back)
            yesterday_data = data[data.index.date == yesterday]
    else:
        yesterday_data = data[data.index.date == yesterday]
    
    # Display today's price if available
    if not today_data.empty:
        try:
            today_open = today_data['Close'].iloc[0]
            today_changes = ((today_data['Close'] - today_open) / today_open) * 100
            
            # Fill gaps for smoother display
            today_times = [t.time() for t in today_data.index]
            today_times, today_values = fill_time_gaps_improved(
                today_times,
                list(today_changes.values),
                symbol,
                interval_minutes
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in today_times],
                    y=today_values,
                    mode='lines',
                    name="Today's Price",
                    line=dict(color='white', width=2, dash='dash'),
                ),
                row=1, col=1
            )
        except Exception as e:
            st.warning(f"Could not plot today's price: {e}")
    
    # Display yesterday's price if available
    if not yesterday_data.empty:
        try:
            yesterday_open = yesterday_data['Close'].iloc[0]
            yesterday_changes = ((yesterday_data['Close'] - yesterday_open) / yesterday_open) * 100
            
            # Fill gaps for smoother display
            yesterday_times = [t.time() for t in yesterday_data.index]
            yesterday_times, yesterday_values = fill_time_gaps_improved(
                yesterday_times,
                list(yesterday_changes.values),
                symbol,
                interval_minutes
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in yesterday_times],
                    y=yesterday_values,
                    mode='lines',
                    name="Yesterday's Price",
                    line=dict(color='rgba(255, 165, 0, 0.9)', width=2, dash='dot'),
                ),
                row=1, col=1
            )
        except Exception as e:
            st.warning(f"Could not plot yesterday's price: {e}")
    
    # Process volume data
    if 'Volume' in data.columns and data['Volume'].sum() > 0:
        process_and_display_volume(fig, data, symbol, today_data, yesterday_data, interval_minutes)
    
    # Update layout for the chart
    update_chart_layout(fig, symbol, trading_hours)
    
    # Add market open/close lines if not a 24h market
    if not trading_hours['is_24h'] and trading_hours['regular_start'] and trading_hours['regular_end']:
        # Market open line
        fig.add_vline(x=trading_hours['regular_start'], line_width=1, line_dash="dash", 
                     line_color="green", row=1, col=1)
        fig.add_vline(x=trading_hours['regular_start'], line_width=1, line_dash="dash", 
                     line_color="green", row=2, col=1)
        
        # Market close line
        fig.add_vline(x=trading_hours['regular_end'], line_width=1, line_dash="dash", 
                     line_color="red", row=1, col=1)
        fig.add_vline(x=trading_hours['regular_end'], line_width=1, line_dash="dash", 
                     line_color="red", row=2, col=1)
    
    # If no patterns were generated, add a note to the chart
    if not has_patterns:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No pattern data available for selected timeframes",
            showarrow=False,
            font=dict(size=20, color="white"),
            align="center",
            bgcolor="rgba(0,0,0,0.5)",
            opacity=0.8
        )
    
    return fig

def get_appropriate_intervals(asset_type):
    """Get the appropriate interval options based on asset type"""
    if asset_type == 'crypto':
        return ["5m", "15m", "30m", "1h"]
    else:
        return ["15m", "30m", "1h"]  # Traditional assets typically limited to 15m minimum
    
# Sidebar controls
with st.sidebar:
    st.title('Market Pattern Analysis')
    
    # Tabs for different asset categories
    category_tabs = st.tabs(["Crypto", "Indices", "Forex/Commodities", "Stocks"])
    
    # Crypto tab content
    with category_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            if st.button('BTC-USD', use_container_width=True):
                st.session_state.symbol = 'BTC-USD'
                st.session_state.asset_type = 'crypto'
                st.session_state.data_loaded = False
            if st.button('SOL-USD', use_container_width=True):
                st.session_state.symbol = 'SOL-USD'
                st.session_state.asset_type = 'crypto'
                st.session_state.data_loaded = False
            if st.button('DOT-USD', use_container_width=True):
                st.session_state.symbol = 'DOT-USD'
                st.session_state.asset_type = 'crypto'
                st.session_state.data_loaded = False
        with col2:
            if st.button('ETH-USD', use_container_width=True):
                st.session_state.symbol = 'ETH-USD'
                st.session_state.asset_type = 'crypto'
                st.session_state.data_loaded = False
            if st.button('DOGE-USD', use_container_width=True):
                st.session_state.symbol = 'DOGE-USD'
                st.session_state.asset_type = 'crypto'
                st.session_state.data_loaded = False
            if st.button('XRP-USD', use_container_width=True):
                st.session_state.symbol = 'XRP-USD'
                st.session_state.asset_type = 'crypto'
                st.session_state.data_loaded = False
    
    # Indices tab content
    with category_tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            if st.button('S&P 500', use_container_width=True):
                st.session_state.symbol = '^GSPC'
                st.session_state.asset_type = 'index'
                st.session_state.data_loaded = False
            if st.button('Russell 2000', use_container_width=True):
                st.session_state.symbol = '^RUT'
                st.session_state.asset_type = 'index'
                st.session_state.data_loaded = False
        with col2:
            if st.button('Dow Jones', use_container_width=True):
                st.session_state.symbol = '^DJI'
                st.session_state.asset_type = 'index'
                st.session_state.data_loaded = False
            if st.button('NASDAQ', use_container_width=True):
                st.session_state.symbol = '^IXIC'
                st.session_state.asset_type = 'index'
                st.session_state.data_loaded = False
    
    # Forex/Commodities tab content
    with category_tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            if st.button('EUR/USD', use_container_width=True):
                st.session_state.symbol = 'EURUSD=X'
                st.session_state.asset_type = 'forex'
                st.session_state.data_loaded = False
            if st.button('Gold', use_container_width=True):
                st.session_state.symbol = 'GC=F'
                st.session_state.asset_type = 'commodity'
                st.session_state.data_loaded = False
            if st.button('Silver', use_container_width=True):
                st.session_state.symbol = 'SI=F'
                st.session_state.asset_type = 'commodity'
                st.session_state.data_loaded = False
        with col2:
            if st.button('USD/JPY', use_container_width=True):
                st.session_state.symbol = 'USDJPY=X'
                st.session_state.asset_type = 'forex'
                st.session_state.data_loaded = False
            if st.button('Crude Oil', use_container_width=True):
                st.session_state.symbol = 'CL=F'
                st.session_state.asset_type = 'commodity'
                st.session_state.data_loaded = False
            if st.button('Nat Gas', use_container_width=True):
                st.session_state.symbol = 'NG=F'
                st.session_state.asset_type = 'commodity'
                st.session_state.data_loaded = False
    
    # Stocks tab content
    with category_tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            if st.button('AAPL', use_container_width=True):
                st.session_state.symbol = 'AAPL'
                st.session_state.asset_type = 'stock'
                st.session_state.data_loaded = False
            if st.button('MSFT', use_container_width=True):
                st.session_state.symbol = 'MSFT'
                st.session_state.asset_type = 'stock'
                st.session_state.data_loaded = False
            if st.button('TSLA', use_container_width=True):
                st.session_state.symbol = 'TSLA'
                st.session_state.asset_type = 'stock'
                st.session_state.data_loaded = False
        with col2:
            if st.button('AMZN', use_container_width=True):
                st.session_state.symbol = 'AMZN'
                st.session_state.asset_type = 'stock'
                st.session_state.data_loaded = False
            if st.button('NVDA', use_container_width=True):
                st.session_state.symbol = 'NVDA'
                st.session_state.asset_type = 'stock'
                st.session_state.data_loaded = False
            if st.button('META', use_container_width=True):
                st.session_state.symbol = 'META'
                st.session_state.asset_type = 'stock'
                st.session_state.data_loaded = False
    
    # Custom symbol input
    st.markdown("### Custom Symbol")
    new_symbol = st.text_input("Enter Yahoo Finance Symbol:", value=st.session_state.symbol)
    if new_symbol != st.session_state.symbol:
        st.session_state.symbol = new_symbol.upper()
        st.session_state.asset_type = get_asset_type(new_symbol.upper())
        st.session_state.data_loaded = False
    
    # Get interval options based on asset type
    interval_options = get_appropriate_intervals(st.session_state.asset_type)
    
    # Interval selection
    interval = st.select_slider(
        "Select Interval:",
        options=interval_options,
        value="15m" if "15m" in interval_options else interval_options[0]
    )
    if st.session_state.current_interval != interval:
        st.session_state.current_interval = interval
        st.session_state.data_loaded = False
    
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
    # If data is not loaded or interval has changed, load new data
    if not st.session_state.data_loaded:
        # Show loading message while fetching data
        with st.spinner(f"Fetching data for {st.session_state.symbol}..."):
            data = fetch_market_data(st.session_state.symbol, interval=st.session_state.current_interval)
            if data is not None:
                st.session_state.data = data
                st.session_state.data_loaded = True
    else:
        # Use cached data
        data = st.session_state.data
    
    if data is not None and len(data) > 0:
        # Create and display plot
        timeframes = [days for days in [5, 10, 20, 30, 60] if show_patterns[days]]
        fig = plot_multi_timeframe_patterns(data, st.session_state.symbol, timeframes, show_patterns)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display some stats
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Volume Statistics")
            if 'Volume' in data.columns and data['Volume'].sum() > 0:
                # Format large numbers with commas
                total_volume = f"{int(data['Volume'].sum()):,}"
                avg_daily_volume = f"{int(data.groupby(data.index.date)['Volume'].sum().mean()):,}"
                
                st.write(f"Total Trading Volume: {total_volume}")
                st.write(f"Average Daily Volume: {avg_daily_volume}")
            else:
                st.write("Volume data not available")
        
        with col2:
            st.subheader("Pattern Statistics")
            st.write(f"Data Points: {len(data)}")
            
            # Format date range
            if len(data) > 0:
                start_date = data.index.min().strftime('%Y-%m-%d')
                end_date = data.index.max().strftime('%Y-%m-%d')
                st.write(f"Time Range: {start_date} to {end_date}")
            
            # Display market hours info for non-crypto
            asset_type = st.session_state.asset_type
            if asset_type != 'crypto':
                st.markdown("**Note:** Traditional markets have specific trading hours. Patterns are based on available trading hours data.")
                
                # For traditional markets during off-hours, show a warning
                now = datetime.now().time()
                # Check if current time is outside of 9:30 AM to 4:00 PM EST on weekdays
                if datetime.now().weekday() >= 5:  # Weekend
                    st.warning("Markets are closed on weekends. Today's data will not be available.")
                elif now < datetime.strptime("09:30", "%H:%M").time() or now > datetime.strptime("16:00", "%H:%M").time():
                    st.info("Markets are currently closed. Real-time updates will be available during market hours.")
    else:
        st.error(f"No data available for {st.session_state.symbol} at {st.session_state.current_interval} interval. Try a different symbol or interval.")
        
        # Provide helpful information for troubleshooting
        asset_type = st.session_state.asset_type
        if asset_type != 'crypto':
            st.info("Traditional financial assets may have limited intraday data availability, especially for older data or during market closures.")
            st.info("For best results with traditional assets:")
            st.markdown("- Use 15m or 30m intervals instead of 5m")
            st.markdown("- Check during market hours (9:30 AM - 4:00 PM EST, Monday-Friday)")
            st.markdown("- Some assets may have delayed data depending on your data provider")
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    import traceback
    st.error(traceback.format_exc())  # Print full traceback for debugging
    st.error("Please try another symbol, interval, or check your internet connection.")    