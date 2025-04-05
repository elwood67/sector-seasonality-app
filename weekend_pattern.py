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
    page_title="Crypto Weekend Patterns",
    page_icon="📈"
)

# Title and description
st.title("Crypto Weekend Pattern Analysis")
st.markdown("Analyze Saturday vs Sunday patterns for crypto assets")

def fetch_crypto_data(symbol, period="60d", interval="15m"):
    """Fetch crypto data from Yahoo Finance"""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    
    if hist.empty:
        return None
        
    # Convert timestamps to EST
    hist.index = hist.index.tz_convert('US/Eastern')
    
    return hist

def calculate_weekend_patterns(data):
    """Calculate separate patterns for Saturdays and Sundays, excluding today"""
    # Get today's date
    today = pd.Timestamp.now(tz='US/Eastern').date()
    
    # Filter only weekend data and exclude today
    weekend_data = data[data.index.dayofweek.isin([5, 6])].copy()  # 5=Saturday, 6=Sunday
    historical_weekend_data = weekend_data[weekend_data.index.date != today].copy()
    
    # Create empty DataFrames for patterns
    saturday_patterns = pd.DataFrame()
    sunday_patterns = pd.DataFrame()
    
    # Get unique dates (excluding today)
    unique_dates = pd.Series(historical_weekend_data.index.date).drop_duplicates()
    
    # Process each date
    for date in unique_dates:
        # Get data for this day
        day_data = historical_weekend_data[historical_weekend_data.index.date == date].copy()
        
        # Skip days with very little data
        if len(day_data) < 10:  # Arbitrary threshold
            continue
            
        # Calculate percentage change from day's open
        day_open = day_data['Close'].iloc[0]
        day_pattern = ((day_data['Close'] - day_open) / day_open) * 100
        
        # Store pattern using time as index
        day_pattern.index = day_data.index.time
        
        # Add to appropriate pattern collection
        day_of_week = day_data.index[0].dayofweek
        if day_of_week == 5:  # Saturday
            saturday_patterns[date] = day_pattern
        else:  # Sunday
            sunday_patterns[date] = day_pattern
    
    # Calculate average patterns
    avg_saturday = saturday_patterns.mean(axis=1) if not saturday_patterns.empty else pd.Series()
    avg_sunday = sunday_patterns.mean(axis=1) if not sunday_patterns.empty else pd.Series()
    
    return avg_saturday, avg_sunday, saturday_patterns, sunday_patterns

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
            if time in pattern.index and time in today_pattern.index:  # Extra safety check
                diff = today_pattern[time] - pattern[time]
                similarity += diff * diff
                count += 1
            
        if count > 0:
            avg_similarity = similarity / count
            
            if avg_similarity < best_similarity:
                best_similarity = avg_similarity
                most_similar_date = date
    
    return most_similar_date, best_similarity

def plot_weekend_patterns(data, symbol, custom_weekend=None, enable_custom_weekend=False):
    """Create plot showing Saturday vs Sunday patterns with volume"""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Check if today is a weekend
    today = pd.Timestamp.now(tz='US/Eastern')
    today_date = today.date()
    is_weekend = today.dayofweek in [5, 6]  # 5=Saturday, 6=Sunday
    
    # Calculate weekend patterns
    avg_saturday, avg_sunday, saturday_patterns, sunday_patterns = calculate_weekend_patterns(data)
    
    # Plot Saturday pattern
    if not avg_saturday.empty:
        fig.add_trace(
            go.Scatter(
                x=[t.strftime('%H:%M') for t in avg_saturday.index],
                y=avg_saturday.values,
                mode='lines',
                name='Saturday Avg Pattern',
                line=dict(color='rgb(66, 135, 245)', width=3),  # Blue
            ),
            row=1, col=1
        )
        
        # Plot individual Saturday patterns with low opacity
        for date, pattern in saturday_patterns.items():
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in pattern.index],
                    y=pattern.values,
                    mode='lines',
                    name=f'Saturday {date_str}',
                    line=dict(color='rgb(66, 135, 245)', width=1),
                    opacity=0.15,  # Reduced opacity
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Plot Sunday pattern
    if not avg_sunday.empty:
        fig.add_trace(
            go.Scatter(
                x=[t.strftime('%H:%M') for t in avg_sunday.index],
                y=avg_sunday.values,
                mode='lines',
                name='Sunday Avg Pattern',
                line=dict(color='rgb(255, 99, 132)', width=3),  # Red
            ),
            row=1, col=1
        )
        
        # Plot individual Sunday patterns with low opacity
        for date, pattern in sunday_patterns.items():
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in pattern.index],
                    y=pattern.values,
                    mode='lines',
                    name=f'Sunday {date_str}',
                    line=dict(color='rgb(255, 99, 132)', width=1),
                    opacity=0.15,  # Reduced opacity
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Find most similar historical day and today's data if it's a weekend
    most_similar_date = None
    similarity_score = float('inf')
    
    if is_weekend:
        today_data = data[data.index.date == today_date]
        
        if not today_data.empty and len(today_data) >= 5:
            # Calculate today's price changes
            today_open = today_data['Close'].iloc[0]
            today_changes = ((today_data['Close'] - today_open) / today_open) * 100
            
            # Find most similar historical day
            if today.dayofweek == 5:  # Saturday
                most_similar_date, similarity_score = find_most_similar_day(today_data, saturday_patterns)
                day_type = "Saturday"
            else:  # Sunday
                most_similar_date, similarity_score = find_most_similar_day(today_data, sunday_patterns)
                day_type = "Sunday"
            
            # Plot today's line
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in today_data.index.time],
                    y=today_changes.values,
                    mode='lines',
                    name=f"Today's Price ({day_type})",
                    line=dict(color='white', width=2, dash='dash'),
                ),
                row=1, col=1
            )
            
            # Plot most similar day with higher visibility if found
            if most_similar_date is not None:
                if today.dayofweek == 5:  # Saturday
                    pattern = saturday_patterns[most_similar_date]
                else:  # Sunday
                    pattern = sunday_patterns[most_similar_date]
                
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
    
    # Calculate weekend patterns
    avg_saturday, avg_sunday, saturday_patterns, sunday_patterns = calculate_weekend_patterns(data)
    
    # Plot Saturday pattern
    if not avg_saturday.empty:
        fig.add_trace(
            go.Scatter(
                x=[t.strftime('%H:%M') for t in avg_saturday.index],
                y=avg_saturday.values,
                mode='lines',
                name='Saturday Avg Pattern',
                line=dict(color='rgb(66, 135, 245)', width=3),  # Blue
            ),
            row=1, col=1
        )
        
        # Plot individual Saturday patterns with low opacity
        for date, pattern in saturday_patterns.items():
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in pattern.index],
                    y=pattern.values,
                    mode='lines',
                    name=f'Saturday {date_str}',
                    line=dict(color='rgb(66, 135, 245)', width=1),
                    opacity=0.3,
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Plot Sunday pattern
    if not avg_sunday.empty:
        fig.add_trace(
            go.Scatter(
                x=[t.strftime('%H:%M') for t in avg_sunday.index],
                y=avg_sunday.values,
                mode='lines',
                name='Sunday Avg Pattern',
                line=dict(color='rgb(255, 99, 132)', width=3),  # Red
            ),
            row=1, col=1
        )
        
        # Plot individual Sunday patterns with low opacity
        for date, pattern in sunday_patterns.items():
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in pattern.index],
                    y=pattern.values,
                    mode='lines',
                    name=f'Sunday {date_str}',
                    line=dict(color='rgb(255, 99, 132)', width=1),
                    opacity=0.3,
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Add today's price if today is a weekend
    if is_weekend:
        today_date = today.date()
        today_data = data[data.index.date == today_date]
        
        if not today_data.empty:
            today_open = today_data['Close'].iloc[0]
            today_changes = ((today_data['Close'] - today_open) / today_open) * 100
            
            day_type = "Saturday" if today.dayofweek == 5 else "Sunday"
            
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in today_data.index.time],
                    y=today_changes.values,
                    mode='lines',
                    name=f"Today's Price ({day_type})",
                    line=dict(color='white', width=2, dash='dash'),
                ),
                row=1, col=1
            )
    
    # Add custom weekend day if requested
    if enable_custom_weekend and custom_weekend:
        custom_data = data[data.index.date == custom_weekend]
        
        if not custom_data.empty and len(custom_data) >= 5:
            # Calculate custom day pattern
            custom_open = custom_data['Close'].iloc[0]
            custom_changes = ((custom_data['Close'] - custom_open) / custom_open) * 100
            
            day_type = "Saturday" if custom_weekend.weekday() == 5 else "Sunday"
            date_str = custom_weekend.strftime('%Y-%m-%d')
            
            fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%H:%M') for t in custom_data.index.time],
                    y=custom_changes.values,
                    mode='lines',
                    name=f"Custom {day_type}: {date_str}",
                    line=dict(color='rgb(0, 255, 255)', width=2),  # Cyan
                ),
                row=1, col=1
            )
    
    # Process volume data
    if 'Volume' in data.columns:
        # Get today's date
        today_date = today.date()
        
        # Create weekend volume profiles (excluding today)
        weekend_data = data[data.index.dayofweek.isin([5, 6])].copy()
        historical_weekend_data = weekend_data[weekend_data.index.date != today_date].copy()
        historical_weekend_data['time_bucket'] = historical_weekend_data.index.strftime('%H:%M')
        historical_weekend_data['is_saturday'] = historical_weekend_data.index.dayofweek == 5
        
        # Saturday volume (historical)
        saturday_volume = historical_weekend_data[historical_weekend_data['is_saturday']].groupby('time_bucket')['Volume'].mean()
        
        # Sunday volume (historical)
        sunday_volume = historical_weekend_data[~historical_weekend_data['is_saturday']].groupby('time_bucket')['Volume'].mean()
        
        # Add today's volume if it's a weekend
        today_volume = None
        if is_weekend:
            today_date = today.date()
            today_data = data[data.index.date == today_date]
            
            if not today_data.empty:
                today_data['time_bucket'] = today_data.index.strftime('%H:%M')
                today_volume = today_data.groupby('time_bucket')['Volume'].mean()
        
        # Normalize volumes
        max_vol_values = [saturday_volume.max() if not saturday_volume.empty else 0, 
                         sunday_volume.max() if not sunday_volume.empty else 0]
        
        if today_volume is not None and not today_volume.empty:
            max_vol_values.append(today_volume.max())
            
        max_vol = max(max_vol_values)
        
        if max_vol > 0:
            if not saturday_volume.empty:
                norm_sat_vol = saturday_volume / max_vol
                fig.add_trace(
                    go.Bar(
                        x=norm_sat_vol.index,
                        y=norm_sat_vol.values,
                        name='Saturday Avg Volume',
                        marker_color='rgba(66, 135, 245, 0.5)',  # Blue
                        showlegend=True
                    ),
                    row=2, col=1
                )
                
            if not sunday_volume.empty:
                norm_sun_vol = sunday_volume / max_vol
                fig.add_trace(
                    go.Bar(
                        x=norm_sun_vol.index,
                        y=norm_sun_vol.values,
                        name='Sunday Avg Volume',
                        marker_color='rgba(255, 99, 132, 0.5)',  # Red
                        showlegend=True
                    ),
                    row=2, col=1
                )
                
            # Add today's volume if it's a weekend
            if today_volume is not None and not today_volume.empty:
                norm_today_vol = today_volume / max_vol
                day_type = "Saturday" if today.dayofweek == 5 else "Sunday"
                
                fig.add_trace(
                    go.Bar(
                        x=norm_today_vol.index,
                        y=norm_today_vol.values,
                        name=f"Today's Volume ({day_type})",
                        marker_color='rgba(255, 255, 255, 0.7)',  # White
                        showlegend=True
                    ),
                    row=2, col=1
                )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Weekend Patterns with Volume',
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=800,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            groupclick="toggleitem",  # Helps with legend management
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
    
    return fig

def get_weekend_stats(data):
    """Calculate statistics for weekend trading (excluding today)"""
    # Get today's date
    today = pd.Timestamp.now(tz='US/Eastern').date()
    
    # Filter weekend data (excluding today)
    weekend_data = data[data.index.dayofweek.isin([5, 6])].copy()
    historical_weekend_data = weekend_data[weekend_data.index.date != today].copy()
    
    saturday_data = historical_weekend_data[historical_weekend_data.index.dayofweek == 5]
    sunday_data = historical_weekend_data[historical_weekend_data.index.dayofweek == 6]
    
    # Get unique weekend dates
    saturday_dates = sorted(set(saturday_data.index.date))
    sunday_dates = sorted(set(sunday_data.index.date))
    
    # Calculate daily returns for weekends
    saturday_returns = []
    sunday_returns = []
    
    for date in saturday_dates:
        day_data = saturday_data[saturday_data.index.date == date]
        if len(day_data) >= 10:  # Only consider days with enough data
            day_open = day_data['Open'].iloc[0]
            day_close = day_data['Close'].iloc[-1]
            day_return = (day_close - day_open) / day_open * 100
            saturday_returns.append(day_return)
    
    for date in sunday_dates:
        day_data = sunday_data[sunday_data.index.date == date]
        if len(day_data) >= 10:  # Only consider days with enough data
            day_open = day_data['Open'].iloc[0]
            day_close = day_data['Close'].iloc[-1]
            day_return = (day_close - day_open) / day_open * 100
            sunday_returns.append(day_return)
    
    stats = {
        'saturday_dates': saturday_dates,
        'sunday_dates': sunday_dates,
        'saturday_returns': saturday_returns,
        'sunday_returns': sunday_returns,
        'avg_saturday_return': sum(saturday_returns) / len(saturday_returns) if saturday_returns else 0,
        'avg_sunday_return': sum(sunday_returns) / len(sunday_returns) if sunday_returns else 0,
        'total_saturday_volume': saturday_data['Volume'].sum(),
        'total_sunday_volume': sunday_data['Volume'].sum(),
        'avg_saturday_volume': saturday_data.groupby(saturday_data.index.date)['Volume'].sum().mean() if not saturday_data.empty else 0,
        'avg_sunday_volume': sunday_data.groupby(sunday_data.index.date)['Volume'].sum().mean() if not sunday_data.empty else 0,
    }
    
    return stats

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
    
    # Fixed period due to Yahoo Finance limitation
    st.info("Note: Yahoo Finance only provides 60 days of intraday data")
    period = "60d"  # Fixed at 60 days
    
    # Option to show individual days
    show_individual_days = st.checkbox("Show Individual Days", value=True)
    
    # Custom weekend date selector
    st.subheader("Custom Weekend")
    enable_custom_weekend = st.checkbox("Show Custom Weekend Day", value=False)
    
    # Get the weekend dates from the last 60 days
    today = pd.Timestamp.now(tz='US/Eastern').date()
    weekend_dates = []
    for i in range(60, 0, -1):
        check_date = today - pd.Timedelta(days=i)
        if check_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            weekend_dates.append(check_date)
    
    if weekend_dates:
        custom_weekend = st.selectbox(
            "Select Weekend Date:",
            options=weekend_dates,
            format_func=lambda x: f"{x.strftime('%Y-%m-%d')} ({'Saturday' if x.weekday() == 5 else 'Sunday'})",
            disabled=not enable_custom_weekend
        )
    else:
        st.write("No weekend dates available in data range.")

# Main app logic
try:
    # Fetch data
    with st.spinner(f"Fetching data for {symbol}..."):
        data = fetch_crypto_data(symbol, period=period, interval=interval)
    
    if data is not None:
        # Create and display plot
        fig = plot_weekend_patterns(data, symbol, 
                                   custom_weekend=custom_weekend if 'custom_weekend' in locals() else None, 
                                   enable_custom_weekend=enable_custom_weekend)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display weekend stats
        stats = get_weekend_stats(data)
        
        # Display summary statistics
        st.subheader("Weekend Trading Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Saturday Statistics")
            st.write(f"Number of Saturdays: {len(stats['saturday_dates'])}")
            dates_str = ", ".join([d.strftime('%Y-%m-%d') for d in stats['saturday_dates']])
            st.write(f"Dates included: {dates_str}")
            st.write(f"Average return: {stats['avg_saturday_return']:.2f}%")
            st.write(f"Average volume: {stats['avg_saturday_volume']:,.0f}")
            
            if stats['saturday_returns']:
                up_days = sum(1 for r in stats['saturday_returns'] if r > 0)
                down_days = sum(1 for r in stats['saturday_returns'] if r < 0)
                st.write(f"Up/Down days: {up_days}/{down_days}")
        
        with col2:
            st.subheader("Sunday Statistics")
            st.write(f"Number of Sundays: {len(stats['sunday_dates'])}")
            dates_str = ", ".join([d.strftime('%Y-%m-%d') for d in stats['sunday_dates']])
            st.write(f"Dates included: {dates_str}")
            st.write(f"Average return: {stats['avg_sunday_return']:.2f}%")
            st.write(f"Average volume: {stats['avg_sunday_volume']:,.0f}")
            
            if stats['sunday_returns']:
                up_days = sum(1 for r in stats['sunday_returns'] if r > 0)
                down_days = sum(1 for r in stats['sunday_returns'] if r < 0)
                st.write(f"Up/Down days: {up_days}/{down_days}")
        
    else:
        st.error("No data available for the selected symbol.")
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}")