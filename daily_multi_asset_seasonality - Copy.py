import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# Set page to wide mode
st.set_page_config(layout="wide")

# Initialize all session state variables at the start
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None
if 'seasonal_pattern' not in st.session_state:
    st.session_state.seasonal_pattern = None

# Cache the data fetching to avoid hitting rate limits
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_market_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="max", interval='1d')
        
        if hist.empty:
            return None
            
        return hist['Close']
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def get_current_day_of_year():
    return datetime.now().timetuple().tm_yday

def handle_symbol_change(new_symbol):
    st.session_state.symbol = new_symbol

def calculate_directional_consistency(data, num_years):
    """
    Calculate the percentage of years where price moves in the same direction for each day,
    and return both the consistency percentage and predominant direction.
    """
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(years=num_years)
    filtered_data = data[data.index >= start_date].copy()
    
    # Create daily returns
    daily_returns = filtered_data.pct_change()
    
    # Create a DataFrame with year and day of year columns
    daily_returns.index = pd.MultiIndex.from_arrays([
        daily_returns.index.year,
        daily_returns.index.dayofyear
    ], names=['year', 'day'])
    
    consistency_dict = {}
    
    # Calculate directional consistency for each day
    for day in range(1, 367):  # 366 days for leap years
        try:
            day_data = daily_returns.xs(day, level='day', drop_level=False)
            if not day_data.empty:
                positive_moves = (day_data > 0).sum()
                negative_moves = (day_data < 0).sum()
                total_moves = len(day_data.dropna())
                
                if total_moves > 0:
                    max_consistent = max(positive_moves, negative_moves)
                    consistency = (max_consistent / total_moves) * 100
                    direction = "Bullish" if positive_moves > negative_moves else "Bearish"
                    consistency_dict[day] = {
                        'consistency': consistency,
                        'direction': direction,
                        'positive_count': int(positive_moves),
                        'negative_count': int(negative_moves),
                        'total_moves': total_moves
                    }
                else:
                    consistency_dict[day] = {
                        'consistency': 0,
                        'direction': "Unknown",
                        'positive_count': 0,
                        'negative_count': 0,
                        'total_moves': 0
                    }
        except KeyError:
            # Day doesn't exist in data
            continue
                
    return consistency_dict

def get_month_day_from_day_of_year(day, year=None):
    """Convert day of year to month-day format (e.g., 32 -> Jan-31)"""
    if year is None:
        # Default to current year, or non-leap year
        year = datetime.now().year
        # Make sure it's not a leap year to handle day 366
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            year = year - 1
    
    # Handle edge case for day 366 in non-leap years
    if day == 366 and (year % 4 != 0 or (year % 100 == 0 and year % 400 != 0)):
        day = 365  # Just use Dec 31
        
    try:
        date = datetime(year, 1, 1) + timedelta(days=day-1)
        return date.strftime("%b-%d")
    except ValueError:
        # If we still have an error, just return the day number
        return f"Day {day}"

def create_seasonality_chart(data, symbol, num_years):
    current_day = get_current_day_of_year()
    
    # Calculate directional consistency
    directional_consistency = calculate_directional_consistency(data, num_years)
    
    # Original seasonality calculation code
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(years=num_years)
    filtered_data = data[data.index >= start_date]
    
    # Create year and day of year indices
    daily_data_by_year = {}
    for year in filtered_data.index.year.unique():
        year_data = filtered_data[filtered_data.index.year == year]
        
        # Create a day-of-year indexed DataFrame for each year
        days_of_year = year_data.index.dayofyear
        values = year_data.values
        
        daily_series = pd.Series(index=days_of_year, data=values)
        
        # Fill any missing days with NaN (there might be days with no trading)
        all_days = pd.Series(index=range(1, 367), dtype=float)
        all_days[daily_series.index] = daily_series.values
        
        # Normalize each year's data from 0-100
        min_val = all_days.dropna().min()
        max_val = all_days.dropna().max()
        normalized = ((all_days - min_val) / (max_val - min_val)) * 100
        
        daily_data_by_year[year] = normalized
    
    # Combine all years' data
    all_years_df = pd.DataFrame(daily_data_by_year)
    
    # Calculate the average for each day across all years
    seasonal_pattern = all_years_df.mean(axis=1)
    
    # Fill gaps (NaN values) with nearby values
    seasonal_pattern = seasonal_pattern.fillna(method='ffill').fillna(method='bfill')
    
    fig = go.Figure()
    
    # Create a single connected line with colored segments between points
    x_values = list(range(1, len(seasonal_pattern) + 1))
    y_values = seasonal_pattern.values
    
    # Create the main line but make it nearly transparent - we'll overlay colored segments
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        name='Seasonal Pattern',
        marker=dict(size=4, color='lightgray'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Create a line segment between each pair of points with color based on consistency
    for i in range(len(x_values) - 1):
        day = i + 1  # Adjust to 1-based day indexing
        consistency = directional_consistency.get(day, {'consistency': 0})['consistency'] if day in directional_consistency else 0
        direction = directional_consistency.get(day, {'direction': 'Unknown'})['direction'] if day in directional_consistency else 'Unknown'
        
        # Determine color based on consistency
        if consistency == 100:
            color = 'red'
            line_width = 3
            group = '100% Consistent'
        elif consistency >= 90:
            color = 'purple'  # Purple for 90-99%
            line_width = 2.5
            group = '90-99% Consistent'
        elif consistency >= 80:
            color = 'yellow'
            line_width = 2
            group = '80-89% Consistent'
        elif consistency >= 70:
            color = 'green'
            line_width = 1.8
            group = '70-79% Consistent'
        elif consistency >= 60:
            color = 'lightblue'
            line_width = 1.5
            group = '60-69% Consistent'
        else:
            color = 'blue'
            line_width = 1
            group = '<60% Consistent'
        
        # Convert day number to month-day format for display
        day_str = get_month_day_from_day_of_year(day)
        
        hover_text = (
            f"{day_str} (Day {day})<br>"
            f"Pattern Value: {y_values[i]:.1f}<br>"
            f"Direction: {direction}<br>"
            f"Consistency: {consistency:.1f}%"
        )
        
        # Add detailed info for days with consistency data
        if day in directional_consistency:
            info = directional_consistency[day]
            hover_text += (
                f"<br>Bullish Years: {info['positive_count']}"
                f"<br>Bearish Years: {info['negative_count']}"
                f"<br>Total Years: {info['total_moves']}"
            )
        
        # Create a segment
        fig.add_trace(go.Scatter(
            x=[x_values[i], x_values[i+1]],
            y=[y_values[i], y_values[i+1]],
            mode='lines',
            line=dict(color=color, width=line_width),
            hoverinfo='text',
            hovertext=[hover_text, hover_text],
            name=group,
            legendgroup=group,
            showlegend=False  # Don't show legend for every segment
        ))
    
    # Add legend entries (just once per consistency level)
    legend_entries = [
        {'name': '100% Consistent', 'color': 'red', 'width': 3},
        {'name': '90-99% Consistent', 'color': 'purple', 'width': 2.5},
        {'name': '80-89% Consistent', 'color': 'yellow', 'width': 2},
        {'name': '70-79% Consistent', 'color': 'green', 'width': 1.8},
        {'name': '60-69% Consistent', 'color': 'lightblue', 'width': 1.5},
        {'name': '<60% Consistent', 'color': 'blue', 'width': 1}
    ]
    
    for entry in legend_entries:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color=entry['color'], width=entry['width']),
            name=entry['name'],
            legendgroup=entry['name'],
            showlegend=True
        ))
    
    # Add current year line
    current_year = datetime.now().year
    current_year_data = data[data.index.year == current_year].copy()
    
    if not current_year_data.empty:
        # Create a Series with all days
        current_days = current_year_data.index.dayofyear
        current_values = current_year_data.values
        
        current_series = pd.Series(index=current_days, data=current_values)
        
        # Fill any missing days with NaN (there might be days with no trading)
        all_current_days = pd.Series(index=range(1, 367), dtype=float)
        all_current_days[current_series.index] = current_series.values
        
        # Normalize
        min_val = all_current_days.dropna().min()
        max_val = all_current_days.dropna().max()
        current_normalized = ((all_current_days - min_val) / (max_val - min_val)) * 100
        
        # Only plot up to the current day
        days_to_plot = min(current_day, len(current_normalized))
        valid_days = current_normalized.iloc[:days_to_plot].dropna()
        
        fig.add_trace(go.Scatter(
            x=valid_days.index.tolist(),
            y=valid_days.values,
            mode='lines',
            name=f'{current_year} Price',
            line=dict(color='white', width=2),
            yaxis='y2'
        ))
    
    # Add month dividers and labels
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_colors = [
        'rgba(144, 238, 144, 0.3)',  # Jan - Light green
        'rgba(255, 182, 193, 0.3)',  # Feb - Light pink
        'rgba(173, 216, 230, 0.3)',  # Mar - Light blue
        'rgba(255, 218, 185, 0.3)',  # Apr - Peach
        'rgba(152, 251, 152, 0.3)',  # May - Pale green
        'rgba(255, 192, 203, 0.3)',  # Jun - Pink
        'rgba(135, 206, 250, 0.3)',  # Jul - Light sky blue
        'rgba(255, 228, 181, 0.3)',  # Aug - Moccasin
        'rgba(144, 238, 144, 0.3)',  # Sep - Light green again
        'rgba(255, 182, 193, 0.3)',  # Oct - Light pink again
        'rgba(173, 216, 230, 0.3)',  # Nov - Light blue again
        'rgba(255, 218, 185, 0.3)',  # Dec - Peach again
    ]
    
    for i in range(12):
        start = month_starts[i]
        end = month_starts[(i+1) % 12] if i < 11 else 366
        
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=month_colors[i],
            opacity=0.5,
            layer="below",
            line_width=0,
            annotation_text=month_names[i],
            annotation_position="top left"
        )
        
        # Add vertical lines at month boundaries
        if i > 0:  # Skip the first month start (day 1)
            fig.add_vline(
                x=start,
                line_width=1,
                line_dash="dash",
                line_color="gray"
            )
    
    # Add "You are here" marker
    fig.add_vline(
        x=current_day,
        line_width=2,
        line_dash="dash",
        line_color="red"
    )
    
    # Calculate y-axis ranges with padding
    seasonal_min = min(seasonal_pattern.dropna().values)
    seasonal_max = max(seasonal_pattern.dropna().values)
    seasonal_range = seasonal_max - seasonal_min
    
    y_min = max(0, seasonal_min - (seasonal_range * 0.4))
    y_max = seasonal_max + (seasonal_range * 0.4)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Daily Seasonal Pattern (Last {num_years} Years)',
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=24)
        ),
        xaxis_title='Day of Year',
        yaxis=dict(
            title='Seasonal Pattern Strength (%)',
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            range=[y_min, y_max],
            side='left'
        ),
        yaxis2=dict(
            title=f'{current_year} Price Movement (%)',
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            overlaying='y',
            side='right'
        ),
        xaxis=dict(
            tickmode='array',
            ticktext=[get_month_day_from_day_of_year(d) if d % 30 == 1 else '' for d in range(1, 367)],
            tickvals=list(range(1, 367, 30)),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickangle=45,
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            itemwidth=40,
            tracegroupgap=20
        ),
        height=800,
        width=None,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=120, b=80)
    )
    
    # Group days by consistency level for sidebar stats
    sidebar_stats = {
        '100% Consistent': [d for d, c in directional_consistency.items() if c['consistency'] == 100],
        '90-99% Consistent': [d for d, c in directional_consistency.items() if 90 <= c['consistency'] < 100],
        '80-89% Consistent': [d for d, c in directional_consistency.items() if 80 <= c['consistency'] < 90]
    }

    # Add cursor line feature
    fig.update_layout(
        hovermode="x unified",  # Show hover info for all traces at the same x position
        hoverdistance=100,      # Expand hover tolerance for easier selection
        spikedistance=1000,     # Allow spikes to appear from further away
        xaxis=dict(
            # Add the previous xaxis settings plus:
            showspikes=True,           # Show spike line for X axis
            spikethickness=2,          # Set thickness of spike line
            spikecolor="gray",         # Set color of spike line
            spikemode="across",        # Draw spike line across the full height
            spikedash="solid",         # Make spike line solid
        ),
        yaxis=dict(
            # Add the previous yaxis settings plus:
            showspikes=True,           # Show spike line for Y axis
            spikethickness=2,          # Set thickness of spike line
            spikecolor="gray",         # Set color of spike line
            spikemode="across"         # Draw spike line across the full width
        )
    )
    
    return fig, seasonal_pattern, directional_consistency, sidebar_stats

def main():
    with st.sidebar:
        st.title('Market Seasonality Chart')
        
        # Text input for custom symbols
        new_symbol = st.text_input('Enter Yahoo Symbol:', value=st.session_state.symbol)
        if new_symbol != st.session_state.symbol:
            handle_symbol_change(new_symbol.upper())
        
        # Crypto - two rows of 3
        st.markdown("### Crypto")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('BTC'):
                handle_symbol_change('BTC-USD')
        with col2:
            if st.button('ETH'):
                handle_symbol_change('ETH-USD')
        with col3:
            if st.button('DOGE'):
                handle_symbol_change('DOGE-USD')
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('BNB'):
                handle_symbol_change('BNB-USD')
        with col2:
            if st.button('SOL'):
                handle_symbol_change('SOL-USD')
        with col3:
            if st.button('LTC'):
                handle_symbol_change('LTC-USD')
        
        # Indices - two rows of 3
        st.markdown("### Indices")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('S&P 500'):
                handle_symbol_change('^SPX')
        with col2:
            if st.button('Dow Jones'):
                handle_symbol_change('^DJI')
        with col3:
            if st.button('RUT'):
                handle_symbol_change('^RUT')
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('NQ'):
                handle_symbol_change('^IXIC')
        with col2:
            if st.button('name6'):
                handle_symbol_change('sym')
        with col3:
            if st.button('DXY'):
                handle_symbol_change('DX-Y.NYB')
        
        # Forex & Commodities - three rows of 3
        st.markdown("### Forex & Commodities")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('EUR/USD'):
                handle_symbol_change('EURUSD=X')
        with col2:
            if st.button('Gold'):
                handle_symbol_change('GC=F')
        with col3:
            if st.button('Oil'):
                handle_symbol_change('CL=F')
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('Silver'):
                handle_symbol_change('SI=F')
        with col2:
            if st.button('Nat Gas'):
                handle_symbol_change('NG=F')
        with col3:
            if st.button('Copper'):
                handle_symbol_change('HG=F')
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('Sugar'):
                handle_symbol_change('SB=F')
        with col2:
            if st.button('Coffee'):
                handle_symbol_change('KC=F')
        with col3:
            if st.button('name10'):
                handle_symbol_change('sym')

        # Stocks
        st.markdown("### Stocks")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('Apple'):
                handle_symbol_change('AAPL')
        with col2:
            if st.button('Google'):
                handle_symbol_change('GOOG')
        with col3:
            if st.button('Microsoft'):
                handle_symbol_change('MSFT')
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('PBPB'):
                handle_symbol_change('PBPB')
        with col2:
            if st.button('GM'):
                handle_symbol_change('GM')
        with col3:
            if st.button('name13'):
                handle_symbol_change('sym')
    
    try:
        # Show loading message while fetching data
        with st.spinner(f'Fetching data for {st.session_state.symbol}...'):
            data = fetch_market_data(st.session_state.symbol)
        
        if data is not None:
            date_range = data.index.max() - data.index.min()
            max_years = int(date_range.days / 365)
            
            with st.sidebar:
                num_years = st.slider('Select number of years to analyze:', 
                                   min_value=1, 
                                   max_value=max_years, 
                                   value=min(5, max_years))
            
            # Use the new function with sidebar_stats return value
            fig, seasonal_pattern, directional_consistency, sidebar_stats = create_seasonality_chart(data, st.session_state.symbol, num_years)
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics about consistent days in the sidebar
            with st.sidebar:
                st.markdown("### Directional Consistency Stats")
                
                tab1, tab2, tab3 = st.tabs(["100%", "90-99%", "80-89%"])
                
                with tab1:
                    high_days = sidebar_stats['100% Consistent']
                    if high_days:
                        day_details = []
                        for d in sorted(high_days):
                            day_str = get_month_day_from_day_of_year(d)
                            direction = directional_consistency[d]['direction']
                            day_details.append(f"{day_str} (Day {d}, {direction})")
                        
                        # Split into columns for better display if there are many days
                        if len(day_details) > 10:
                            col1, col2 = st.columns(2)
                            half = len(day_details) // 2
                            col1.markdown("<br>".join(day_details[:half]), unsafe_allow_html=True)
                            col2.markdown("<br>".join(day_details[half:]), unsafe_allow_html=True)
                        else:
                            st.markdown("<br>".join(day_details), unsafe_allow_html=True)
                    else:
                        st.write("No days with 100% consistency")
                
                with tab2:
                    med_days = sidebar_stats['90-99% Consistent']
                    if med_days:
                        day_details = []
                        for d in sorted(med_days):
                            day_str = get_month_day_from_day_of_year(d)
                            consistency = directional_consistency[d]['consistency']
                            direction = directional_consistency[d]['direction']
                            day_details.append(f"{day_str} (Day {d}, {consistency:.1f}%, {direction})")
                        
                        # Split into columns for better display if there are many days
                        if len(day_details) > 10:
                            col1, col2 = st.columns(2)
                            half = len(day_details) // 2
                            col1.markdown("<br>".join(day_details[:half]), unsafe_allow_html=True)
                            col2.markdown("<br>".join(day_details[half:]), unsafe_allow_html=True)
                        else:
                            st.markdown("<br>".join(day_details), unsafe_allow_html=True)
                    else:
                        st.write("No days with 90-99% consistency")
                
                with tab3:
                    low_days = sidebar_stats['80-89% Consistent']
                    if low_days:
                        day_details = []
                        for d in sorted(low_days):
                            day_str = get_month_day_from_day_of_year(d)
                            consistency = directional_consistency[d]['consistency']
                            direction = directional_consistency[d]['direction']
                            day_details.append(f"{day_str} (Day {d}, {consistency:.1f}%, {direction})")
                        
                        # Split into columns for better display if there are many days
                        if len(day_details) > 15:
                            col1, col2 = st.columns(2)
                            half = len(day_details) // 2
                            col1.markdown("<br>".join(day_details[:half]), unsafe_allow_html=True)
                            col2.markdown("<br>".join(day_details[half:]), unsafe_allow_html=True)
                        else:
                            st.markdown("<br>".join(day_details), unsafe_allow_html=True)
                    else:
                        st.write("No days with 80-89% consistency")
                        
        else:
            st.error(f"No data available for {st.session_state.symbol}. Please check the symbol and try again.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())  # Print full traceback for debugging
        st.error("Please try another symbol or check your internet connection.")

if __name__ == "__main__":
    main()