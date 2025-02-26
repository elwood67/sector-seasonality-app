import streamlit as st
import pandas as pd
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

def get_current_week():
    return datetime.now().isocalendar()[1]

def handle_symbol_change(new_symbol):
    st.session_state.symbol = new_symbol

def calculate_directional_consistency(data, num_years):
    """
    Calculate the percentage of years where price moves in the same direction for each week,
    and return both the consistency percentage and predominant direction.
    """
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(years=num_years)
    filtered_data = data[data.index >= start_date].copy()
    
    # Create weekly returns
    filtered_data = filtered_data.resample('W').last()
    weekly_returns = filtered_data.pct_change()
    
    # Create a DataFrame with year and week columns
    weekly_returns.index = pd.MultiIndex.from_arrays([
        weekly_returns.index.year,
        weekly_returns.index.isocalendar().week
    ], names=['year', 'week'])
    
    consistency_dict = {}
    
    # Calculate directional consistency for each week
    for week in range(1, 53):
        week_data = weekly_returns.xs(week, level='week', drop_level=False)
        if not week_data.empty:
            positive_moves = (week_data > 0).sum()
            negative_moves = (week_data < 0).sum()
            total_moves = len(week_data.dropna())
            
            if total_moves > 0:
                max_consistent = max(positive_moves, negative_moves)
                consistency = (max_consistent / total_moves) * 100
                direction = "Bullish" if positive_moves > negative_moves else "Bearish"
                consistency_dict[week] = {
                    'consistency': consistency,
                    'direction': direction,
                    'positive_count': int(positive_moves),
                    'negative_count': int(negative_moves),
                    'total_moves': total_moves
                }
            else:
                consistency_dict[week] = {
                    'consistency': 0,
                    'direction': "Unknown",
                    'positive_count': 0,
                    'negative_count': 0,
                    'total_moves': 0
                }
                
    return consistency_dict

def create_seasonality_chart(data, symbol, num_years):
    current_week = get_current_week()
    
    # Calculate directional consistency
    directional_consistency = calculate_directional_consistency(data, num_years)
    
    # Original seasonality calculation code
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(years=num_years)
    filtered_data = data[data.index >= start_date]
    
    filtered_data = filtered_data.copy()
    filtered_data.index = pd.MultiIndex.from_arrays([
        filtered_data.index.year,
        filtered_data.index.isocalendar().week
    ], names=['year', 'week'])
    
    weekly_data = []
    years = filtered_data.index.get_level_values('year').unique()
    
    for year in years:
        year_data = filtered_data.xs(year, level='year')
        weekly_avg = year_data.groupby(level='week').mean()
        year_min = weekly_avg.min()
        year_max = weekly_avg.max()
        normalized = ((weekly_avg - year_min) / (year_max - year_min)) * 100
        weekly_data.append(normalized)
    
    combined_data = pd.concat(weekly_data)
    seasonal_pattern = combined_data.groupby(level='week').mean()
    
    fig = go.Figure()
    
    # Create a single connected line with colored segments between points
    x_values = list(range(1, 53))
    y_values = seasonal_pattern.values
    
    # Create the main line but make it nearly transparent - we'll overlay colored segments
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        name='Seasonal Pattern',
        marker=dict(size=8, color='lightgray'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Create a line segment between each pair of points with color based on consistency
    for i in range(1, 52):
        week = i + 1  # Adjust to 1-based week indexing
        consistency = directional_consistency.get(week, {'consistency': 0})['consistency'] if week in directional_consistency else 0
        direction = directional_consistency.get(week, {'direction': 'Unknown'})['direction'] if week in directional_consistency else 'Unknown'
        
        # Determine color based on consistency
        if consistency == 100:
            color = 'red'
            line_width = 4
            group = '100% Consistent'
        elif consistency >= 90:
            color = 'orange'
            line_width = 3.5
            group = '90-99% Consistent'
        elif consistency >= 80:
            color = 'yellow'
            line_width = 3
            group = '80-89% Consistent'
        elif consistency >= 70:
            color = 'green'
            line_width = 2.5
            group = '70-79% Consistent'
        elif consistency >= 60:
            color = 'lightblue'
            line_width = 2
            group = '60-69% Consistent'
        else:
            color = 'blue'
            line_width = 1.5
            group = '<60% Consistent'
        
        hover_text = (
            f"Week {week}<br>"
            f"Pattern Value: {y_values[i]:.1f}<br>"
            f"Direction: {direction}<br>"
            f"Consistency: {consistency:.1f}%"
        )
        
        # Add detailed info for weeks with consistency data
        if week in directional_consistency:
            info = directional_consistency[week]
            hover_text += (
                f"<br>Bullish Years: {info['positive_count']}"
                f"<br>Bearish Years: {info['negative_count']}"
                f"<br>Total Years: {info['total_moves']}"
            )
        
        # Create a segment
        fig.add_trace(go.Scatter(
            x=[i, i+1],
            y=[y_values[i-1], y_values[i]],
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
        {'name': '100% Consistent', 'color': 'red', 'width': 4},
        {'name': '90-99% Consistent', 'color': 'orange', 'width': 3.5},
        {'name': '80-89% Consistent', 'color': 'yellow', 'width': 3},
        {'name': '70-79% Consistent', 'color': 'green', 'width': 2.5},
        {'name': '60-69% Consistent', 'color': 'lightblue', 'width': 2},
        {'name': '<60% Consistent', 'color': 'blue', 'width': 1.5}
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
    current_year_data.index = pd.MultiIndex.from_arrays([
        current_year_data.index.year,
        current_year_data.index.isocalendar().week
    ], names=['year', 'week'])

    current_weekly_avg = current_year_data.groupby(level='week').mean()
    
    if not current_weekly_avg.empty:
        year_min = current_weekly_avg.min()
        year_max = current_weekly_avg.max()
        current_normalized = ((current_weekly_avg - year_min) / (year_max - year_min)) * 100

        fig.add_trace(go.Scatter(
            x=list(range(1, len(current_normalized) + 1)),
            y=current_normalized.values,
            mode='lines',
            name=f'{current_year} Price',
            line=dict(color='white', width=2),
            yaxis='y2'
        ))
    
    # Add quarterly backgrounds
    quarters = [
        (1, 13, 'Q1', 'rgba(144, 238, 144, 0.3)'),
        (14, 26, 'Q2', 'rgba(255, 182, 193, 0.3)'),
        (27, 39, 'Q3', 'rgba(210, 180, 140, 0.3)'),
        (40, 52, 'Q4', 'rgba(176, 224, 230, 0.3)')
    ]
    
    for start, end, quarter, color in quarters:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=0.5,
            layer="below",
            line_width=0,
            annotation_text=quarter,
            annotation_position="top left"
        )
    
    # Add "You are here" marker
    fig.add_vline(
        x=current_week,
        line_width=2,
        line_dash="dash",
        line_color="red"
    )
    
    # Calculate y-axis ranges with padding
    seasonal_min = min(seasonal_pattern.values)
    seasonal_max = max(seasonal_pattern.values)
    seasonal_range = seasonal_max - seasonal_min
    
    y_min = max(0, seasonal_min - (seasonal_range * 0.4))
    y_max = seasonal_max + (seasonal_range * 0.4)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Seasonal Pattern (Last {num_years} Years)',
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=24)
        ),
        xaxis_title='Week of Year',
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
            ticktext=[f'W{i}' for i in range(1, 53)],  # Shortened to just W1, W2, etc.
            tickvals=list(range(1, 53)),
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
            # Increase item width for better spacing
            itemwidth=40,
            # Add some tracegroup gap to space things out horizontally
            tracegroupgap=20
        ),
        height=800,
        width=None,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=120, b=80)  # Increased top margin for legend
    )
    
    # Update sidebar stats
    sidebar_stats = {
        '100% Consistent': [w for w, c in directional_consistency.items() if c['consistency'] == 100],
        '90-99% Consistent': [w for w, c in directional_consistency.items() if 90 <= c['consistency'] < 100],
        '80-89% Consistent': [w for w, c in directional_consistency.items() if 80 <= c['consistency'] < 90]
    }
    
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
            
            # Display statistics about consistent weeks in the sidebar
            with st.sidebar:
                st.markdown("### Directional Consistency Stats")
                for consistency_level, weeks in sidebar_stats.items():
                    if weeks:
                        st.markdown(f"**{consistency_level}:**")
                        week_details = []
                        for w in sorted(weeks):
                            direction = directional_consistency[w]['direction']
                            week_details.append(f"Week {w} ({direction})")
                        week_list = ", ".join(week_details)
                        st.markdown(f"{week_list}")
                        
        else:
            st.error(f"No data available for {st.session_state.symbol}. Please check the symbol and try again.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try another symbol or check your internet connection.")

if __name__ == "__main__":
    main()