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

def create_seasonality_chart(data, symbol, num_years):
    current_week = get_current_week()
    
    # Filter for the last n years
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
    
    # Add the seasonal pattern line with its own y-axis
    fig.add_trace(go.Scatter(
        x=list(range(1, 53)),
        y=seasonal_pattern.values,
        mode='lines',
        name='Seasonal Pattern',
        line=dict(color='blue', width=2),
        yaxis='y'
    ))
    
    # Get current year's data
    current_year = datetime.now().year
    current_year_data = data[data.index.year == current_year].copy()
    current_year_data.index = pd.MultiIndex.from_arrays([
        current_year_data.index.year,
        current_year_data.index.isocalendar().week
    ], names=['year', 'week'])

    # Calculate weekly averages for current year
    current_weekly_avg = current_year_data.groupby(level='week').mean()

    # Normalize current year data
    year_min = current_weekly_avg.min()
    year_max = current_weekly_avg.max()
    current_normalized = ((current_weekly_avg - year_min) / (year_max - year_min)) * 100

    # Add the current year line with a secondary y-axis
    fig.add_trace(go.Scatter(
        x=list(range(1, len(current_normalized) + 1)),
        y=current_normalized.values,
        mode='lines',
        name=f'{current_year} Price',
        line=dict(color='yellow', width=2),
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
    
    # Add padding for seasonal pattern (40% padding for more spread)
    y_min = max(0, seasonal_min - (seasonal_range * 0.4))
    y_max = seasonal_max + (seasonal_range * 0.4)
    
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
            ticktext=[f'Week {i}' for i in range(1, 53)],
            tickvals=list(range(1, 53)),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickangle=45,
        ),
        showlegend=True,
        height=800,
        width=None,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=100, b=80)
    )
    
    return fig, seasonal_pattern

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
    # First row
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
    # Second row
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
    # Third row
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

    # Stocks (renamed from Tech Stocks) - two rows of 3
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
            
            fig, seasonal_pattern = create_seasonality_chart(data, st.session_state.symbol, num_years)
            st.plotly_chart(fig, use_container_width=True)
            
                        
        else:
            st.error(f"No data available for {st.session_state.symbol}. Please check the symbol and try again.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try another symbol or check your internet connection.")

if __name__ == "__main__":
    main()