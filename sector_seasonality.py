import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# Set page to wide mode
st.set_page_config(layout="wide")

# Initialize session state variables
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = None
if 'selected_industry' not in st.session_state:
    st.session_state.selected_industry = None
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None

# Load stock classifications
@st.cache_data
def load_stock_classifications():
    classifications = pd.read_csv('stock_sectors.csv')
    sector_stocks = classifications.groupby('sector')['symbol'].apply(list).to_dict()
    industry_stocks = classifications.groupby('industry')['symbol'].apply(list).to_dict()
    return sector_stocks, industry_stocks

# Cache the data fetching
@st.cache_data(ttl=3600)
def fetch_market_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="max", interval='1d')  # Already using "max"
        if hist.empty:
            return None
        return hist['Close']
    except Exception as e:
        return None

def get_current_week():
    return datetime.now().isocalendar()[1]

def calculate_group_seasonality(symbols, num_years):
    all_patterns = []
    weekly_directions = {week: {'up': 0, 'down': 0} for week in range(1, 53)}
    valid_symbols = 0
    current_year_patterns = []
    
    with st.spinner(f'Calculating seasonality for {len(symbols)} symbols...'):
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(symbols):
            try:
                data = fetch_market_data(symbol)
                if data is not None:
                    # Get current year's data
                    current_year = datetime.now().year
                    current_year_data = data[data.index.year == current_year].copy()
                    if not current_year_data.empty:
                        current_year_data.index = pd.MultiIndex.from_arrays([
                            current_year_data.index.year,
                            current_year_data.index.isocalendar().week
                        ], names=['year', 'week'])
                        current_weekly_avg = current_year_data.groupby(level='week').mean()
                        current_min = current_weekly_avg.min()
                        current_max = current_weekly_avg.max()
                        if current_max > current_min:
                            current_normalized = ((current_weekly_avg - current_min) / (current_max - current_min)) * 100
                            current_year_patterns.append(current_normalized)

                    # Filter for the last n years
                    end_date = data.index.max()
                    start_date = end_date - pd.DateOffset(years=num_years)
                    filtered_data = data[data.index >= start_date]
                    
                    if not filtered_data.empty:
                        filtered_data = filtered_data.copy()
                        filtered_data.index = pd.MultiIndex.from_arrays([
                            filtered_data.index.year,
                            filtered_data.index.isocalendar().week
                        ], names=['year', 'week'])
                        
                        # Calculate weekly returns for direction
                        weekly_avg = filtered_data.groupby(level='week').mean()
                        weekly_returns = weekly_avg.pct_change()
                        
                        # Track direction for each week
                        for week in range(1, 53):
                            if week in weekly_returns.index:
                                if weekly_returns.loc[week] > 0:
                                    weekly_directions[week]['up'] += 1
                                else:
                                    weekly_directions[week]['down'] += 1
                        
                        # Calculate normalized pattern
                        year_min = weekly_avg.min()
                        year_max = weekly_avg.max()
                        normalized = ((weekly_avg - year_min) / (year_max - year_min)) * 100
                        all_patterns.append(normalized)
                        valid_symbols += 1
                        
            except Exception as e:
                continue
                
            progress_bar.progress((i + 1) / len(symbols))
    
    if valid_symbols > 0:
        # Calculate average pattern
        patterns_df = pd.concat(all_patterns)
        mean_pattern = patterns_df.groupby(level='week').mean()
        
        # Calculate strength of pattern for each week
        pattern_strength = {}
        for week in range(1, 53):
            total = weekly_directions[week]['up'] + weekly_directions[week]['down']
            if total > 0:
                max_direction = max(weekly_directions[week]['up'], weekly_directions[week]['down'])
                pattern_strength[week] = (max_direction / total) * 100
            else:
                pattern_strength[week] = 0
        
        # Calculate current year average if we have data
        current_year_pattern = None
        if current_year_patterns:
            current_year_df = pd.concat(current_year_patterns)
            current_year_pattern = current_year_df.groupby(level='week').mean()
                
        return mean_pattern, pattern_strength, valid_symbols, current_year_pattern
    return None, None, 0, None

def create_seasonality_chart(seasonal_pattern, pattern_strength, current_year_pattern, group_name, num_years, num_symbols):
    current_week = get_current_week()
    
    fig = go.Figure()
    
    # Add the main seasonal pattern line
    fig.add_trace(go.Scatter(
        x=list(range(1, 53)),
        y=seasonal_pattern.values,
        mode='lines',
        name='Seasonal Pattern',
        line=dict(color='blue', width=2)
    ))
    
    # Add current year pattern if available
    if current_year_pattern is not None:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(current_year_pattern) + 1)),
            y=current_year_pattern.values,
            mode='lines',
            name=f'{datetime.now().year} Price',
            line=dict(color='yellow', width=2)
        ))
    
    # Add markers for strong seasonal weeks
    strong_weeks = []
    strong_values = []
    annotations = []
    
    for week in range(1, 53):
        if pattern_strength[week] >= 75:
            strong_weeks.append(week)
            strong_values.append(seasonal_pattern.iloc[week-1])
            
            # Modified direction comparison
            if week == 1:
                # For week 1, compare with the last week of the pattern
                direction = "UP" if seasonal_pattern.iloc[0] > seasonal_pattern.iloc[-1] else "DOWN"
            elif week == 2:
                # For week 2, compare with week 1
                direction = "UP" if seasonal_pattern.iloc[1] > seasonal_pattern.iloc[0] else "DOWN"
            else:
                # For all other weeks
                direction = "UP" if seasonal_pattern.iloc[week-1] > seasonal_pattern.iloc[week-2] else "DOWN"
            
            annotations.append(dict(
                x=week,
                y=seasonal_pattern.iloc[week-1],
                text=f"Week {week}<br>{pattern_strength[week]:.0f}% {direction}",
                showarrow=True,
                arrowhead=1,
                yshift=20,
                font=dict(size=10)
            ))
    
    # Add markers for strong seasonal weeks
    if strong_weeks:
        fig.add_trace(go.Scatter(
            x=strong_weeks,
            y=strong_values,
            mode='markers',
            name='>75% move together',
            marker=dict(size=12, color='yellow', symbol='star'),
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
    y_min = min(seasonal_pattern.values)
    y_max = max(seasonal_pattern.values)
    if current_year_pattern is not None:
        y_min = min(y_min, min(current_year_pattern.values))
        y_max = max(y_max, max(current_year_pattern.values))
    y_range = y_max - y_min
    y_min = max(0, y_min - (y_range * 0.1))
    y_max = y_max + (y_range * 0.1)
    
    fig.update_layout(
        title=dict(
            text=f'{group_name} Seasonal Pattern (Last {num_years} Years)\nBased on {num_symbols} stocks',
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=24)
        ),
        xaxis_title='Week of Year',
        yaxis_title='Normalized Strength (%)',
        xaxis=dict(
            tickmode='array',
            ticktext=[f'Week {i}' for i in range(1, 53)],
            tickvals=list(range(1, 53)),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickangle=45,
        ),
        yaxis=dict(
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            range=[y_min, y_max]
        ),
        showlegend=True,
        height=800,
        width=None,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=100, b=80),
        annotations=annotations
    )
    
    return fig

def main():
    st.title('Sector & Industry Seasonality Analysis')
    
    # Load classifications
    sector_stocks, industry_stocks = load_stock_classifications()
    
    # Sidebar controls
    with st.sidebar:
        # View type selector
        view_type = st.radio(
            "Select Analysis Type",
            ["Sector", "Industry"]
        )
        
        if view_type == "Sector":
            selected_group = st.selectbox(
                "Select Sector",
                list(sector_stocks.keys())
            )
            symbols = sector_stocks[selected_group]
        else:
            selected_group = st.selectbox(
                "Select Industry",
                list(industry_stocks.keys())
            )
            symbols = industry_stocks[selected_group]
        
        # First, get max available years for this group
        max_available_years = 0
        for symbol in symbols:
            try:
                data = fetch_market_data(symbol)
                if data is not None:
                    years_available = (data.index.max() - data.index.min()).days / 365
                    max_available_years = max(max_available_years, int(years_available))
            except:
                continue
        
        # Use max_available_years for the slider
        num_years = st.slider(
            'Select number of years to analyze:', 
            min_value=1,
            max_value=max_available_years,
            value=min(25, max_available_years),  # Default to 25 years or max available if less
            help="Choose how many years of historical data to include in the analysis. Some industries may have less history available."
        )
    
    try:
        # Calculate seasonality
        pattern, pattern_strength, num_symbols, current_year_pattern = calculate_group_seasonality(symbols, num_years)
        
        if pattern is not None and num_symbols > 0:
            fig = create_seasonality_chart(pattern, pattern_strength, current_year_pattern, selected_group, num_years, num_symbols)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display group statistics
            st.sidebar.markdown(f"### Group Statistics")
            st.sidebar.markdown(f"Number of stocks analyzed: {num_symbols}")
            st.sidebar.markdown(f"Maximum years of history: {max_available_years}")
            
            # Display strongest seasonal weeks
            strong_weeks = [week for week in range(1, 53) if pattern_strength[week] >= 75]
            if strong_weeks:
                st.sidebar.markdown("### Strongest Seasonal Weeks")
                for week in strong_weeks:
                    if week == 1:
                        direction = "UP" if pattern.iloc[0] > pattern.iloc[-1] else "DOWN"
                    elif week == 2:
                        direction = "UP" if pattern.iloc[1] > pattern.iloc[0] else "DOWN"
                    else:
                        direction = "UP" if pattern.iloc[week-1] > pattern.iloc[week-2] else "DOWN"
                    st.sidebar.markdown(f"Week {week}: {pattern_strength[week]:.0f}% {direction}")
        else:
            st.error("No valid data available for the selected group and time period.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try another selection or check your data.")

if __name__ == "__main__":
    main()