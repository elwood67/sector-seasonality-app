import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np

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
        hist = ticker.history(period="max", interval='1d')
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
    all_stock_patterns = {}  # Store each stock's pattern
    
    with st.spinner(f'Processing {len(symbols)} stocks...'):
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        total_symbols = len(symbols)
        for i, symbol in enumerate(symbols):
            try:
                progress_text.text(f"Loading {symbol} ({i+1}/{total_symbols})")
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
                        all_stock_patterns[symbol] = normalized  # Store individual pattern
                        valid_symbols += 1
                        
            except Exception as e:
                continue
                
            # Update progress bar and show percentage
            progress = (i + 1) / total_symbols
            progress_bar.progress(progress)
            progress_text.text(f"Processed {i+1}/{total_symbols} stocks ({(progress * 100):.0f}%)")
            
        progress_text.text(f"Completed! Processed {valid_symbols} valid stocks out of {total_symbols}")

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
        
        # Calculate correlation with the mean pattern for each stock
        pattern_correlations = {}
        for symbol, pattern in all_stock_patterns.items():
            try:
                # Calculate correlation with mean pattern
                correlation = pattern.corr(mean_pattern)
                if not np.isnan(correlation):  # Filter out any NaN correlations
                    pattern_correlations[symbol] = correlation
            except:
                continue
        
        # Sort stocks by correlation
        correlation_leaders = sorted(pattern_correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate current year average if we have data
        current_year_pattern = None
        if current_year_patterns:
            current_year_df = pd.concat(current_year_patterns)
            current_year_pattern = current_year_df.groupby(level='week').mean()
                
        return mean_pattern, pattern_strength, valid_symbols, current_year_pattern, correlation_leaders, weekly_directions
    return None, None, 0, None, None, None

def create_seasonality_chart(seasonal_pattern, pattern_strength, current_year_pattern, group_name, num_years, num_symbols, weekly_directions):
    current_week = get_current_week()
    
    fig = go.Figure()
    
    # Create the main seasonal pattern with colored segments based on consistency
    x_values = list(range(1, len(seasonal_pattern) + 1))
    y_values = seasonal_pattern.values
    
    # Create a nearly invisible line to help Plotly connect the dots (will be overlaid with colored segments)
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(size=1, color='lightgray', opacity=0.2),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add colored line segments based on consistency percentages
    for i in range(len(x_values) - 1):
        week = i + 1  # Adjust to 1-based week indexing
        
        # Get consistency and direction info
        consistency = pattern_strength.get(week, 0)
        
        # Determine the predominant direction
        if week in weekly_directions:
            up_count = weekly_directions[week]['up']
            down_count = weekly_directions[week]['down']
            direction = "UP" if up_count > down_count else "DOWN"
            total_count = up_count + down_count
            if total_count > 0:
                max_consistent = max(up_count, down_count)
                consistency_text = f"{(max_consistent / total_count) * 100:.0f}% {direction}"
            else:
                consistency_text = "N/A"
        else:
            consistency_text = "N/A"
        
        # Determine color and line width based on consistency
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
        
        # Create hover text
        hover_text = f"Week {week}<br>Value: {y_values[i]:.1f}<br>{consistency_text}"
        
        # Create segment
        fig.add_trace(go.Scatter(
            x=[x_values[i], x_values[i+1]],
            y=[y_values[i], y_values[i+1]],
            mode='lines',
            line=dict(color=color, width=line_width),
            hoverinfo='text',
            hovertext=[hover_text, hover_text],
            name=group,
            legendgroup=group,
            showlegend=False  # Will add legend entries separately
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
    
    # Add current year pattern if available
    if current_year_pattern is not None:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(current_year_pattern) + 1)),
            y=current_year_pattern.values,
            mode='lines',
            name=f'{datetime.now().year} Price',
            line=dict(color='white', width=2),
            hovertemplate=None,
            showlegend=True
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
            showspikes=True,
            spikecolor='white',
            spikethickness=1,
            spikedash='solid',
            spikesnap='cursor'
        ),
        yaxis=dict(
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            range=[y_min, y_max],
            showspikes=True,
            spikecolor='white',
            spikethickness=1,
            spikedash='solid'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        height=800,
        width=None,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=100, b=80),
        hovermode='x',
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=11,
            font_family="Arial"
        )
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
            value=min(25, max_available_years),
            help="Choose how many years of historical data to include in the analysis. Some industries may have less history available."
        )
    
    try:
        # Calculate seasonality (with weekly_directions added to return values)
        pattern, pattern_strength, num_symbols, current_year_pattern, correlation_leaders, weekly_directions = calculate_group_seasonality(symbols, num_years)
        
        if pattern is not None and num_symbols > 0:
            # Updated chart creation function with weekly_directions
            fig = create_seasonality_chart(pattern, pattern_strength, current_year_pattern, selected_group, num_years, num_symbols, weekly_directions)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display group statistics
            st.sidebar.markdown(f"### Group Statistics")
            st.sidebar.markdown(f"Number of stocks analyzed: {num_symbols}")
            st.sidebar.markdown(f"Maximum years of history: {max_available_years}")
            
            # Display strongest pattern following stocks
            if correlation_leaders:
                st.sidebar.markdown("### Stocks correlating with industry pattern")
                num_to_show = min(10, len(correlation_leaders))
                
                for i, (symbol, corr) in enumerate(correlation_leaders[:num_to_show], 1):
                    # Convert correlation to percentage for easier reading
                    correlation_pct = corr * 100
                    st.sidebar.markdown(f"{i}. {symbol}: {correlation_pct:.1f}% pattern correlation")
            
            # Display strongest seasonal weeks in tabs to organize by consistency level
            st.sidebar.markdown("### Strongest Seasonal Weeks")
            tab1, tab2, tab3 = st.sidebar.tabs(["100%", "90-99%", "80-89%"])
            
            # Prepare week data by consistency levels
            weeks_100 = [w for w in range(1, 53) if pattern_strength.get(w, 0) == 100]
            weeks_90_99 = [w for w in range(1, 53) if 90 <= pattern_strength.get(w, 0) < 100]
            weeks_80_89 = [w for w in range(1, 53) if 80 <= pattern_strength.get(w, 0) < 90]
            
            with tab1:
                if weeks_100:
                    week_details = []
                    for week in sorted(weeks_100):
                        if week in weekly_directions:
                            up_count = weekly_directions[week]['up']
                            down_count = weekly_directions[week]['down']
                            direction = "UP" if up_count > down_count else "DOWN"
                            week_details.append(f"Week {week}: 100% {direction}")
                    
                    st.markdown("<br>".join(week_details), unsafe_allow_html=True)
                else:
                    st.write("No weeks with 100% consistency")
            
            with tab2:
                if weeks_90_99:
                    week_details = []
                    for week in sorted(weeks_90_99):
                        if week in weekly_directions:
                            up_count = weekly_directions[week]['up']
                            down_count = weekly_directions[week]['down']
                            direction = "UP" if up_count > down_count else "DOWN"
                            consistency = pattern_strength.get(week, 0)
                            week_details.append(f"Week {week}: {consistency:.1f}% {direction}")
                    
                    st.markdown("<br>".join(week_details), unsafe_allow_html=True)
                else:
                    st.write("No weeks with 90-99% consistency")
            
            with tab3:
                if weeks_80_89:
                    week_details = []
                    for week in sorted(weeks_80_89):
                        if week in weekly_directions:
                            up_count = weekly_directions[week]['up']
                            down_count = weekly_directions[week]['down']
                            direction = "UP" if up_count > down_count else "DOWN"
                            consistency = pattern_strength.get(week, 0)
                            week_details.append(f"Week {week}: {consistency:.1f}% {direction}")
                    
                    st.markdown("<br>".join(week_details), unsafe_allow_html=True)
                else:
                    st.write("No weeks with 80-89% consistency")

        else:
            st.error("No valid data available for the selected group and time period.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try another selection or check your data.")

if __name__ == "__main__":
    main()