import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

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

def find_seasonal_inflection_points(pattern):
    """Find peaks and troughs in the seasonal pattern"""
    values = pattern.values
    weeks = pattern.index.values
    
    # Find peaks (local maxima)
    peaks, peak_props = find_peaks(values, prominence=5)  # Minimum prominence to avoid noise
    
    # Find troughs (local minima) by finding peaks in inverted data
    troughs, trough_props = find_peaks(-values, prominence=5)
    
    inflection_points = []
    
    # Add peaks
    for peak in peaks:
        if peak < len(weeks):
            inflection_points.append({
                'week': weeks[peak],
                'value': values[peak],
                'type': 'peak',
                'strength': peak_props['prominences'][list(peaks).index(peak)]
            })
    
    # Add troughs
    for trough in troughs:
        if trough < len(weeks):
            inflection_points.append({
                'week': weeks[trough],
                'value': values[trough],
                'type': 'trough',
                'strength': trough_props['prominences'][list(troughs).index(trough)]
            })
    
    return sorted(inflection_points, key=lambda x: x['week'])

def calculate_deviation_from_seasonal(current_pattern, historical_pattern, current_week):
    """Calculate how much current performance deviates from historical seasonal expectation"""
    if current_pattern is None or len(current_pattern) == 0:
        return None, None
    
    # Get overlapping weeks
    common_weeks = set(current_pattern.index) & set(historical_pattern.index)
    if len(common_weeks) == 0:
        return None, None
    
    # Calculate deviations for each week
    deviations = {}
    for week in common_weeks:
        if week <= current_week:  # Only look at weeks up to current week
            historical_val = historical_pattern.loc[week]
            current_val = current_pattern.loc[week]
            deviation = current_val - historical_val
            deviations[week] = deviation
    
    if not deviations:
        return None, None
    
    # Calculate current deviation and trend
    recent_weeks = [w for w in deviations.keys() if w >= max(1, current_week - 4)]
    if len(recent_weeks) >= 2:
        recent_deviations = [deviations[w] for w in sorted(recent_weeks)]
        deviation_trend = np.polyfit(range(len(recent_deviations)), recent_deviations, 1)[0]
    else:
        deviation_trend = 0
    
    current_deviation = deviations.get(current_week, 0)
    return current_deviation, deviation_trend

def detect_momentum_shifts(current_pattern, historical_pattern, current_week):
    """Detect when current momentum diverges from seasonal expectations"""
    if current_pattern is None or len(current_pattern) < 3:
        return None, None, None
    
    # Get the last few weeks of data
    recent_weeks = [w for w in current_pattern.index if w <= current_week and w >= max(1, current_week - 3)]
    if len(recent_weeks) < 3:
        return None, None, None
    
    recent_weeks = sorted(recent_weeks)
    
    # Calculate current momentum (rate of change)
    current_values = [current_pattern.loc[w] for w in recent_weeks]
    current_momentum = np.polyfit(range(len(current_values)), current_values, 1)[0]
    
    # Calculate expected seasonal momentum
    historical_values = [historical_pattern.loc[w] for w in recent_weeks if w in historical_pattern.index]
    if len(historical_values) < 3:
        return None, None, None
    
    historical_momentum = np.polyfit(range(len(historical_values)), historical_values, 1)[0]
    
    # Calculate momentum divergence
    momentum_divergence = current_momentum - historical_momentum
    
    # Determine if this is a significant shift
    momentum_strength = abs(momentum_divergence)
    
    return current_momentum, historical_momentum, momentum_divergence

def analyze_trend_shifts_for_group(symbols, group_name, num_years):
    """Analyze all three types of trend shift indicators for a group"""
    # First get the basic seasonality data
    pattern, pattern_strength, num_symbols, current_year_pattern, correlation_leaders, weekly_directions = calculate_group_seasonality(symbols, num_years)
    
    if pattern is None:
        return None
    
    current_week = get_current_week()
    
    # 1. Find inflection points
    inflection_points = find_seasonal_inflection_points(pattern)
    
    # Find approaching inflection points (within 2 weeks)
    approaching_inflections = []
    for point in inflection_points:
        weeks_away = point['week'] - current_week
        if 0 <= weeks_away <= 2:  # Current week or next 2 weeks
            approaching_inflections.append({
                **point,
                'weeks_away': weeks_away
            })
    
    # 2. Calculate deviation from seasonal norm
    current_deviation, deviation_trend = calculate_deviation_from_seasonal(
        current_year_pattern, pattern, current_week
    )
    
    # 3. Detect momentum shifts
    current_momentum, historical_momentum, momentum_divergence = detect_momentum_shifts(
        current_year_pattern, pattern, current_week
    )
    
    # Calculate overall trend shift score
    shift_score = 0
    shift_reasons = []
    
    # Score approaching inflection points
    if approaching_inflections:
        for inflection in approaching_inflections:
            if inflection['weeks_away'] == 0:
                shift_score += 30  # At inflection point
                shift_reasons.append(f"At seasonal {inflection['type']} (Week {inflection['week']})")
            elif inflection['weeks_away'] == 1:
                shift_score += 20  # One week away
                shift_reasons.append(f"1 week from seasonal {inflection['type']} (Week {inflection['week']})")
            else:
                shift_score += 10  # Two weeks away
                shift_reasons.append(f"2 weeks from seasonal {inflection['type']} (Week {inflection['week']})")
    
    # Score deviation from seasonal norm
    if current_deviation is not None and abs(current_deviation) > 15:  # Significant deviation
        shift_score += min(25, abs(current_deviation) / 2)  # Up to 25 points
        direction = "above" if current_deviation > 0 else "below"
        shift_reasons.append(f"Running {abs(current_deviation):.1f}% {direction} seasonal expectation")
        
        # Extra points if deviation trend is accelerating
        if deviation_trend is not None and abs(deviation_trend) > 5:
            shift_score += 15
            trend_direction = "accelerating away from" if (current_deviation * deviation_trend) > 0 else "correcting toward"
            shift_reasons.append(f"Deviation {trend_direction} seasonal norm")
    
    # Score momentum divergence
    if momentum_divergence is not None and abs(momentum_divergence) > 2:
        shift_score += min(20, abs(momentum_divergence) * 5)  # Up to 20 points
        if momentum_divergence > 0:
            shift_reasons.append(f"Momentum stronger than seasonal expectation")
        else:
            shift_reasons.append(f"Momentum weaker than seasonal expectation")
    
    return {
        'group_name': group_name,
        'shift_score': min(100, shift_score),  # Cap at 100
        'shift_reasons': shift_reasons,
        'approaching_inflections': approaching_inflections,
        'current_deviation': current_deviation,
        'deviation_trend': deviation_trend,
        'momentum_divergence': momentum_divergence,
        'pattern': pattern,
        'current_pattern': current_year_pattern,
        'num_symbols': num_symbols
    }

def calculate_group_seasonality(symbols, num_years):
    """Original seasonality calculation function"""
    all_patterns = []
    weekly_directions = {week: {'up': 0, 'down': 0} for week in range(1, 53)}
    valid_symbols = 0
    current_year_patterns = []
    all_stock_patterns = {}
    
    # Show progress for individual group analysis
    if len(symbols) > 10:  # Only show progress for larger groups
        with st.spinner(f'Processing {len(symbols)} stocks...'):
            progress_bar = st.progress(0)
            progress_text = st.empty()
            show_progress = True
    else:
        show_progress = False
    
    total_symbols = len(symbols)
    for i, symbol in enumerate(symbols):
        try:
            if show_progress:
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
                    all_stock_patterns[symbol] = normalized
                    valid_symbols += 1
                    
        except Exception as e:
            continue
            
        # Update progress bar
        if show_progress:
            progress = (i + 1) / total_symbols
            progress_bar.progress(progress)
            progress_text.text(f"Processed {i+1}/{total_symbols} stocks ({(progress * 100):.0f}%)")
        
    if show_progress:
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
                correlation = pattern.corr(mean_pattern)
                if not np.isnan(correlation):
                    pattern_correlations[symbol] = correlation
            except:
                continue
        
        correlation_leaders = sorted(pattern_correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate current year average if we have data
        current_year_pattern = None
        if current_year_patterns:
            current_year_df = pd.concat(current_year_patterns)
            current_year_pattern = current_year_df.groupby(level='week').mean()
                
        return mean_pattern, pattern_strength, valid_symbols, current_year_pattern, correlation_leaders, weekly_directions
    return None, None, 0, None, None, None

def create_seasonality_chart(seasonal_pattern, pattern_strength, current_year_pattern, group_name, num_years, num_symbols, weekly_directions, trend_shift_data=None):
    """Enhanced chart with trend shift indicators"""
    current_week = get_current_week()
    
    fig = go.Figure()
    
    # Create the main seasonal pattern with colored segments based on consistency
    x_values = list(range(1, len(seasonal_pattern) + 1))
    y_values = seasonal_pattern.values
    
    # Create a nearly invisible line to help Plotly connect the dots
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
        week = i + 1
        
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
            color = 'purple'
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
            showlegend=False
        ))
    
    # Add legend entries
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
    
    # Add trend shift indicators
    if trend_shift_data and trend_shift_data['approaching_inflections']:
        for inflection in trend_shift_data['approaching_inflections']:
            # Add marker for approaching inflection points
            marker_color = 'orange' if inflection['type'] == 'peak' else 'cyan'
            marker_symbol = 'triangle-up' if inflection['type'] == 'peak' else 'triangle-down'
            
            fig.add_trace(go.Scatter(
                x=[inflection['week']],
                y=[inflection['value']],
                mode='markers',
                marker=dict(
                    size=15,
                    color=marker_color,
                    symbol=marker_symbol,
                    line=dict(width=2, color='white')
                ),
                name=f"Approaching {inflection['type'].title()}",
                hovertemplate=f"<b>Seasonal {inflection['type'].title()}</b><br>" +
                             f"Week: {inflection['week']}<br>" +
                             f"Days away: {inflection['weeks_away'] * 7}<br>" +
                             "<extra></extra>",
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
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text="You are here",
        annotation_position="top"
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
    
    # Add trend shift score to title if available
    title_text = f'{group_name} Seasonal Pattern (Last {num_years} Years)\nBased on {num_symbols} stocks'
    if trend_shift_data and trend_shift_data['shift_score'] > 0:
        title_text += f' | Trend Shift Score: {trend_shift_data["shift_score"]:.0f}/100'
    
    fig.update_layout(
        title=dict(
            text=title_text,
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=22)
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
            font=dict(size=10)
        ),
        height=800,
        width=None,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=120, b=80),
        hovermode='x',
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=11,
            font_family="Arial"
        )
    )
    
    return fig

def main():
    st.title('🔄 Advanced Seasonality & Trend Shift Analysis')
    
    # Load classifications
    sector_stocks, industry_stocks = load_stock_classifications()
    
    # Sidebar controls
    with st.sidebar:
        # View type selector
        analysis_mode = st.radio(
            "Select Analysis Mode",
            ["Individual Analysis", "Trend Shift Scanner"]
        )
        
        if analysis_mode == "Individual Analysis":
            # Original single group analysis
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
            
            # Get max available years for this group
            max_available_years = 0
            for symbol in symbols:
                try:
                    data = fetch_market_data(symbol)
                    if data is not None:
                        years_available = (data.index.max() - data.index.min()).days / 365
                        max_available_years = max(max_available_years, int(years_available))
                except:
                    continue
            
            num_years = st.slider(
                'Select number of years to analyze:', 
                min_value=1,
                max_value=max_available_years,
                value=min(25, max_available_years),
                help="Choose how many years of historical data to include in the analysis."
            )
            
        else:
            # Trend shift scanner mode
            st.markdown("### Trend Shift Scanner Settings")
            scan_type = st.radio(
                "Scan Type",
                ["All Industries", "All Sectors", "Custom Selection"]
            )
            
            num_years = st.slider(
                'Years of historical data:', 
                min_value=5,
                max_value=25,
                value=15,
                help="Years of historical data for pattern analysis"
            )
            
            min_shift_score = st.slider(
                'Minimum trend shift score:',
                min_value=10,
                max_value=80,
                value=25,
                help="Only show groups with shift scores above this threshold"
            )
    
    if analysis_mode == "Individual Analysis":
        # Original individual analysis code
        try:
            # Get trend shift data
            trend_shift_data = analyze_trend_shifts_for_group(symbols, selected_group, num_years)
            
            if trend_shift_data and trend_shift_data['pattern'] is not None:
                # Get original seasonality data for chart
                pattern, pattern_strength, num_symbols, current_year_pattern, correlation_leaders, weekly_directions = calculate_group_seasonality(symbols, num_years)
                
                # Create enhanced chart with trend shift indicators
                fig = create_seasonality_chart(
                    pattern, pattern_strength, current_year_pattern, 
                    selected_group, num_years, num_symbols, weekly_directions,
                    trend_shift_data
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display trend shift analysis
                if trend_shift_data['shift_score'] > 0:
                    st.markdown("### 🚨 Trend Shift Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Trend Shift Score", 
                            f"{trend_shift_data['shift_score']:.0f}/100",
                            delta="High Alert" if trend_shift_data['shift_score'] > 50 else "Moderate" if trend_shift_data['shift_score'] > 25 else "Low"
                        )
                    
                    with col2:
                        if trend_shift_data['current_deviation'] is not None:
                            st.metric(
                                "Deviation from Seasonal",
                                f"{trend_shift_data['current_deviation']:.1f}%",
                                delta="Above" if trend_shift_data['current_deviation'] > 0 else "Below"
                            )
                    
                    with col3:
                        if trend_shift_data['approaching_inflections']:
                            next_inflection = min(trend_shift_data['approaching_inflections'], key=lambda x: x['weeks_away'])
                            st.metric(
                                f"Next {next_inflection['type'].title()}",
                                f"Week {next_inflection['week']}",
                                delta=f"{next_inflection['weeks_away']} weeks away"
                            )
                    
                    if trend_shift_data['shift_reasons']:
                        st.markdown("**Key Indicators:**")
                        for reason in trend_shift_data['shift_reasons']:
                            st.markdown(f"• {reason}")
                else:
                    st.info("No significant trend shift indicators detected at this time.")
                
                # Original sidebar statistics
                st.sidebar.markdown(f"### Group Statistics")
                st.sidebar.markdown(f"Number of stocks analyzed: {num_symbols}")
                st.sidebar.markdown(f"Maximum years of history: {max_available_years}")
                
                # Display correlation leaders
                if correlation_leaders:
                    st.sidebar.markdown("### Top Pattern Followers")
                    num_to_show = min(10, len(correlation_leaders))
                    
                    for i, (symbol, corr) in enumerate(correlation_leaders[:num_to_show], 1):
                        correlation_pct = corr * 100
                        st.sidebar.markdown(f"{i}. {symbol}: {correlation_pct:.1f}%")
                
            else:
                st.error("No valid data available for the selected group and time period.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try another selection or check your data.")
    
    else:
        # Trend Shift Scanner Mode
        st.markdown("### 🔍 Trend Shift Scanner Results")
        
        # Determine which groups to scan
        if scan_type == "All Industries":
            groups_to_scan = [(name, symbols, "Industry") for name, symbols in industry_stocks.items()]
        elif scan_type == "All Sectors":
            groups_to_scan = [(name, symbols, "Sector") for name, symbols in sector_stocks.items()]
        else:
            # Custom selection - let user pick multiple
            st.sidebar.markdown("### Custom Selection")
            selected_sectors = st.sidebar.multiselect("Select Sectors:", list(sector_stocks.keys()))
            selected_industries = st.sidebar.multiselect("Select Industries:", list(industry_stocks.keys()))
            
            groups_to_scan = []
            for sector in selected_sectors:
                groups_to_scan.append((sector, sector_stocks[sector], "Sector"))
            for industry in selected_industries:
                groups_to_scan.append((industry, industry_stocks[industry], "Industry"))
        
        if groups_to_scan:
            # Run the trend shift analysis on all selected groups
            with st.spinner(f'Scanning {len(groups_to_scan)} groups for trend shifts...'):
                trend_shifts = []
                scan_progress = st.progress(0)
                scan_status = st.empty()
                
                for i, (group_name, symbols, group_type) in enumerate(groups_to_scan):
                    scan_status.text(f"Analyzing {group_name} ({i+1}/{len(groups_to_scan)})")
                    
                    try:
                        shift_data = analyze_trend_shifts_for_group(symbols, group_name, num_years)
                        if shift_data and shift_data['shift_score'] >= min_shift_score:
                            shift_data['group_type'] = group_type
                            trend_shifts.append(shift_data)
                    except Exception as e:
                        continue
                    
                    scan_progress.progress((i + 1) / len(groups_to_scan))
                
                scan_status.text(f"Scan complete! Found {len(trend_shifts)} groups with significant trend shift potential.")
            
            if trend_shifts:
                # Sort by shift score
                trend_shifts.sort(key=lambda x: x['shift_score'], reverse=True)
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["🚨 High Alert (50+)", "⚠️ Moderate (25-49)", "📊 All Results"])
                
                with tab1:
                    high_alert = [ts for ts in trend_shifts if ts['shift_score'] >= 50]
                    if high_alert:
                        st.markdown("### Groups with High Trend Shift Probability")
                        for shift_data in high_alert:
                            display_trend_shift_card(shift_data)
                    else:
                        st.info("No groups currently showing high alert trend shift signals.")
                
                with tab2:
                    moderate_alert = [ts for ts in trend_shifts if 25 <= ts['shift_score'] < 50]
                    if moderate_alert:
                        st.markdown("### Groups with Moderate Trend Shift Signals")
                        for shift_data in moderate_alert:
                            display_trend_shift_card(shift_data)
                    else:
                        st.info("No groups currently showing moderate trend shift signals.")
                
                with tab3:
                    st.markdown("### All Detected Trend Shifts")
                    for shift_data in trend_shifts:
                        display_trend_shift_card(shift_data, compact=True)
                
                # Summary statistics
                st.markdown("### Scanner Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Groups Scanned", len(groups_to_scan))
                
                with col2:
                    st.metric("Trend Shifts Detected", len(trend_shifts))
                
                with col3:
                    high_count = len([ts for ts in trend_shifts if ts['shift_score'] >= 50])
                    st.metric("High Alert", high_count)
                
                with col4:
                    if trend_shifts:
                        avg_score = sum(ts['shift_score'] for ts in trend_shifts) / len(trend_shifts)
                        st.metric("Avg Shift Score", f"{avg_score:.1f}")
                    else:
                        st.metric("Avg Shift Score", "0.0")
                
            else:
                st.info(f"No groups found with trend shift scores above {min_shift_score}. Try lowering the minimum threshold or selecting different groups.")
        else:
            st.warning("Please select at least one group to scan.")

def display_trend_shift_card(shift_data, compact=False):
    """Display a trend shift analysis card"""
    with st.container():
        # Create colored border based on shift score
        if shift_data['shift_score'] >= 50:
            border_color = "red"
        elif shift_data['shift_score'] >= 25:
            border_color = "orange"
        else:
            border_color = "blue"
        
        # Create the card content
        if not compact:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{shift_data['group_name']}** ({shift_data.get('group_type', 'Group')})")
                st.markdown(f"*Based on {shift_data['num_symbols']} stocks*")
                
                if shift_data['shift_reasons']:
                    for reason in shift_data['shift_reasons'][:3]:  # Show top 3 reasons
                        st.markdown(f"• {reason}")
                    if len(shift_data['shift_reasons']) > 3:
                        st.markdown(f"• ... and {len(shift_data['shift_reasons']) - 3} more indicators")
            
            with col2:
                # Score display
                score_color = "red" if shift_data['shift_score'] >= 50 else "orange" if shift_data['shift_score'] >= 25 else "blue"
                st.markdown(f"<div style='text-align: center; font-size: 24px; color: {score_color}; font-weight: bold;'>{shift_data['shift_score']:.0f}/100</div>", unsafe_allow_html=True)
                st.markdown("<div style='text-align: center; font-size: 12px;'>Shift Score</div>", unsafe_allow_html=True)
                
                # Quick metrics
                if shift_data['approaching_inflections']:
                    next_inflection = min(shift_data['approaching_inflections'], key=lambda x: x['weeks_away'])
                    st.markdown(f"<div style='text-align: center; font-size: 10px;'>Next {next_inflection['type']}: Week {next_inflection['week']}</div>", unsafe_allow_html=True)
        else:
            # Compact display for "All Results" tab
            col1, col2, col3 = st.columns([4, 2, 1])
            
            with col1:
                st.markdown(f"**{shift_data['group_name']}** ({shift_data.get('group_type', 'Group')})")
            
            with col2:
                if shift_data['shift_reasons']:
                    st.markdown(f"{shift_data['shift_reasons'][0][:50]}...")
            
            with col3:
                st.markdown(f"**{shift_data['shift_score']:.0f}/100**")
        
        st.divider()

if __name__ == "__main__":
    main()