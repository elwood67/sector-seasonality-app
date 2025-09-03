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
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'scan_params' not in st.session_state:
    st.session_state.scan_params = None
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'individual_results' not in st.session_state:
    st.session_state.individual_results = None
if 'individual_params' not in st.session_state:
    st.session_state.individual_params = None

@st.cache_data
def load_stock_classifications():
    """Load stock classifications from CSV file"""
    classifications = pd.read_csv('stock_sectors.csv')
    sector_stocks = classifications.groupby('sector')['symbol'].apply(list).to_dict()
    industry_stocks = classifications.groupby('industry')['symbol'].apply(list).to_dict()
    return sector_stocks, industry_stocks

@st.cache_data(ttl=3600)
def fetch_market_data(symbol):
    """Fetch market data for a given symbol"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="max", interval='1d')
        if hist.empty:
            return None
        return hist['Close']
    except Exception as e:
        return None

def get_current_week():
    """Get current week number"""
    return datetime.now().isocalendar()[1]

def find_seasonal_inflection_points(pattern):
    """Find major peaks and troughs in the seasonal pattern"""
    values = pattern.values
    weeks = pattern.index.values
    
    # Use higher prominence and distance to find only major inflection points
    min_prominence = 10  # Filter out noise
    min_distance = 4     # Minimum 4 weeks between inflection points
    
    # Find peaks (local maxima)
    peaks, peak_props = find_peaks(values, prominence=min_prominence, distance=min_distance)
    
    # Find troughs (local minima)
    troughs, trough_props = find_peaks(-values, prominence=min_prominence, distance=min_distance)
    
    inflection_points = []
    
    # Add peaks with strength calculation
    for peak in peaks:
        if peak < len(weeks):
            # Calculate trend strength by looking at surrounding weeks
            start_idx = max(0, peak - 3)
            end_idx = min(len(values), peak + 4)
            trend_range = max(values[start_idx:end_idx]) - min(values[start_idx:end_idx])
            
            inflection_points.append({
                'week': weeks[peak],
                'value': values[peak],
                'type': 'peak',
                'strength': peak_props['prominences'][list(peaks).index(peak)],
                'trend_range': trend_range
            })
    
    # Add troughs with strength calculation
    for trough in troughs:
        if trough < len(weeks):
            # Calculate trend strength by looking at surrounding weeks
            start_idx = max(0, trough - 3)
            end_idx = min(len(values), trough + 4)
            trend_range = max(values[start_idx:end_idx]) - min(values[start_idx:end_idx])
            
            inflection_points.append({
                'week': weeks[trough],
                'value': values[trough],
                'type': 'trough',
                'strength': trough_props['prominences'][list(troughs).index(trough)],
                'trend_range': trend_range
            })
    
    # Filter out weak inflection points
    significant_points = [point for point in inflection_points if point['trend_range'] > 15]
    
    return sorted(significant_points, key=lambda x: x['week'])

def calculate_deviation_from_seasonal(current_pattern, historical_pattern, current_week):
    """Calculate deviation from historical seasonal pattern"""
    if current_pattern is None or len(current_pattern) == 0:
        return None, None
    
    # Get overlapping weeks
    common_weeks = set(current_pattern.index) & set(historical_pattern.index)
    if len(common_weeks) == 0:
        return None, None
    
    # Calculate deviations for each week
    deviations = {}
    for week in common_weeks:
        if week <= current_week:
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

def calculate_longer_term_trend(current_pattern, historical_pattern, current_week):
    """Calculate longer-term trend divergence (4-8 week periods)"""
    if current_pattern is None or len(current_pattern) < 8:
        return None, None
    
    trends_4w = {}
    trends_8w = {}
    
    for period, trends_dict in [(4, trends_4w), (8, trends_8w)]:
        if current_week >= period:
            # Get recent weeks for current year
            recent_current_weeks = [w for w in current_pattern.index 
                                  if current_week - period + 1 <= w <= current_week]
            
            # Get corresponding historical weeks
            recent_historical_weeks = [w for w in historical_pattern.index 
                                     if current_week - period + 1 <= w <= current_week]
            
            if len(recent_current_weeks) >= 3 and len(recent_historical_weeks) >= 3:
                # Calculate trends
                current_values = [current_pattern.loc[w] for w in sorted(recent_current_weeks)]
                historical_values = [historical_pattern.loc[w] for w in sorted(recent_historical_weeks)]
                
                current_trend = np.polyfit(range(len(current_values)), current_values, 1)[0]
                historical_trend = np.polyfit(range(len(historical_values)), historical_values, 1)[0]
                
                trends_dict['current'] = current_trend
                trends_dict['historical'] = historical_trend
                trends_dict['divergence'] = current_trend - historical_trend
    
    return trends_4w, trends_8w

def detect_momentum_shifts(current_pattern, historical_pattern, current_week):
    """Detect momentum shifts from seasonal expectations"""
    if current_pattern is None or len(current_pattern) < 3:
        return None, None, None
    
    # Get the last few weeks of data
    recent_weeks = [w for w in current_pattern.index if w <= current_week and w >= max(1, current_week - 3)]
    if len(recent_weeks) < 3:
        return None, None, None
    
    recent_weeks = sorted(recent_weeks)
    
    # Calculate current momentum
    current_values = [current_pattern.loc[w] for w in recent_weeks]
    current_momentum = np.polyfit(range(len(current_values)), current_values, 1)[0]
    
    # Calculate expected seasonal momentum
    historical_values = [historical_pattern.loc[w] for w in recent_weeks if w in historical_pattern.index]
    if len(historical_values) < 3:
        return None, None, None
    
    historical_momentum = np.polyfit(range(len(historical_values)), historical_values, 1)[0]
    
    # Calculate momentum divergence
    momentum_divergence = current_momentum - historical_momentum
    
    return current_momentum, historical_momentum, momentum_divergence

def calculate_group_seasonality(symbols, num_years):
    """Calculate seasonality pattern for a group of symbols"""
    all_patterns = []
    weekly_directions = {week: {'up': 0, 'down': 0} for week in range(1, 53)}
    valid_symbols = 0
    current_year_patterns = []
    all_stock_patterns = {}
    
    # Show progress for larger groups
    if len(symbols) > 10:
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

def analyze_trend_shifts_for_group(symbols, group_name, num_years):
    """Analyze trend shift indicators for a group"""
    # Get basic seasonality data
    pattern, pattern_strength, num_symbols, current_year_pattern, correlation_leaders, weekly_directions = calculate_group_seasonality(symbols, num_years)
    
    if pattern is None:
        return None
    
    current_week = get_current_week()
    
    # 1. Find inflection points
    inflection_points = find_seasonal_inflection_points(pattern)
    
    # Find approaching inflection points (within 4 weeks)
    approaching_inflections = []
    for point in inflection_points:
        weeks_away = point['week'] - current_week
        if 0 <= weeks_away <= 4:
            approaching_inflections.append({
                **point,
                'weeks_away': weeks_away
            })
    
    # 2. Calculate deviation from seasonal norm
    current_deviation, deviation_trend = calculate_deviation_from_seasonal(
        current_year_pattern, pattern, current_week
    )
    
    # 3. Calculate longer-term trends
    trends_4w, trends_8w = calculate_longer_term_trend(
        current_year_pattern, pattern, current_week
    )
    
    # 4. Enhanced momentum detection
    current_momentum, historical_momentum, momentum_divergence = detect_momentum_shifts(
        current_year_pattern, pattern, current_week
    )
    
    # Calculate trend shift score
    shift_score = 0
    shift_reasons = []
    
    # Score approaching inflection points
    if approaching_inflections:
        for inflection in approaching_inflections:
            base_score = min(25, inflection['trend_range'])
            
            if inflection['weeks_away'] == 0:
                shift_score += base_score
                shift_reasons.append(f"At major seasonal {inflection['type']} (Week {inflection['week']})")
            elif inflection['weeks_away'] <= 2:
                shift_score += base_score * 0.7
                shift_reasons.append(f"{inflection['weeks_away']} weeks from major seasonal {inflection['type']} (Week {inflection['week']})")
            else:
                shift_score += base_score * 0.4
                shift_reasons.append(f"{inflection['weeks_away']} weeks from seasonal {inflection['type']} (Week {inflection['week']})")
    
    # Score longer-term trend divergences
    if trends_8w and abs(trends_8w.get('divergence', 0)) > 3:
        divergence_score = min(30, abs(trends_8w['divergence']) * 3)
        shift_score += divergence_score
        direction = "stronger" if trends_8w['divergence'] > 0 else "weaker"
        shift_reasons.append(f"8-week trend {direction} than seasonal pattern")
    
    if trends_4w and abs(trends_4w.get('divergence', 0)) > 4:
        divergence_score = min(20, abs(trends_4w['divergence']) * 2.5)
        shift_score += divergence_score
        direction = "stronger" if trends_4w['divergence'] > 0 else "weaker"
        shift_reasons.append(f"4-week trend {direction} than seasonal pattern")
    
    # Score persistent deviation
    if current_deviation is not None and abs(current_deviation) > 20:
        if deviation_trend is not None and abs(deviation_trend) > 2:
            shift_score += min(20, abs(current_deviation) / 3)
            direction = "above" if current_deviation > 0 else "below"
            persistence = "accelerating away from" if (current_deviation * deviation_trend) > 0 else "correcting toward"
            shift_reasons.append(f"Persistent {abs(current_deviation):.1f}% {direction} seasonal norm, {persistence} trend")
    
    # Score significant momentum divergence
    if momentum_divergence is not None and abs(momentum_divergence) > 4:
        shift_score += min(15, abs(momentum_divergence) * 2)
        direction = "stronger" if momentum_divergence > 0 else "weaker"
        shift_reasons.append(f"Short-term momentum significantly {direction} than seasonal expectation")
    
    return {
        'group_name': group_name,
        'shift_score': min(100, shift_score),
        'shift_reasons': shift_reasons,
        'approaching_inflections': approaching_inflections,
        'current_deviation': current_deviation,
        'deviation_trend': deviation_trend,
        'momentum_divergence': momentum_divergence,
        'trends_4w': trends_4w,
        'trends_8w': trends_8w,
        'pattern': pattern,
        'current_pattern': current_year_pattern,
        'num_symbols': num_symbols
    }

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
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'scan_params' not in st.session_state:
    st.session_state.scan_params = None
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'individual_results' not in st.session_state:
    st.session_state.individual_results = None
if 'individual_params' not in st.session_state:
    st.session_state.individual_params = None

@st.cache_data
def load_stock_classifications():
    """Load stock classifications from CSV file"""
    classifications = pd.read_csv('stock_sectors.csv')
    sector_stocks = classifications.groupby('sector')['symbol'].apply(list).to_dict()
    industry_stocks = classifications.groupby('industry')['symbol'].apply(list).to_dict()
    return sector_stocks, industry_stocks

@st.cache_data(ttl=3600)
def fetch_market_data(symbol):
    """Fetch market data for a given symbol"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="max", interval='1d')
        if hist.empty:
            return None
        return hist['Close']
    except Exception as e:
        return None

def get_current_week():
    """Get current week number"""
    return datetime.now().isocalendar()[1]

def find_seasonal_inflection_points(pattern):
    """Find major peaks and troughs in the seasonal pattern"""
    values = pattern.values
    weeks = pattern.index.values
    
    # Use higher prominence and distance to find only major inflection points
    min_prominence = 10  # Filter out noise
    min_distance = 4     # Minimum 4 weeks between inflection points
    
    # Find peaks (local maxima)
    peaks, peak_props = find_peaks(values, prominence=min_prominence, distance=min_distance)
    
    # Find troughs (local minima)
    troughs, trough_props = find_peaks(-values, prominence=min_prominence, distance=min_distance)
    
    inflection_points = []
    
    # Add peaks with strength calculation
    for peak in peaks:
        if peak < len(weeks):
            # Calculate trend strength by looking at surrounding weeks
            start_idx = max(0, peak - 3)
            end_idx = min(len(values), peak + 4)
            trend_range = max(values[start_idx:end_idx]) - min(values[start_idx:end_idx])
            
            inflection_points.append({
                'week': weeks[peak],
                'value': values[peak],
                'type': 'peak',
                'strength': peak_props['prominences'][list(peaks).index(peak)],
                'trend_range': trend_range
            })
    
    # Add troughs with strength calculation
    for trough in troughs:
        if trough < len(weeks):
            # Calculate trend strength by looking at surrounding weeks
            start_idx = max(0, trough - 3)
            end_idx = min(len(values), trough + 4)
            trend_range = max(values[start_idx:end_idx]) - min(values[start_idx:end_idx])
            
            inflection_points.append({
                'week': weeks[trough],
                'value': values[trough],
                'type': 'trough',
                'strength': trough_props['prominences'][list(troughs).index(trough)],
                'trend_range': trend_range
            })
    
    # Filter out weak inflection points
    significant_points = [point for point in inflection_points if point['trend_range'] > 15]
    
    return sorted(significant_points, key=lambda x: x['week'])

def calculate_deviation_from_seasonal(current_pattern, historical_pattern, current_week):
    """Calculate deviation from historical seasonal pattern"""
    if current_pattern is None or len(current_pattern) == 0:
        return None, None
    
    # Get overlapping weeks
    common_weeks = set(current_pattern.index) & set(historical_pattern.index)
    if len(common_weeks) == 0:
        return None, None
    
    # Calculate deviations for each week
    deviations = {}
    for week in common_weeks:
        if week <= current_week:
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

def calculate_longer_term_trend(current_pattern, historical_pattern, current_week):
    """Calculate longer-term trend divergence (4-8 week periods)"""
    if current_pattern is None or len(current_pattern) < 8:
        return None, None
    
    trends_4w = {}
    trends_8w = {}
    
    for period, trends_dict in [(4, trends_4w), (8, trends_8w)]:
        if current_week >= period:
            # Get recent weeks for current year
            recent_current_weeks = [w for w in current_pattern.index 
                                  if current_week - period + 1 <= w <= current_week]
            
            # Get corresponding historical weeks
            recent_historical_weeks = [w for w in historical_pattern.index 
                                     if current_week - period + 1 <= w <= current_week]
            
            if len(recent_current_weeks) >= 3 and len(recent_historical_weeks) >= 3:
                # Calculate trends
                current_values = [current_pattern.loc[w] for w in sorted(recent_current_weeks)]
                historical_values = [historical_pattern.loc[w] for w in sorted(recent_historical_weeks)]
                
                current_trend = np.polyfit(range(len(current_values)), current_values, 1)[0]
                historical_trend = np.polyfit(range(len(historical_values)), historical_values, 1)[0]
                
                trends_dict['current'] = current_trend
                trends_dict['historical'] = historical_trend
                trends_dict['divergence'] = current_trend - historical_trend
    
    return trends_4w, trends_8w

def detect_momentum_shifts(current_pattern, historical_pattern, current_week):
    """Detect momentum shifts from seasonal expectations"""
    if current_pattern is None or len(current_pattern) < 3:
        return None, None, None
    
    # Get the last few weeks of data
    recent_weeks = [w for w in current_pattern.index if w <= current_week and w >= max(1, current_week - 3)]
    if len(recent_weeks) < 3:
        return None, None, None
    
    recent_weeks = sorted(recent_weeks)
    
    # Calculate current momentum
    current_values = [current_pattern.loc[w] for w in recent_weeks]
    current_momentum = np.polyfit(range(len(current_values)), current_values, 1)[0]
    
    # Calculate expected seasonal momentum
    historical_values = [historical_pattern.loc[w] for w in recent_weeks if w in historical_pattern.index]
    if len(historical_values) < 3:
        return None, None, None
    
    historical_momentum = np.polyfit(range(len(historical_values)), historical_values, 1)[0]
    
    # Calculate momentum divergence
    momentum_divergence = current_momentum - historical_momentum
    
    return current_momentum, historical_momentum, momentum_divergence

def calculate_group_seasonality(symbols, num_years):
    """Calculate seasonality pattern for a group of symbols"""
    all_patterns = []
    weekly_directions = {week: {'up': 0, 'down': 0} for week in range(1, 53)}
    valid_symbols = 0
    current_year_patterns = []
    all_stock_patterns = {}
    
    # Show progress for larger groups
    if len(symbols) > 10:
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

def analyze_trend_shifts_for_group(symbols, group_name, num_years):
    """Analyze trend shift indicators for a group"""
    # Get basic seasonality data
    pattern, pattern_strength, num_symbols, current_year_pattern, correlation_leaders, weekly_directions = calculate_group_seasonality(symbols, num_years)
    
    if pattern is None:
        return None
    
    current_week = get_current_week()
    
    # 1. Find inflection points
    inflection_points = find_seasonal_inflection_points(pattern)
    
    # Find approaching inflection points (within 4 weeks)
    approaching_inflections = []
    for point in inflection_points:
        weeks_away = point['week'] - current_week
        if 0 <= weeks_away <= 4:
            approaching_inflections.append({
                **point,
                'weeks_away': weeks_away
            })
    
    # 2. Calculate deviation from seasonal norm
    current_deviation, deviation_trend = calculate_deviation_from_seasonal(
        current_year_pattern, pattern, current_week
    )
    
    # 3. Calculate longer-term trends
    trends_4w, trends_8w = calculate_longer_term_trend(
        current_year_pattern, pattern, current_week
    )
    
    # 4. Enhanced momentum detection
    current_momentum, historical_momentum, momentum_divergence = detect_momentum_shifts(
        current_year_pattern, pattern, current_week
    )
    
    # Calculate trend shift score
    shift_score = 0
    shift_reasons = []
    
    # Score approaching inflection points
    if approaching_inflections:
        for inflection in approaching_inflections:
            base_score = min(25, inflection['trend_range'])
            
            if inflection['weeks_away'] == 0:
                shift_score += base_score
                shift_reasons.append(f"At major seasonal {inflection['type']} (Week {inflection['week']})")
            elif inflection['weeks_away'] <= 2:
                shift_score += base_score * 0.7
                shift_reasons.append(f"{inflection['weeks_away']} weeks from major seasonal {inflection['type']} (Week {inflection['week']})")
            else:
                shift_score += base_score * 0.4
                shift_reasons.append(f"{inflection['weeks_away']} weeks from seasonal {inflection['type']} (Week {inflection['week']})")
    
    # Score longer-term trend divergences
    if trends_8w and abs(trends_8w.get('divergence', 0)) > 3:
        divergence_score = min(30, abs(trends_8w['divergence']) * 3)
        shift_score += divergence_score
        direction = "stronger" if trends_8w['divergence'] > 0 else "weaker"
        shift_reasons.append(f"8-week trend {direction} than seasonal pattern")
    
    if trends_4w and abs(trends_4w.get('divergence', 0)) > 4:
        divergence_score = min(20, abs(trends_4w['divergence']) * 2.5)
        shift_score += divergence_score
        direction = "stronger" if trends_4w['divergence'] > 0 else "weaker"
        shift_reasons.append(f"4-week trend {direction} than seasonal pattern")
    
    # Score persistent deviation
    if current_deviation is not None and abs(current_deviation) > 20:
        if deviation_trend is not None and abs(deviation_trend) > 2:
            shift_score += min(20, abs(current_deviation) / 3)
            direction = "above" if current_deviation > 0 else "below"
            persistence = "accelerating away from" if (current_deviation * deviation_trend) > 0 else "correcting toward"
            shift_reasons.append(f"Persistent {abs(current_deviation):.1f}% {direction} seasonal norm, {persistence} trend")
    
    # Score significant momentum divergence
    if momentum_divergence is not None and abs(momentum_divergence) > 4:
        shift_score += min(15, abs(momentum_divergence) * 2)
        direction = "stronger" if momentum_divergence > 0 else "weaker"
        shift_reasons.append(f"Short-term momentum significantly {direction} than seasonal expectation")
    
    return {
        'group_name': group_name,
        'shift_score': min(100, shift_score),
        'shift_reasons': shift_reasons,
        'approaching_inflections': approaching_inflections,
        'current_deviation': current_deviation,
        'deviation_trend': deviation_trend,
        'momentum_divergence': momentum_divergence,
        'trends_4w': trends_4w,
        'trends_8w': trends_8w,
        'pattern': pattern,
        'current_pattern': current_year_pattern,
        'num_symbols': num_symbols
    }

def analyze_market_wide_shifts(trend_shifts):
    """Analyze market-wide shift patterns with enhanced sensitivity"""
    if not trend_shifts:
        return None
    
    current_week = get_current_week()
    total_groups = len(trend_shifts)
    
    # Group by upcoming inflection points
    inflection_clusters = {}
    
    for shift_data in trend_shifts:
        if shift_data.get('approaching_inflections'):
            for inflection in shift_data['approaching_inflections']:
                week = inflection['week']
                inflection_type = inflection['type']
                
                key = f"{inflection_type}_week_{week}"
                if key not in inflection_clusters:
                    inflection_clusters[key] = {
                        'week': week,
                        'type': inflection_type,
                        'groups': [],
                        'weeks_away': inflection['weeks_away']
                    }
                
                inflection_clusters[key]['groups'].append({
                    'name': shift_data['group_name'],
                    'score': shift_data['shift_score'],
                    'num_symbols': shift_data['num_symbols']
                })
    
    # Find significant market-wide patterns (ADJUSTED THRESHOLDS)
    market_shifts = []
    
    for cluster_key, cluster_data in inflection_clusters.items():
        group_count = len(cluster_data['groups'])
        percentage = (group_count / total_groups) * 100
        
        # Calculate weighted importance
        total_stocks = sum(group['num_symbols'] for group in cluster_data['groups'])
        avg_score = sum(group['score'] for group in cluster_data['groups']) / group_count
        
        # LOWERED THRESHOLDS for market-wide shift detection:
        high_score_groups = len([g for g in cluster_data['groups'] if g['score'] >= 60])
        
        is_significant = (
            percentage >= 8 or          # Lowered from 15% to 8%
            group_count >= 5 or         # Lowered from 10 to 5 groups
            (high_score_groups >= 3 and avg_score >= 50)  # Lowered thresholds
        )
        
        if is_significant:
            market_shifts.append({
                'type': cluster_data['type'],
                'week': cluster_data['week'],
                'weeks_away': cluster_data['weeks_away'],
                'group_count': group_count,
                'percentage': percentage,
                'total_stocks': total_stocks,
                'avg_score': avg_score,
                'groups': sorted(cluster_data['groups'], key=lambda x: x['score'], reverse=True)
            })
    
    # NEW: Detect broad trend patterns (not just inflection points)
    trend_pattern_analysis = analyze_broad_trend_patterns(trend_shifts)
    
    # Sort by significance
    market_shifts.sort(key=lambda x: x['percentage'], reverse=True)
    
    # Analyze trend direction consensus
    trend_consensus = {
        'bullish_trends': 0,
        'bearish_trends': 0,
        'mixed_signals': 0
    }
    
    for shift_data in trend_shifts:
        bullish_signals = 0
        bearish_signals = 0
        
        for reason in shift_data.get('shift_reasons', []):
            if any(word in reason.lower() for word in ['stronger', 'above', 'peak']):
                bullish_signals += 1
            elif any(word in reason.lower() for word in ['weaker', 'below', 'trough']):
                bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            trend_consensus['bullish_trends'] += 1
        elif bearish_signals > bullish_signals:
            trend_consensus['bearish_trends'] += 1
        else:
            trend_consensus['mixed_signals'] += 1
    
    return {
        'market_shifts': market_shifts,
        'trend_consensus': trend_consensus,
        'total_groups_analyzed': total_groups,
        'broad_patterns': trend_pattern_analysis  # NEW
    }

def analyze_broad_trend_patterns(trend_shifts):
    """NEW: Analyze broad market trend patterns beyond just inflection points"""
    if not trend_shifts:
        return None
    
    total_groups = len(trend_shifts)
    
    # Count common trend patterns
    pattern_counts = {
        'above_seasonal_norm': 0,
        'below_seasonal_norm': 0,
        'stronger_momentum': 0,
        'weaker_momentum': 0,
        'accelerating_away': 0,
        'correcting_toward': 0,
        '8_week_stronger': 0,
        '4_week_stronger': 0
    }
    
    for shift_data in trend_shifts:
        reasons = shift_data.get('shift_reasons', [])
        
        for reason in reasons:
            reason_lower = reason.lower()
            if 'above seasonal norm' in reason_lower:
                pattern_counts['above_seasonal_norm'] += 1
            elif 'below seasonal norm' in reason_lower:
                pattern_counts['below_seasonal_norm'] += 1
                
            if 'accelerating away from' in reason_lower:
                pattern_counts['accelerating_away'] += 1
            elif 'correcting toward' in reason_lower:
                pattern_counts['correcting_toward'] += 1
                
            if '8-week trend stronger' in reason_lower:
                pattern_counts['8_week_stronger'] += 1
            elif '4-week trend stronger' in reason_lower:
                pattern_counts['4_week_stronger'] += 1
                
            if 'momentum significantly stronger' in reason_lower:
                pattern_counts['stronger_momentum'] += 1
            elif 'momentum significantly weaker' in reason_lower:
                pattern_counts['weaker_momentum'] += 1
    
    # Calculate percentages and identify dominant patterns
    broad_patterns = []
    
    for pattern, count in pattern_counts.items():
        percentage = (count / total_groups) * 100
        if percentage >= 20:  # If 20%+ of groups show this pattern
            broad_patterns.append({
                'pattern': pattern.replace('_', ' ').title(),
                'count': count,
                'percentage': percentage
            })
    
    # Sort by prevalence
    broad_patterns.sort(key=lambda x: x['percentage'], reverse=True)
    
    return {
        'patterns': broad_patterns,
        'total_analyzed': total_groups
    }

def create_seasonality_chart(seasonal_pattern, pattern_strength, current_year_pattern, group_name, num_years, num_symbols, weekly_directions, trend_shift_data=None):
    """Create enhanced seasonality chart with trend shift indicators"""
    current_week = get_current_week()
    
    fig = go.Figure()
    
    # Create the main seasonal pattern
    x_values = list(range(1, len(seasonal_pattern) + 1))
    y_values = seasonal_pattern.values
    
    # Add invisible baseline for connection
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(size=1, color='lightgray', opacity=0.2),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add colored line segments based on consistency
    for i in range(len(x_values) - 1):
        week = i + 1
        consistency = pattern_strength.get(week, 0)
        
        # Get direction info
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
        
        # Color and width based on consistency
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
        
        hover_text = f"Week {week}<br>Value: {y_values[i]:.1f}<br>{consistency_text}"
        
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
    
    # Add current year pattern
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
    
    # Calculate y-axis ranges
    y_min = min(seasonal_pattern.values)
    y_max = max(seasonal_pattern.values)
    if current_year_pattern is not None:
        y_min = min(y_min, min(current_year_pattern.values))
        y_max = max(y_max, max(current_year_pattern.values))
    y_range = y_max - y_min
    y_min = max(0, y_min - (y_range * 0.1))
    y_max = y_max + (y_range * 0.1)
    
    # Add trend shift score to title
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

def display_trend_shift_card(shift_data, compact=False):
    """Display a trend shift analysis card"""
    with st.container():
        if not compact:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{shift_data['group_name']}** ({shift_data.get('group_type', 'Group')})")
                st.markdown(f"*Based on {shift_data['num_symbols']} stocks*")
                
                if shift_data['shift_reasons']:
                    for reason in shift_data['shift_reasons'][:3]:
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

def display_market_wide_alert(market_analysis):
    """Display market-wide shift alert section with enhanced pattern detection"""
    # Check for both inflection-based shifts and broad patterns
    has_inflection_shifts = market_analysis and market_analysis['market_shifts']
    has_broad_patterns = market_analysis and market_analysis.get('broad_patterns') and market_analysis['broad_patterns']['patterns']
    
    if not has_inflection_shifts and not has_broad_patterns:
        return
    
    st.markdown("## 🌊 **MARKET-WIDE PATTERNS DETECTED**")
    st.markdown("---")
    
    # Display inflection point clusters if they exist
    if has_inflection_shifts:
        st.markdown("### 📍 Coordinated Seasonal Inflection Points")
        for shift in market_analysis['market_shifts']:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                shift_emoji = "📈" if shift['type'] == 'trough' else "📉"
                st.markdown(f"#### {shift_emoji} Market-Wide Seasonal {shift['type'].title()}")
                st.markdown(f"**Week {shift['week']}** ({shift['weeks_away']} weeks away)")
                st.markdown(f"Affecting **{shift['group_count']} industries** ({shift['percentage']:.1f}% of analyzed groups)")
                st.markdown(f"Covering **{shift['total_stocks']:,} stocks** with avg shift score of **{shift['avg_score']:.1f}**")
            
            with col2:
                st.markdown("**Top Industries:**")
                for group in shift['groups'][:5]:
                    st.markdown(f"• {group['name'][:25]}...")
            
            with col3:
                st.markdown("**Risk Level:**")
                if shift['percentage'] >= 20:  # Adjusted thresholds
                    st.error("🔴 EXTREME")
                elif shift['percentage'] >= 12:
                    st.warning("🟠 HIGH") 
                else:
                    st.info("🟡 MODERATE")
    
    # NEW: Display broad market trend patterns
    if has_broad_patterns:
        st.markdown("### 🌍 Broad Market Trend Patterns")
        
        broad_data = market_analysis['broad_patterns']
        total_analyzed = broad_data['total_analyzed']
        
        st.markdown(f"**Widespread patterns affecting large portions of the market:**")
        
        for i, pattern in enumerate(broad_data['patterns'][:5], 1):  # Show top 5 patterns
            pattern_name = pattern['pattern']
            count = pattern['count']
            percentage = pattern['percentage']
            
            # Color code by severity
            if percentage >= 40:
                alert_level = "🔴 CRITICAL"
                alert_color = "error"
            elif percentage >= 30:
                alert_level = "🟠 MAJOR"
                alert_color = "warning"
            else:
                alert_level = "🟡 NOTABLE"
                alert_color = "info"
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{i}. {pattern_name}**")
                st.markdown(f"Pattern observed across {count} industries")
                
            with col2:
                st.metric("Market Coverage", f"{percentage:.1f}%")
                
            with col3:
                if alert_color == "error":
                    st.error(alert_level)
                elif alert_color == "warning":
                    st.warning(alert_level)
                else:
                    st.info(alert_level)
        
        # Interpretation help
        if broad_data['patterns']:
            dominant_pattern = broad_data['patterns'][0]
            st.markdown("### 📊 Pattern Analysis")
            
            interpretation = get_pattern_interpretation(dominant_pattern['pattern'], dominant_pattern['percentage'])
            st.markdown(f"**Market Interpretation:** {interpretation}")
    
    # Market sentiment overview
    if market_analysis['trend_consensus']:
        st.markdown("### Market Sentiment Analysis")
        consensus = market_analysis['trend_consensus']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Bullish Trends", consensus['bullish_trends'])
        with col2:
            st.metric("Bearish Trends", consensus['bearish_trends'])  
        with col3:
            st.metric("Mixed Signals", consensus['mixed_signals'])
    
    st.markdown("---")

def get_pattern_interpretation(pattern_name, percentage):
    """Provide interpretation for detected broad market patterns"""
    interpretations = {
        'Above Seasonal Norm': f"The market is running significantly hotter than historical seasonal expectations across {percentage:.0f}% of analyzed industries. This could indicate broad-based strength or potential overextension.",
        'Below Seasonal Norm': f"Market performance is lagging historical seasonal patterns across {percentage:.0f}% of industries. This may signal broader weakness or potential opportunity.",
        'Accelerating Away': f"Trends are accelerating away from seasonal norms in {percentage:.0f}% of industries, suggesting momentum-driven moves that may be unsustainable.",
        'Correcting Toward': f"Industries are correcting back toward seasonal norms in {percentage:.0f}% of cases, indicating mean reversion forces at work.",
        '8 Week Stronger': f"Medium-term (8-week) trends are outpacing seasonal expectations in {percentage:.0f}% of industries, suggesting sustained momentum.",
        '4 Week Stronger': f"Short-term (4-week) trends are stronger than seasonal patterns in {percentage:.0f}% of industries, indicating recent acceleration."
    }
    
    return interpretations.get(pattern_name, f"Widespread {pattern_name.lower()} pattern detected across {percentage:.0f}% of analyzed industries.")

def display_trend_shift_tabs(trend_shifts, market_analysis):
    """Display the tabbed results interface"""
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 High Alert (50+)", "⚠️ Moderate (25-49)", "📊 All Results", "📈 Market Analysis"])
    
    with tab1:
        high_alert = [ts for ts in trend_shifts if ts['shift_score'] >= 50]
        if high_alert:
            st.markdown("### Individual Groups with High Trend Shift Probability")
            for shift_data in high_alert:
                display_trend_shift_card(shift_data)
        else:
            st.info("No individual groups currently showing high alert trend shift signals.")
    
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
    
    with tab4:
        if market_analysis:
            st.markdown("### Detailed Market Analysis")
            
            if market_analysis['market_shifts']:
                st.markdown("#### Market-Wide Seasonal Inflection Points")
                for i, shift in enumerate(market_analysis['market_shifts'], 1):
                    with st.expander(f"{i}. {shift['type'].title()} in Week {shift['week']} - {shift['group_count']} industries"):
                        st.markdown("**Affected Industries:**")
                        for group in shift['groups']:
                            st.markdown(f"• **{group['name']}**: {group['score']:.0f}/100 score, {group['num_symbols']} stocks")
            else:
                st.info("No significant market-wide patterns detected. Industries are showing individual rather than coordinated shifts.")
            
            # Additional market metrics
            st.markdown("#### Market Breadth Analysis")
            total_groups = market_analysis['total_groups_analyzed']
            high_alert_count = len([ts for ts in trend_shifts if ts['shift_score'] >= 50])
            moderate_alert_count = len([ts for ts in trend_shifts if 25 <= ts['shift_score'] < 50])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Participation", f"{(len(trend_shifts)/total_groups*100):.1f}%", 
                         help=f"{len(trend_shifts)} out of {total_groups} groups showing trend shift signals")
            with col2:
                st.metric("High Alert Rate", f"{(high_alert_count/total_groups*100):.1f}%")
            with col3:
                st.metric("Total Alert Rate", f"{((high_alert_count + moderate_alert_count)/total_groups*100):.1f}%")
        else:
            st.info("Market analysis data not available.")

def display_summary_statistics(groups_scanned, trend_shifts, market_analysis):
    """Display the scanner summary statistics"""
    st.markdown("### Scanner Summary")
    
    # Add current week context at the top
    current_week = get_current_week()
    current_date = datetime.now().strftime("%B %d, %Y")
    st.info(f"📅 **Current Position:** Week {current_week} of 2025 ({current_date})")
    
    # Add download button prominently at the top
    if trend_shifts and len(trend_shifts) > 0:
        download_data = create_download_data(trend_shifts, market_analysis, groups_scanned)
        st.download_button(
            label="📥 Download Complete Results (CSV)",
            data=download_data,
            file_name=f"trend_shift_scan_week{current_week}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download all scan results including market analysis",
            key="download_results"
        )
        st.markdown("---")  # Add separator
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Groups Scanned", groups_scanned)
    
    with col2:
        st.metric("Trend Shifts Detected", len(trend_shifts))
    
    with col3:
        high_count = len([ts for ts in trend_shifts if ts['shift_score'] >= 50])
        st.metric("High Alert", high_count)
    
    with col4:
        market_shifts_count = len(market_analysis['market_shifts']) if market_analysis and market_analysis['market_shifts'] else 0
        st.metric("Market-Wide Shifts", market_shifts_count)
    
    with col5:
        if trend_shifts:
            avg_score = sum(ts['shift_score'] for ts in trend_shifts) / len(trend_shifts)
            st.metric("Avg Shift Score", f"{avg_score:.1f}")
        else:
            st.metric("Avg Shift Score", "0.0")

def create_download_data(trend_shifts, market_analysis, groups_scanned):
    """Create CSV download data from scan results"""
    # Create main results dataframe
    results_data = []
    
    for shift_data in trend_shifts:
        # Get next inflection info
        next_inflection = ""
        weeks_away = ""
        if shift_data.get('approaching_inflections'):
            next_inf = min(shift_data['approaching_inflections'], key=lambda x: x['weeks_away'])
            next_inflection = f"{next_inf['type'].title()} Week {next_inf['week']}"
            weeks_away = next_inf['weeks_away']
        
        # Join shift reasons
        reasons = "; ".join(shift_data.get('shift_reasons', []))
        
        results_data.append({
            'Group_Name': shift_data['group_name'],
            'Group_Type': shift_data.get('group_type', 'Unknown'),
            'Shift_Score': shift_data['shift_score'],
            'Num_Stocks': shift_data['num_symbols'],
            'Next_Inflection': next_inflection,
            'Weeks_Away': weeks_away,
            'Current_Deviation': shift_data.get('current_deviation', ''),
            'Shift_Reasons': reasons,
            'Scan_Timestamp': st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_scan_time else ''
        })
    
    # Convert to CSV
    df = pd.DataFrame(results_data)
    csv = df.to_csv(index=False)
    
    # Add market-wide analysis as header comments
    if market_analysis and market_analysis['market_shifts']:
        header_lines = [
            "# Market-Wide Shift Analysis",
            f"# Scan Date: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_scan_time else 'Unknown'}",
            f"# Total Groups Scanned: {groups_scanned}",
            f"# Market-Wide Shifts Detected: {len(market_analysis['market_shifts'])}",
            ""
        ]
        
        for shift in market_analysis['market_shifts']:
            header_lines.append(f"# Market Shift: {shift['type'].title()} Week {shift['week']} - {shift['group_count']} industries ({shift['percentage']:.1f}% of market)")
        
        header_lines.extend(["", "# Individual Group Results:", ""])
        
        csv = "\n".join(header_lines) + csv
    
    return csv

def handle_individual_analysis(selected_group, view_type, num_years, symbols, sector_stocks, industry_stocks):
    """Handle individual analysis mode with enhanced caching"""
    
    # Create mode-specific cache key
    current_params = {
        'analysis_mode': 'individual',  # Add this to distinguish from scanner mode
        'selected_group': selected_group,
        'view_type': view_type,
        'num_years': num_years
    }
    
    # Check cache with mode-specific validation
    use_cached = (
        st.session_state.individual_params is not None and
        st.session_state.individual_params.get('analysis_mode') == 'individual' and
        st.session_state.individual_params == current_params and 
        st.session_state.individual_results is not None
    )
    
    # Add refresh button for individual analysis too
    if use_cached:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("📋 Showing cached individual analysis results.")
        with col2:
            if st.button("🔄 Refresh Analysis"):
                use_cached = False
                st.session_state.individual_results = None
                st.session_state.individual_params = None
    
    if not use_cached:
        # Run new analysis
        try:
            with st.spinner(f"Analyzing {selected_group}..."):
                # Get trend shift data
                trend_shift_data = analyze_trend_shifts_for_group(symbols, selected_group, num_years)
                
                if trend_shift_data and trend_shift_data['pattern'] is not None:
                    # Get original seasonality data for chart
                    pattern, pattern_strength, num_symbols, current_year_pattern, correlation_leaders, weekly_directions = calculate_group_seasonality(def main():
    st.title('🔄 Advanced Seasonality & Trend Shift Analysis')
    
    # Load classifications
    sector_stocks, industry_stocks = load_stock_classifications()
    
    # Sidebar controls
    with st.sidebar:
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
            max_available_years = 25
            try:
                for symbol in symbols[:3]:  # Check first few symbols
                    data = fetch_market_data(symbol)
                    if data is not None:
                        years_available = (data.index.max() - data.index.min()).days / 365
                        max_available_years = max(max_available_years, int(years_available))
            except:
                pass
            
            num_years = st.slider(
                'Select number of years to analyze:', 
                min_value=1,
                max_value=min(30, max_available_years),
                value=min(15, max_available_years),
                help="Choose how many years of historical data to include in the analysis."
            )
            
        else:
            # Trend shift scanner mode
            st.markdown("### Trend Shift Scanner Settings")
            scan_type = st.radio(
                "Scan Type",
                ["All Industries", "All Sectors", "Custom Selection"]
            )
            
            if scan_type == "Custom Selection":
                st.markdown("### Custom Selection")
                selected_sectors = st.multiselect("Select Sectors:", list(sector_stocks.keys()))
                selected_industries = st.multiselect("Select Industries:", list(industry_stocks.keys()))
            
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
        handle_individual_analysis(selected_group, view_type, num_years, symbols, sector_stocks, industry_stocks)
    else:
        handle_scanner_mode(scan_type, num_years, min_shift_score, sector_stocks, industry_stocks)

                    pattern, pattern_strength, num_symbols, current_year_pattern, correlation_leaders, weekly_directions = calculate_group_seasonality(symbols, num_years)
                    
                    # Store results in session state
                    st.session_state.individual_results = {
                        'trend_shift_data': trend_shift_data,
                        'pattern': pattern,
                        'pattern_strength': pattern_strength,
                        'num_symbols': num_symbols,
                        'current_year_pattern': current_year_pattern,
                        'correlation_leaders': correlation_leaders,
                        'weekly_directions': weekly_directions
                    }
                    st.session_state.individual_params = current_params
                    
                    # Clear scanner cache to prevent conflicts
                    if st.session_state.scan_params and st.session_state.scan_params.get('analysis_mode') != 'individual':
                        st.session_state.scan_params = None
                        st.session_state.scan_results = None
                else:
                    st.session_state.individual_results = None
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.individual_results = None
    
    # Display results (cached or new)
    if st.session_state.individual_results:
        results = st.session_state.individual_results
        
        # Create enhanced chart with trend shift indicators
        fig = create_seasonality_chart(
            results['pattern'], results['pattern_strength'], results['current_year_pattern'], 
            selected_group, num_years, results['num_symbols'], results['weekly_directions'],
            results['trend_shift_data']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display trend shift analysis
        if results['trend_shift_data']['shift_score'] > 0:
            st.markdown("### 🚨 Trend Shift Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Trend Shift Score", 
                    f"{results['trend_shift_data']['shift_score']:.0f}/100",
                    delta="High Alert" if results['trend_shift_data']['shift_score'] > 50 else "Moderate" if results['trend_shift_data']['shift_score'] > 25 else "Low"
                )
            
            with col2:
                if results['trend_shift_data']['current_deviation'] is not None:
                    st.metric(
                        "Deviation from Seasonal",
                        f"{results['trend_shift_data']['current_deviation']:.1f}%",
                        delta="Above" if results['trend_shift_data']['current_deviation'] > 0 else "Below"
                    )
            
            with col3:
                if results['trend_shift_data']['approaching_inflections']:
                    next_inflection = min(results['trend_shift_data']['approaching_inflections'], key=lambda x: x['weeks_away'])
                    st.metric(
                        f"Next {next_inflection['type'].title()}",
                        f"Week {next_inflection['week']}",
                        delta=f"{next_inflection['weeks_away']} weeks away"
                    )
            
            if results['trend_shift_data']['shift_reasons']:
                st.markdown("**Key Indicators:**")
                for reason in results['trend_shift_data']['shift_reasons']:
                    st.markdown(f"• {reason}")
        else:
            st.info("No significant trend shift indicators detected at this time.")
        
        # Sidebar statistics
        display_individual_sidebar_stats(results)
        
    else:
        st.error("No valid data available for the selected group and time period.")

def get_groups_to_scan(scan_type, sector_stocks, industry_stocks):
    """Get the list of groups to scan based on scan type - DEPRECATED"""
    # This function is no longer used as we handle group selection directly in handle_scanner_mode
    return []

def display_scan_results():
    """Display the scan results with all tabs and analysis"""
    results = st.session_state.scan_results
    trend_shifts = results['trend_shifts']
    market_analysis = results['market_analysis']
    groups_scanned = results['groups_scanned']
    
    if trend_shifts:
        # Sort by shift score
        trend_shifts.sort(key=lambda x: x['shift_score'], reverse=True)
        
        # Display market-wide analysis first if significant patterns found
        if market_analysis and market_analysis['market_shifts']:
            display_market_wide_alert(market_analysis)
        
        # Display individual results in tabs
        display_trend_shift_tabs(trend_shifts, market_analysis)
        
        # Summary statistics with download button
        display_summary_statistics(groups_scanned, trend_shifts, market_analysis)
        
    else:
        min_shift_score = st.session_state.scan_params.get('min_shift_score', 25)
        st.info(f"No groups found with trend shift scores above {min_shift_score}. Try lowering the minimum threshold.")

def handle_custom_selection_scanner(sector_stocks, industry_stocks, num_years, min_shift_score):
    """Handle custom selection scanner mode with proper caching"""
    st.markdown("### 🔍 Custom Selection Scanner")
    
    # Custom selection interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Select Sectors")
        selected_sectors = st.multiselect(
            "Choose sectors to analyze:",
            list(sector_stocks.keys()),
            help="Select one or more sectors for analysis"
        )
    
    with col2:
        st.markdown("#### Select Industries")  
        selected_industries = st.multiselect(
            "Choose industries to analyze:",
            list(industry_stocks.keys()),
            help="Select one or more industries for analysis"
        )
    
    # Build groups to scan
    groups_to_scan = []
    for sector in selected_sectors:
        groups_to_scan.append((sector, sector_stocks[sector], "Sector"))
    for industry in selected_industries:
        groups_to_scan.append((industry, industry_stocks[industry], "Industry"))
    
    if groups_to_scan:
        # Create cache key for custom selection
        current_scan_params = {
            'analysis_mode': 'custom_scanner',
            'scan_type': 'Custom Selection',
            'selected_sectors': sorted(selected_sectors),
            'selected_industries': sorted(selected_industries),
            'num_years': num_years,
            'min_shift_score': min_shift_score
        }
        
        use_cached_scan = (
            st.session_state.scan_params is not None and
            st.session_state.scan_params.get('analysis_mode') == 'custom_scanner' and
            st.session_state.scan_params == current_scan_params and 
            st.session_state.scan_results is not None
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if use_cached_scan:
                st.success(f"📋 Cached results for {len(groups_to_scan)} groups")
            else:
                st.info(f"Ready to scan {len(groups_to_scan)} selected groups")
        
        with col2:
            if st.button("🚀 Start Custom Scan", disabled=use_cached_scan):
                st.session_state.scan_results = None
                run_trend_shift_scan(groups_to_scan, num_years, min_shift_score, current_scan_params)
                st.rerun()
        
        with col3:
            if use_cached_scan and st.button("🔄 Refresh Custom"):
                st.session_state.scan_results = None
                st.session_state.scan_params = None
                st.rerun()
        
        # Display results if available
        if st.session_state.scan_results and use_cached_scan:
            display_scan_results()
        elif not use_cached_scan and st.session_state.scan_results is None:
            st.info("Click 'Start Custom Scan' to analyze your selection.")
    
    else:
        st.warning("Please select at least one sector or industry to scan.")

# Enhanced main function that includes all modes
def main_enhanced():
    st.title('🔄 Advanced Seasonality & Trend Shift Analysis')
    
    # Display app info
    display_app_info()
    
    # Load classifications
    try:
        sector_stocks, industry_stocks = load_stock_classifications()
    except Exception as e:
        st.error(f"Error loading stock classifications: {str(e)}")
        st.error("Please make sure 'stock_sectors.csv' file is available.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## 🎛️ Analysis Controls")
        
        analysis_mode = st.radio(
            "Select Analysis Mode",
            ["Individual Analysis", "Trend Shift Scanner"],
            help="Choose between analyzing a single group or scanning multiple groups"
        )
        
        st.markdown("---")
        
        if analysis_mode == "Individual Analysis":
            st.markdown("### 📊 Individual Analysis Settings")
            
            view_type = st.radio(
                "Analysis Type",
                ["Sector", "Industry"]
            )
            
            if view_type == "Sector":
                selected_group = st.selectbox(
                    "Select Sector",
                    list(sector_stocks.keys()),
                    help="Choose a sector to analyze"
                )
                symbols = sector_stocks[selected_group]
            else:
                selected_group = st.selectbox(
                    "Select Industry", 
                    list(industry_stocks.keys()),
                    help="Choose an industry to analyze"
                )
                symbols = industry_stocks[selected_group]
            
            # Years slider
            num_years = st.slider(
                'Years of historical data:', 
                min_value=5,
                max_value=25,
                value=15,
                help="More years = more reliable patterns, but less recent relevance"
            )
            
            # Quick stats
            st.markdown(f"**Selected:** {selected_group}")
            st.markdown(f"**Stocks:** {len(symbols)}")
            
        else:
            st.markdown("### 🔍 Scanner Settings")
            
            scan_type = st.radio(
                "Scan Type",
                ["All Industries", "All Sectors", "Custom Selection"],
                help="Choose scope of analysis"
            )
            
            num_years = st.slider(
                'Years of historical data:', 
                min_value=5,
                max_value=25,
                value=15,
                help="Years of historical data for pattern analysis"
            )
            
            min_shift_score = st.slider(
                'Minimum shift score:',
                min_value=10,
                max_value=80,
                value=25,
                help="Only show groups above this threshold"
            )
            
            # Scan scope info
            if scan_type == "All Industries":
                st.markdown(f"**Scope:** {len(industry_stocks)} industries")
            elif scan_type == "All Sectors": 
                st.markdown(f"**Scope:** {len(sector_stocks)} sectors")
            else:
                st.markdown("**Scope:** Custom selection")
    
    # Main content area
    if analysis_mode == "Individual Analysis":
        handle_individual_analysis(selected_group, view_type, num_years, symbols, sector_stocks, industry_stocks)
    else:
        if scan_type == "Custom Selection":
            handle_custom_selection_scanner(sector_stocks, industry_stocks, num_years, min_shift_score)
        else:
            handle_scanner_mode(scan_type, num_years, min_shift_score, sector_stocks, industry_stocks)

if __name__ == "__main__":
    main_enhanced()

def display_individual_sidebar_stats(results):
    """Display sidebar statistics for individual analysis"""
    st.sidebar.markdown(f"### Group Statistics")
    st.sidebar.markdown(f"Number of stocks analyzed: {results['num_symbols']}")
    
    # Display correlation leaders
    if results['correlation_leaders']:
        st.sidebar.markdown("### Top Pattern Followers")
        num_to_show = min(10, len(results['correlation_leaders']))
        
        for i, (symbol, corr) in enumerate(results['correlation_leaders'][:num_to_show], 1):
            correlation_pct = corr * 100
            st.sidebar.markdown(f"{i}. {symbol}: {correlation_pct:.1f}%")

def handle_scanner_mode(scan_type, num_years, min_shift_score, sector_stocks, industry_stocks):
    """Handle trend shift scanner mode with enhanced caching"""
    
    st.markdown("### 🔍 Trend Shift Scanner Results")
    
    # Determine which groups to scan
    if scan_type == "All Industries":
        groups_to_scan = [(name, symbols, "Industry") for name, symbols in industry_stocks.items()]
    elif scan_type == "All Sectors":
        groups_to_scan = [(name, symbols, "Sector") for name, symbols in sector_stocks.items()]
    else:
        # Custom selection - show the interface
        handle_custom_selection_scanner(sector_stocks, industry_stocks, num_years, min_shift_score)
        return
    
    # Create a unique cache key that includes the analysis mode
    current_scan_params = {
        'analysis_mode': 'scanner',  # Add this to distinguish from individual mode
        'scan_type': scan_type,
        'groups_to_scan': [g[0] for g in groups_to_scan],
        'num_years': num_years,
        'min_shift_score': min_shift_score
    }
    
    # Check cache with mode-specific key
    use_cached_scan = (
        st.session_state.scan_params == current_scan_params and 
        st.session_state.scan_results is not None and
        st.session_state.last_scan_time is not None
    )
    
    # Add status and refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if use_cached_scan:
            scan_time = st.session_state.last_scan_time
            st.success(f"📋 Using cached scan from {scan_time.strftime('%H:%M:%S')} - Switch tabs freely!")
        else:
            st.info(f"Ready to scan {len(groups_to_scan)} groups")
    
    with col2:
        if st.button("🚀 Start Scan", disabled=use_cached_scan):
            use_cached_scan = False
            st.session_state.scan_results = None
    
    with col3:
        if use_cached_scan and st.button("🔄 Refresh"):
            use_cached_scan = False
            st.session_state.scan_results = None
            st.rerun()
    
    if groups_to_scan:
        if not use_cached_scan:
            # Run new scan
            run_trend_shift_scan(groups_to_scan, num_years, min_shift_score, current_scan_params)
            st.rerun()  # Refresh to show results
        
        # Display results (cached or new)
        if st.session_state.scan_results:
            display_scan_results()
        else:
            st.info("Click 'Start Scan' to begin analysis.")
    else:
        st.warning("No groups available to scan.")

def handle_custom_selection_scanner(sector_stocks, industry_stocks, num_years, min_shift_score):
    """Handle custom selection scanner mode"""
    st.markdown("### 🔍 Custom Selection Scanner")
    
    # Custom selection interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Select Sectors")
        selected_sectors = st.multiselect(
            "Choose sectors to analyze:",
            list(sector_stocks.keys()),
            help="Select one or more sectors for analysis"
        )
    
    with col2:
        st.markdown("#### Select Industries")  
        selected_industries = st.multiselect(
            "Choose industries to analyze:",
            list(industry_stocks.keys()),
            help="Select one or more industries for analysis"
        )
    
    # Build groups to scan
    groups_to_scan = []
    for sector in selected_sectors:
        groups_to_scan.append((sector, sector_stocks[sector], "Sector"))
    for industry in selected_industries:
        groups_to_scan.append((industry, industry_stocks[industry], "Industry"))
    
    if groups_to_scan:
        # Create cache key for custom selection
        current_scan_params = {
            'analysis_mode': 'custom_scanner',
            'scan_type': 'Custom Selection',
            'selected_sectors': sorted(selected_sectors),
            'selected_industries': sorted(selected_industries),
            'num_years': num_years,
            'min_shift_score': min_shift_score
        }
        
        use_cached_scan = (
            st.session_state.scan_params == current_scan_params and 
            st.session_state.scan_results is not None
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if use_cached_scan:
                st.success(f"📋 Cached results for {len(groups_to_scan)} groups")
            else:
                st.info(f"Ready to scan {len(groups_to_scan)} selected groups")
        
        with col2:
            if st.button("🚀 Start Custom Scan", disabled=use_cached_scan):
                run_trend_shift_scan(groups_to_scan, num_years, min_shift_score, current_scan_params)
                st.rerun()
        
        with col3:
            if use_cached_scan and st.button("🔄 Refresh Custom"):
                st.session_state.scan_results = None
                st.rerun()
        
        # Display results if available
        if st.session_state.scan_results and use_cached_scan:
            display_scan_results()
        elif not use_cached_scan and st.session_state.scan_results is None:
            st.info("Click 'Start Custom Scan' to analyze your selection.")
    
    else:
        st.warning("Please select at least one sector or industry to scan.")

def get_groups_to_scan(scan_type, sector_stocks, industry_stocks):
    """Get the list of groups to scan based on scan type"""
    if scan_type == "All Industries":
        return [(name, symbols, "Industry") for name, symbols in industry_stocks.items()]
    elif scan_type == "All Sectors":
        return [(name, symbols, "Sector") for name, symbols in sector_stocks.items()]
    else:
        # Custom selection
        groups_to_scan = []
        # Note: selected_sectors and selected_industries would be defined in the sidebar
        # For now, return empty list for custom selection
        return groups_to_scan

def handle_scanner_mode(scan_type, num_years, min_shift_score, sector_stocks, industry_stocks):
    """Handle trend shift scanner mode with enhanced caching"""
    
    st.markdown("### 🔍 Trend Shift Scanner Results")
    
    # Determine which groups to scan
    if scan_type == "All Industries":
        groups_to_scan = [(name, symbols, "Industry") for name, symbols in industry_stocks.items()]
    elif scan_type == "All Sectors":
        groups_to_scan = [(name, symbols, "Sector") for name, symbols in sector_stocks.items()]
    else:
        # This case is handled in handle_custom_selection_scanner
        return
    
    # Create a mode-specific cache key to prevent cross-contamination
    current_scan_params = {
        'analysis_mode': 'scanner',  # Key addition to separate from individual mode
        'scan_type': scan_type,
        'groups_to_scan': [g[0] for g in groups_to_scan],
        'num_years': num_years,
        'min_shift_score': min_shift_score
    }
    
    # Check cache with strict mode checking
    use_cached_scan = (
        st.session_state.scan_params is not None and
        st.session_state.scan_params.get('analysis_mode') == 'scanner' and
        st.session_state.scan_params == current_scan_params and 
        st.session_state.scan_results is not None and
        st.session_state.last_scan_time is not None
    )
    
    # Add status and control buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if use_cached_scan:
            scan_time = st.session_state.last_scan_time
            st.success(f"📋 Cached scan from {scan_time.strftime('%H:%M:%S')} - Switch tabs freely!")
        else:
            st.info(f"Ready to scan {len(groups_to_scan)} groups")
    
    with col2:
        if st.button("🚀 Start Scan", disabled=use_cached_scan, help="Start new trend shift analysis"):
            # Clear cache and trigger scan
            st.session_state.scan_results = None
            run_trend_shift_scan(groups_to_scan, num_years, min_shift_score, current_scan_params)
            st.rerun()
    
    with col3:
        if use_cached_scan and st.button("🔄 Refresh", help="Force new scan with same parameters"):
            st.session_state.scan_results = None
            st.session_state.scan_params = None
            st.rerun()
    
    # Display results
    if st.session_state.scan_results and use_cached_scan:
        display_scan_results()
    elif not use_cached_scan and st.session_state.scan_results is None:
        st.info("Click 'Start Scan' to begin trend shift analysis.")

def run_trend_shift_scan(groups_to_scan, num_years, min_shift_score, current_scan_params):
    """Run the trend shift scan and store results with enhanced state management"""
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
        
        # Analyze market-wide patterns
        market_analysis = analyze_market_wide_shifts(trend_shifts)
        
        # Store results in session state with timestamp
        st.session_state.scan_results = {
            'trend_shifts': trend_shifts,
            'market_analysis': market_analysis,
            'groups_scanned': len(groups_to_scan)
        }
        st.session_state.scan_params = current_scan_params
        st.session_state.last_scan_time = datetime.now()
        
        # Clear individual analysis cache to prevent conflicts
        if st.session_state.individual_params:
            st.session_state.individual_params = None
            st.session_state.individual_results = None

def display_scan_results():
    """Display the scan results with all tabs and analysis"""
    results = st.session_state.scan_results
    trend_shifts = results['trend_shifts']
    market_analysis = results['market_analysis']
    groups_scanned = results['groups_scanned']
    
    if trend_shifts:
        # Sort by shift score
        trend_shifts.sort(key=lambda x: x['shift_score'], reverse=True)
        
        # Display market-wide analysis first if significant patterns found
        if market_analysis and market_analysis['market_shifts']:
            display_market_wide_alert(market_analysis)
        
        # Display individual results in tabs
        display_trend_shift_tabs(trend_shifts, market_analysis)
        
        # Summary statistics
        display_summary_statistics(groups_scanned, trend_shifts, market_analysis)
        
    else:
        min_shift_score = st.session_state.scan_params.get('min_shift_score', 25)
        st.info(f"No groups found with trend shift scores above {min_shift_score}. Try lowering the minimum threshold or selecting different groups.")

def handle_custom_selection_scanner(sector_stocks, industry_stocks, num_years, min_shift_score):
    """Handle custom selection scanner mode"""
    st.markdown("### 🔍 Custom Selection Scanner")
    
    # Custom selection interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Select Sectors")
        selected_sectors = st.multiselect(
            "Choose sectors to analyze:",
            list(sector_stocks.keys()),
            help="Select one or more sectors for analysis"
        )
    
    with col2:
        st.markdown("#### Select Industries")  
        selected_industries = st.multiselect(
            "Choose industries to analyze:",
            list(industry_stocks.keys()),
            help="Select one or more industries for analysis"
        )
    
    # Build groups to scan
    groups_to_scan = []
    for sector in selected_sectors:
        groups_to_scan.append((sector, sector_stocks[sector], "Sector"))
    for industry in selected_industries:
        groups_to_scan.append((industry, industry_stocks[industry], "Industry"))
    
    if groups_to_scan:
        st.info(f"Ready to scan {len(groups_to_scan)} selected groups")
        
        # Check for cached results
        current_scan_params = {
            'scan_type': 'Custom Selection',
            'groups_to_scan': [g[0] for g in groups_to_scan],
            'num_years': num_years,
            'min_shift_score': min_shift_score
        }
        
        use_cached_scan = (st.session_state.scan_params == current_scan_params and 
                          st.session_state.scan_results is not None)
        
        if use_cached_scan:
            st.success("📋 Using cached results for your selection")
            display_scan_results()
        else:
            if st.button("🚀 Start Custom Scan"):
                run_trend_shift_scan(groups_to_scan, num_years, min_shift_score, current_scan_params)
                st.rerun()
    else:
        st.warning("Please select at least one sector or industry to scan.")

def display_app_info():
    """Display app information and instructions"""
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        ### 🔄 Advanced Seasonality & Trend Shift Analysis
        
        This app analyzes seasonal patterns and detects potential trend shifts in market sectors and industries.
        
        **Two Analysis Modes:**
        
        **1. Individual Analysis**
        - Analyze a specific sector or industry
        - View detailed seasonal patterns and current trend shift indicators
        - See correlation leaders and pattern followers
        
        **2. Trend Shift Scanner** 
        - Scan all industries, all sectors, or custom selection
        - Detect market-wide seasonal shift patterns
        - Get alerts when multiple groups approach similar inflection points
        
        **Key Features:**
        - 🎯 **Longer-term Focus**: Filters out noise, focuses on 4+ week trends
        - 🌊 **Market-wide Detection**: Alerts when 15%+ of groups show coordinated shifts
        - 💾 **Smart Caching**: Results persist when you switch tabs or adjust settings
        - 📊 **Enhanced Charts**: Visual indicators for approaching seasonal turning points
        
        **Trend Shift Scoring:**
        - **50+ Points**: High Alert - Major seasonal shifts likely
        - **25-49 Points**: Moderate - Some shift indicators present  
        - **<25 Points**: Low - Limited shift signals detected
        """)

# Enhanced main function that includes all modes
def main_enhanced():
    st.title('🔄 Advanced Seasonality & Trend Shift Analysis')
    
    # Display app info
    display_app_info()
    
    # Load classifications
    try:
        sector_stocks, industry_stocks = load_stock_classifications()
    except Exception as e:
        st.error(f"Error loading stock classifications: {str(e)}")
        st.error("Please make sure 'stock_sectors.csv' file is available.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## 🎛️ Analysis Controls")
        
        analysis_mode = st.radio(
            "Select Analysis Mode",
            ["Individual Analysis", "Trend Shift Scanner"],
            help="Choose between analyzing a single group or scanning multiple groups"
        )
        
        st.markdown("---")
        
        if analysis_mode == "Individual Analysis":
            st.markdown("### 📊 Individual Analysis Settings")
            
            view_type = st.radio(
                "Analysis Type",
                ["Sector", "Industry"]
            )
            
            if view_type == "Sector":
                selected_group = st.selectbox(
                    "Select Sector",
                    list(sector_stocks.keys()),
                    help="Choose a sector to analyze"
                )
                symbols = sector_stocks[selected_group]
            else:
                selected_group = st.selectbox(
                    "Select Industry", 
                    list(industry_stocks.keys()),
                    help="Choose an industry to analyze"
                )
                symbols = industry_stocks[selected_group]
            
            # Years slider
            num_years = st.slider(
                'Years of historical data:', 
                min_value=5,
                max_value=25,
                value=15,
                help="More years = more reliable patterns, but less recent relevance"
            )
            
            # Quick stats
            st.markdown(f"**Selected:** {selected_group}")
            st.markdown(f"**Stocks:** {len(symbols)}")
            
        else:
            st.markdown("### 🔍 Scanner Settings")
            
            scan_type = st.radio(
                "Scan Type",
                ["All Industries", "All Sectors", "Custom Selection"],
                help="Choose scope of analysis"
            )
            
            num_years = st.slider(
                'Years of historical data:', 
                min_value=5,
                max_value=25,
                value=15,
                help="Years of historical data for pattern analysis"
            )
            
            min_shift_score = st.slider(
                'Minimum shift score:',
                min_value=10,
                max_value=80,
                value=25,
                help="Only show groups above this threshold"
            )
            
            # Scan scope info
            if scan_type == "All Industries":
                st.markdown(f"**Scope:** {len(industry_stocks)} industries")
            elif scan_type == "All Sectors": 
                st.markdown(f"**Scope:** {len(sector_stocks)} sectors")
            else:
                st.markdown("**Scope:** Custom selection")
    
    # Main content area
    if analysis_mode == "Individual Analysis":
        handle_individual_analysis(selected_group, view_type, num_years, symbols, sector_stocks, industry_stocks)
    else:
        if scan_type == "Custom Selection":
            handle_custom_selection_scanner(sector_stocks, industry_stocks, num_years, min_shift_score)
        else:
            handle_scanner_mode(scan_type, num_years, min_shift_score, sector_stocks, industry_stocks)

if __name__ == "__main__":
    main_enhanced()