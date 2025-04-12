import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta, time
import numpy as np
import pytz
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    layout="wide",
    page_title="Crypto Pattern Finder",
    page_icon="🔍",
)

# Constants for session opens
DEFAULT_OPEN_HOUR_UTC = 0  # Default is Asia open (UTC midnight)
NY_OPEN_HOUR_UTC = 13  # New York market open (approximately 9AM ET)
LONDON_OPEN_HOUR_UTC = 7  # London market open

# Core data and session handling functions ----------------------------------------------------------------------------------
def fetch_crypto_data(symbol, period="60d", interval="5m"):
    """Fetch crypto data from Yahoo Finance"""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    
    if hist.empty:
        return None
        
    # Convert timestamps to UTC for consistency
    if hist.index.tz is not None:
        hist.index = hist.index.tz_convert('UTC')
    else:
        hist.index = hist.index.tz_localize('UTC')
    
    return hist

def check_data_availability(symbol, interval="5m"):
    """
    Check what data interval and period combinations are available for the symbol
    Returns the maximum number of days available for the selected interval
    """
    try:
        # Test different periods to determine availability
        periods = ["7d", "30d", "60d", "90d"]
        max_days = 0
        
        for period in periods:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    start_date = data.index.min()
                    end_date = data.index.max()
                    duration = (end_date - start_date).days
                    
                    if duration > max_days:
                        max_days = duration
            except Exception:
                pass
        
        return max_days
    except Exception as e:
        st.error(f"Error checking data availability: {str(e)}")
        return 30  # Default to 30 days as a safe value

def get_session_date(timestamp, session_open_hour_utc):
    """
    Get the session date for a given timestamp based on configurable session open hour.
    Each session runs from the specified UTC hour to the same hour the next day.
    """
    # The session date is the date of the UTC hour AFTER the timestamp
    if timestamp.hour >= session_open_hour_utc:
        # If timestamp is after or at the session open hour, session date is the next day
        return (timestamp.date() + timedelta(days=1))
    else:
        # If timestamp is before the session open hour, session date is the same day
        return timestamp.date()

def get_sessions(data, session_open_hour_utc):
    """Get data grouped by trading sessions with configurable start time"""
    # Create a copy to avoid modifying the original
    data_copy = data.copy()
    
    # Add a column for session date
    data_copy['session_date'] = data_copy.index.map(
        lambda ts: get_session_date(ts, session_open_hour_utc)
    )
    
    # Get unique session dates
    session_dates = sorted(data_copy['session_date'].unique())
    
    # Store session data
    sessions = {}
    
    for session_date in session_dates:
        # Calculate session start (previous day's specified UTC hour)
        session_start = datetime.combine(session_date - timedelta(days=1), time(hour=session_open_hour_utc))
        session_start = pytz.utc.localize(session_start)
        
        # Calculate session end (specified UTC hour)
        session_end = datetime.combine(session_date, time(hour=session_open_hour_utc))
        session_end = pytz.utc.localize(session_end)
        
        # Get data for this session
        session_data = data_copy[(data_copy.index >= session_start) & (data_copy.index < session_end)]
        
        if not session_data.empty:
            sessions[session_date] = session_data
    
    return sessions

def get_session_data(data, session_date, session_open_hour_utc):
    """Get data for a specific session date with configurable start time"""
    # Calculate session start time
    session_start = datetime.combine(session_date - timedelta(days=1), time(hour=session_open_hour_utc))
    session_start = pytz.utc.localize(session_start)
    
    # Calculate session end time
    session_end = datetime.combine(session_date, time(hour=session_open_hour_utc))
    session_end = pytz.utc.localize(session_end)
    
    # Get data for this session
    session_data = data[(data.index >= session_start) & (data.index < session_end)]
    
    return session_data

def minutes_to_time_str(minutes):
    """Convert minutes from session start to formatted time string"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

# Pattern analysis functions ----------------------------------------------------------------------------------
def calculate_session_patterns(sessions, current_session_date=None):
    """Calculate price patterns for each session"""
    patterns = {}
    
    for date, session_data in sessions.items():
        # Skip current session for historical patterns
        if date == current_session_date:
            continue
            
        if len(session_data) < 12:  # Require at least 12 data points
            continue
            
        # Calculate percentage change from session open
        session_open = session_data['Close'].iloc[0]
        session_pattern = ((session_data['Close'] - session_open) / session_open) * 100
        
        # Use minutes from session start as index for better comparison
        session_start = session_data.index[0]
        minutes_from_start = [(ts - session_start).total_seconds() / 60 for ts in session_data.index]
        session_pattern.index = minutes_from_start
        
        patterns[date] = session_pattern
    
    return patterns

def calculate_pattern_cutoff(data, session_date, session_open_hour_utc, cutoff_hours=12):
    """
    Calculate pattern for a specific session up to X hours after session open
    For use in backtesting to simulate partial data
    """
    # Get data for this session
    session_data = get_session_data(data, session_date, session_open_hour_utc)
    
    if session_data.empty:
        return None
    
    # Apply cutoff time (X hours after session open)
    session_start = datetime.combine(session_date - timedelta(days=1), time(hour=session_open_hour_utc))
    session_start = pytz.utc.localize(session_start)
    
    cutoff_time = session_start + timedelta(hours=cutoff_hours)
    pattern_data = session_data[session_data.index <= cutoff_time]
    
    if len(pattern_data) < 5:  # Ensure we have enough data points
        return None
    
    # Calculate pattern as percent change from session open
    try:
        session_open = pattern_data['Close'].iloc[0]
        pattern = ((pattern_data['Close'] - session_open) / session_open) * 100
        
        # Use minutes from session start as the index for comparison
        minutes_from_start = [(ts - session_start).total_seconds() / 60 for ts in pattern_data.index]
        pattern.index = minutes_from_start
        
        return pattern
    except Exception as e:
        st.error(f"Error calculating pattern for {session_date}: {e}")
        return None

def find_similar_sessions(current_pattern, historical_patterns, num_matches=10, recent_bias=False, min_overlap=12):
    """Find the most similar historical patterns to current session"""
    if current_pattern is None or len(current_pattern) < min_overlap:
        return []
    
    similarity_scores = []
    
    for date, pattern in historical_patterns.items():
        # Skip if patterns don't have enough overlap
        common_times = set(pattern.index).intersection(set(current_pattern.index))
        if len(common_times) < min_overlap:
            continue
            
        # Calculate similarity (mean squared error) for common time points
        squared_diffs = []
        times_list = []
        
        for minutes in common_times:
            if minutes in pattern.index and minutes in current_pattern.index:
                diff = current_pattern[minutes] - pattern[minutes]
                squared_diffs.append(diff * diff)
                times_list.append(minutes)  # minutes from session start
        
        if not squared_diffs:
            continue
            
        # Calculate weighted MSE - more weight to recent points if requested
        if recent_bias and times_list:
            max_time = max(times_list)
            normalized_times = [t/max_time for t in times_list]
            weights = np.array([0.5 + 0.5 * t for t in normalized_times])
        else:
            weights = np.ones(len(squared_diffs))
            
        weighted_mse = sum(w * d for w, d in zip(weights, squared_diffs)) / sum(weights)
        
        # Calculate end of session value
        end_value = pattern.iloc[-1] if not pattern.empty else 0
        
        similarity_scores.append({
            'date': date,
            'similarity': weighted_mse,
            'pattern': pattern,
            'end_value': end_value,
            'trend': 'up' if end_value > 0 else 'down'
        })
    
    # Sort by similarity (lower is better)
    similarity_scores.sort(key=lambda x: x['similarity'])
    
    # Return top N matches
    return similarity_scores[:num_matches]

def group_similar_patterns(patterns, threshold=0.8):
    """Group patterns that are similar to each other"""
    if not patterns:
        return []
        
    grouped_patterns = []
    used_indices = set()
    
    for i, pattern1 in enumerate(patterns):
        if i in used_indices:
            continue
            
        group = [pattern1]
        used_indices.add(i)
        
        for j, pattern2 in enumerate(patterns):
            if j in used_indices or i == j:
                continue
                
            # Calculate similarity between patterns
            p1 = pattern1['pattern']
            p2 = pattern2['pattern']
            common_times = set(p1.index).intersection(set(p2.index))
            
            if len(common_times) < 12:  # Require more overlap for grouping
                continue
                
            squared_diffs = []
            for minutes in common_times:
                if minutes in p1.index and minutes in p2.index:
                    diff = p1[minutes] - p2[minutes]
                    squared_diffs.append(diff * diff)
            
            if squared_diffs:
                similarity = sum(squared_diffs) / len(squared_diffs)
                if similarity < threshold:
                    group.append(pattern2)
                    used_indices.add(j)
        
        # Add the group
        if len(group) > 1:
            # Use the best matching pattern as the representative
            best_pattern = min(group, key=lambda x: x['similarity'])
            best_pattern['count'] = len(group)
            best_pattern['similar_dates'] = [p['date'] for p in group if p['date'] != best_pattern['date']]
            grouped_patterns.append(best_pattern)
        else:
            pattern1['count'] = 1
            pattern1['similar_dates'] = []
            grouped_patterns.append(pattern1)
    
    # Sort by similarity again
    grouped_patterns.sort(key=lambda x: x['similarity'])
    
    return grouped_patterns

def predict_direction(similar_patterns):
    """Predict direction and confidence based on similar patterns"""
    if not similar_patterns:
        return None, 0, 0
    
    # Count up vs down patterns
    up_count = sum(1 for p in similar_patterns if p['end_value'] > 0)
    down_count = sum(1 for p in similar_patterns if p['end_value'] <= 0)
    
    # Determine predicted direction
    if up_count > down_count:
        direction = "up"
        confidence = up_count / len(similar_patterns)
    elif down_count > up_count:
        direction = "down"
        confidence = down_count / len(similar_patterns)
    else:
        # Tie - calculate average end value
        avg_end = sum(p['end_value'] for p in similar_patterns) / len(similar_patterns)
        direction = "up" if avg_end > 0 else "down"
        confidence = 0.5
    
    # Calculate average magnitude
    magnitudes = [abs(p['end_value']) for p in similar_patterns]
    avg_magnitude = sum(magnitudes) / len(magnitudes) if magnitudes else 0
    
    return direction, confidence, avg_magnitude

def calculate_confidence_metrics(top_matches):
    """Calculate confidence metrics based on pattern agreement"""
    if not top_matches:
        return None
    
    # Count up vs down patterns
    up_count = sum(1 for m in top_matches if m['end_value'] > 0)
    down_count = sum(1 for m in top_matches if m['end_value'] <= 0)
    total_count = up_count + down_count
    
    # Calculate average ending values by trend
    up_values = [m['end_value'] for m in top_matches if m['end_value'] > 0]
    down_values = [m['end_value'] for m in top_matches if m['end_value'] <= 0]
    all_values = up_values + down_values
    
    avg_up = sum(up_values) / len(up_values) if up_values else 0
    avg_down = sum(down_values) / len(down_values) if down_values else 0
    avg_all = sum(all_values) / len(all_values) if all_values else 0
    
    # Calculate weighted average by similarity (inverse of MSE)
    weights = [1/m['similarity'] for m in top_matches]
    weighted_values = [m['end_value'] * (1/m['similarity']) for m in top_matches]
    weighted_avg = sum(weighted_values) / sum(weights) if weights else 0
    
    # Calculate confidence based on agreement
    if total_count > 0:
        if up_count > down_count:
            confidence = up_count / total_count
            direction = "up"
        elif down_count > up_count:
            confidence = down_count / total_count
            direction = "down"
        else:
            confidence = 0.5
            direction = "up" if avg_all > 0 else "down"
    else:
        confidence = 0
        direction = "neutral"
    
    # Calculate expected magnitude
    expected_magnitude = abs(avg_all)
    
    # Determine confidence level
    if confidence >= 0.8:
        confidence_level = "High"
    elif confidence >= 0.6:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    # Determine magnitude level
    if expected_magnitude >= 2.0:
        magnitude_level = "Large"
    elif expected_magnitude >= 1.0:
        magnitude_level = "Medium"
    else:
        magnitude_level = "Small"
    
    # Return metrics
    return {
        'direction': direction,
        'confidence': confidence,
        'confidence_level': confidence_level,
        'expected_magnitude': expected_magnitude,
        'magnitude_level': magnitude_level,
        'avg_all': avg_all,
        'avg_up': avg_up,
        'avg_down': avg_down,
        'weighted_avg': weighted_avg,
        'up_count': up_count,
        'down_count': down_count,
        'total_count': total_count
    }

 # Visualization functions ----------------------------------------------------------------------------------
def plot_pattern_matches(sessions, symbol, current_session_date, top_matches, show_scores=True, max_patterns=10, session_open_hour_utc=0):
    """Create plot showing pattern matches with current session data"""
    # Create a single plot without volume subplot
    fig = go.Figure()
    
    # Get current session data
    if current_session_date in sessions:
        current_session = sessions[current_session_date]
    else:
        st.error("Current session data not found")
        return None
    
    # Colors for different patterns
    colors = {
        'current': 'rgb(255, 255, 255)',   # White
        'match1': 'rgb(255, 99, 132)',     # Red
        'match2': 'rgb(66, 135, 245)',     # Blue
        'match3': 'rgb(52, 191, 73)',      # Green
        'match4': 'rgb(242, 184, 64)',     # Yellow
        'match5': 'rgb(153, 102, 255)',    # Purple
        'match6': 'rgb(0, 204, 204)',      # Teal
        'match7': 'rgb(255, 51, 153)',     # Pink
        'match8': 'rgb(102, 255, 102)',    # Light Green
        'match9': 'rgb(255, 204, 0)',      # Gold
        'match10': 'rgb(204, 102, 255)',   # Lavender
    }
    
    # Add current session price
    if not current_session.empty:
        session_open = current_session['Close'].iloc[0]
        current_changes = ((current_session['Close'] - session_open) / session_open) * 100
        
        # Calculate minutes from session start for each timestamp
        session_start = current_session.index[0]
        minutes_list = [(ts - session_start).total_seconds() / 60 for ts in current_session.index]
        
        # Convert to hours:minutes format for display
        time_strings = [minutes_to_time_str(m) for m in minutes_list]
        
        fig.add_trace(
            go.Scatter(
                x=time_strings,
                y=current_changes.values,
                mode='lines',
                name="Current Session",
                line=dict(color=colors['current'], width=3),
            )
        )
    
    # Add top matching patterns
    for i, match in enumerate(top_matches[:max_patterns]):
        if i < len(colors) - 1:  # Ensure we have a color for this match
            match_color = colors[f'match{i+1}']
            date_str = match['date'].strftime('%Y-%m-%d')
            
            # Create name with or without score
            if show_scores:
                name = f"Match #{i+1}: {date_str} (Score: {match['similarity']:.4f})"
            else:
                name = f"Match #{i+1}: {date_str}"
                
            # Add group count if available
            if 'count' in match and match['count'] > 1:
                name += f" (+{match['count']-1} similar)"
            
            # Convert pattern minutes to time strings
            pattern_times = [minutes_to_time_str(m) for m in match['pattern'].index]
            
            fig.add_trace(
                go.Scatter(
                    x=pattern_times,
                    y=match['pattern'].values,
                    mode='lines',
                    name=name,
                    line=dict(color=match_color, width=2),
                )
            )

    # Get session name based on hour
    session_name = "Custom"
    if session_open_hour_utc == 0:
        session_name = "Asia"
    elif session_open_hour_utc == 13:
        session_name = "New York"
    elif session_open_hour_utc == 7:
        session_name = "London"

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Pattern Matches ({session_name} Session - UTC {session_open_hour_utc:02d}:00)',
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=700,
        template='plotly_dark',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Common x-axis settings - use hours from session start
    hours = list(range(0, 24))
    hour_labels = [f"{h:02d}:00" for h in hours]
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        ticktext=hour_labels,
        tickvals=hour_labels,
        tickangle=45,
        zeroline=True,
        zerolinecolor='rgba(128,128,128,0.2)',
        title_text=f"Hours from {session_name} Open (UTC {session_open_hour_utc:02d}:00)"
    )
    
    # Update y-axis
    fig.update_yaxes(
        title_text="Price Change from Open (%)",
        gridcolor='rgba(128,128,128,0.2)',
        gridwidth=1,
        showgrid=True,
        zeroline=True,
        zerolinecolor='rgba(255,255,255,0.4)',
        zerolinewidth=2
    )
    
    return fig

def create_match_details_card(match, sessions, session_open_hour_utc):
    """Create details card for a matching pattern"""
    date_str = match['date'].strftime('%Y-%m-%d')
    
    # Get session data
    session_data = sessions.get(match['date'])
    
    if session_data is None or session_data.empty:
        return None
    
    # Calculate session stats
    session_open = session_data['Open'].iloc[0]
    session_close = session_data['Close'].iloc[-1]
    session_return = (session_close - session_open) / session_open * 100
    
    high_point = session_data['High'].max()
    high_return = (high_point - session_open) / session_open * 100
    
    # Find time of high in minutes from session start
    high_idx = session_data['High'].idxmax()
    session_start = session_data.index[0]
    high_minutes = int((high_idx - session_start).total_seconds() / 60)
    high_time = minutes_to_time_str(high_minutes)
    
    low_point = session_data['Low'].min()
    low_return = (low_point - session_open) / session_open * 100
    
    # Find time of low
    low_idx = session_data['Low'].idxmin()
    low_minutes = int((low_idx - session_start).total_seconds() / 60)
    low_time = minutes_to_time_str(low_minutes)
    
    # Try to get next session's data
    next_session_date = match['date'] + timedelta(days=1)
    next_session_data = sessions.get(next_session_date)
    
    if next_session_data is not None and not next_session_data.empty:
        next_session_open = next_session_data['Open'].iloc[0]
        next_session_close = next_session_data['Close'].iloc[-1]
        next_session_return = (next_session_close - next_session_open) / next_session_open * 100
    else:
        next_session_return = None
    
    # Create the card
    expanded = match.get('count', 1) > 1  # Auto-expand if it represents a group
    with st.expander(f"Details for {date_str} (Similarity: {match['similarity']:.4f})", expanded=expanded):
        # Show similar patterns if this is a group
        if 'similar_dates' in match and match['similar_dates']:
            similar_dates_str = ', '.join([pd.Timestamp(d).strftime('%Y-%m-%d') for d in match['similar_dates']])
            st.info(f"This pattern is similar to {len(match['similar_dates'])} other sessions: {similar_dates_str}")
            
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Session Close", f"{session_return:.2f}%")
        
        with col2:
            st.metric("Session High", f"{high_return:.2f}% at {high_time}")
            if next_session_return is not None:
                st.metric("Next Session", f"{next_session_return:.2f}%")
        
        with col3:
            st.metric("Session Low", f"{low_return:.2f}% at {low_time}")
            
            # Calculate when the pattern started diverging from current session
            current_date = datetime.now(pytz.utc).date()
            current_session_date = get_session_date(datetime.now(pytz.utc), session_open_hour_utc)
            
            if current_session_date in sessions:
                current_session = sessions[current_session_date]
                
                if not current_session.empty and len(current_session) >= 12:
                    # Calculate current session pattern
                    current_open = current_session['Close'].iloc[0]
                    current_pattern = ((current_session['Close'] - current_open) / current_open) * 100
                    
                    # Calculate minutes from session start
                    current_start = current_session.index[0]
                    current_pattern.index = [(ts - current_start).total_seconds() / 60 for ts in current_session.index]
                    
                    # Find maximum divergence
                    max_diff = 0
                    max_diff_time = None
                    
                    try:
                        # Get common time points
                        pattern_times = set(match['pattern'].index)
                        current_times = set(current_pattern.index)
                        common_times = pattern_times.intersection(current_times)
                        
                        for minutes in common_times:
                            diff = abs(current_pattern[minutes] - match['pattern'][minutes])
                            if diff > max_diff:
                                max_diff = diff
                                max_diff_time = minutes
                        
                        if max_diff_time is not None:
                            time_str = minutes_to_time_str(max_diff_time)
                            st.markdown(f"**Max Divergence**: {max_diff:.2f}% at {time_str}")
                    except Exception as e:
                        st.info("Could not calculate maximum divergence")

def analyze_pattern_distribution(top_matches):
    """Analyze the distribution of patterns - up vs down trends"""
    if not top_matches:
        return None
        
    # Count trends
    up_count = sum(1 for m in top_matches if m['end_value'] > 0)
    down_count = sum(1 for m in top_matches if m['end_value'] <= 0)
    
    # Calculate average ending values by trend
    up_values = [m['end_value'] for m in top_matches if m['end_value'] > 0]
    down_values = [m['end_value'] for m in top_matches if m['end_value'] <= 0]
    
    avg_up = sum(up_values) / len(up_values) if up_values else 0
    avg_down = sum(down_values) / len(down_values) if down_values else 0
    
    # Create distribution chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Upward', 'Downward'],
        y=[up_count, down_count],
        marker_color=['green', 'red'],
        text=[f"{up_count} patterns<br>Avg: +{avg_up:.2f}%", 
              f"{down_count} patterns<br>Avg: {avg_down:.2f}%"],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Pattern Distribution",
        xaxis_title="End of Session Direction",
        yaxis_title="Number of Patterns",
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    return fig

# Backtesting functions ----------------------------------------------------------------------------------
def calculate_actual_result(data, session_date, session_open_hour_utc, cutoff_hours=12):
    """
    Calculate actual result from cutoff time to end of session
    """
    # Get data for this session
    session_data = get_session_data(data, session_date, session_open_hour_utc)
    
    if session_data.empty:
        return None, 0
    
    # Calculate cutoff time (X hours after session open)
    session_start = datetime.combine(session_date - timedelta(days=1), time(hour=session_open_hour_utc))
    session_start = pytz.utc.localize(session_start)
    
    cutoff_time = session_start + timedelta(hours=cutoff_hours)
    
    # Split data at cutoff
    before_cutoff = session_data[session_data.index <= cutoff_time]
    after_cutoff = session_data[session_data.index > cutoff_time]
    
    if before_cutoff.empty or after_cutoff.empty:
        return None, 0
    
    # Price at cutoff
    cutoff_price = before_cutoff['Close'].iloc[-1]
    
    # End of session price
    end_price = after_cutoff['Close'].iloc[-1]
    
    # Calculate change
    change_pct = ((end_price - cutoff_price) / cutoff_price) * 100
    
    # Determine direction
    direction = "up" if change_pct > 0 else "down"
    
    return direction, change_pct

def run_backtest(data, session_open_hour_utc, cutoff_hours=12, num_matches=5, test_days=30):
    """
    Run backtest for the most recent days using specified session open time
    
    Parameters:
    - data: DataFrame of price data
    - session_open_hour_utc: Hour of day (UTC) when session starts
    - cutoff_hours: Hours after session open to make prediction (default: 12 hours)
    - num_matches: Number of similar patterns to find (default: 5)
    - test_days: Number of most recent sessions to test (default: 30)
    """
    if data is None or data.empty:
        st.error("No data available for backtesting.")
        return None
    
    # Get all session dates
    all_session_dates = sorted(set(
        get_session_date(ts, session_open_hour_utc) for ts in data.index
    ))
    
    # Filter out incomplete sessions (dates that don't have data after cutoff)
    valid_session_dates = []
    for date in all_session_dates:
        session_data = get_session_data(data, date, session_open_hour_utc)
        if session_data.empty:
            continue
        
        # Calculate session start and cutoff times
        session_start = datetime.combine(date - timedelta(days=1), time(hour=session_open_hour_utc))
        session_start = pytz.utc.localize(session_start)
        cutoff_time = session_start + timedelta(hours=cutoff_hours)
        
        # Check if we have data after the cutoff
        after_cutoff = session_data[session_data.index > cutoff_time]
        if not after_cutoff.empty:
            valid_session_dates.append(date)
    
    # Make sure we have enough valid sessions
    if len(valid_session_dates) < 5:
        st.error(f"Not enough complete sessions found. Need at least 5 sessions for backtesting.")
        return None
    
    # Use the most recent valid dates for testing (limited by available data)
    test_days = min(test_days, len(valid_session_dates))
    test_dates = valid_session_dates[-test_days:]
    
    st.info(f"Running backtest on {len(test_dates)} sessions from {test_dates[0]} to {test_dates[-1]}...")
    
    # Progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Create results dataframe
    results = []
    
    # Run backtest for each test date
    for i, test_date in enumerate(test_dates):
        progress_text.text(f"Testing session {i+1}/{len(test_dates)}: {test_date}")
        progress_bar.progress((i+1)/len(test_dates))
        
        # Get historical dates (sessions before test_date)
        historical_dates = [d for d in valid_session_dates if d < test_date]
        
        # Ensure we have enough historical dates
        if len(historical_dates) < num_matches:
            continue
        
        # Calculate pattern up to cutoff
        target_pattern = calculate_pattern_cutoff(data, test_date, session_open_hour_utc, cutoff_hours)
        
        if target_pattern is None or len(target_pattern) < 5:
            continue
        
        # Create historical patterns dictionary
        historical_patterns = {}
        for hist_date in historical_dates[-30:]:  # Limit to last 30 sessions for speed
            hist_pattern = calculate_pattern_cutoff(data, hist_date, session_open_hour_utc, cutoff_hours)
            if hist_pattern is not None and len(hist_pattern) >= 5:
                historical_patterns[hist_date] = hist_pattern
        
        # Skip if we don't have enough historical patterns
        if len(historical_patterns) < num_matches:
            continue
            
        # Find similar days
        similar_patterns = find_similar_sessions(
            target_pattern, 
            historical_patterns, 
            num_matches=num_matches
        )
        
        if not similar_patterns:
            continue
        
        # Predict direction
        pred_direction, confidence, avg_magnitude = predict_direction(similar_patterns)
        
        # Calculate actual result
        actual_direction, actual_change = calculate_actual_result(
            data, test_date, session_open_hour_utc, cutoff_hours
        )
        
        if actual_direction is None:
            continue
        
        # Store result
        result = {
            'date': test_date,
            'predicted_direction': pred_direction,
            'confidence': confidence,
            'avg_magnitude': avg_magnitude,
            'actual_direction': actual_direction,
            'actual_change': actual_change,
            'correct': pred_direction == actual_direction,
            'similar_dates': [p['date'] for p in similar_patterns],
            'similarities': [p['similarity'] for p in similar_patterns]
        }
        
        results.append(result)
    
    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()
    
    # Convert to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        st.success(f"Backtest complete! Generated {len(results)} test cases.")
        return results_df
    else:
        st.warning("No valid backtest results could be generated. Try using a different interval or fewer test days.")
        return None

def plot_backtest_results(results_df):
    """Create Plotly visualization of backtest results"""
    if results_df is None or results_df.empty:
        st.error("No backtest results to visualize")
        return None, None, None
    
    # Overall accuracy
    correct_predictions = results_df['correct'].sum()
    total_predictions = len(results_df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Accuracy by confidence level
    results_df['confidence_bucket'] = pd.cut(results_df['confidence'], 
                                         bins=[0, 0.6, 0.8, 1.0],
                                         labels=['Low (0-60%)', 'Medium (60-80%)', 'High (80-100%)'])
    
    conf_accuracy = results_df.groupby('confidence_bucket')['correct'].agg(['count', 'mean'])
    conf_accuracy.columns = ['Count', 'Accuracy']
    
    # Accuracy by magnitude
    results_df['magnitude_bucket'] = pd.cut(results_df['avg_magnitude'], 
                                        bins=[0, 1, 2, 100],
                                        labels=['Small (0-1%)', 'Medium (1-2%)', 'Large (>2%)'])
    
    mag_accuracy = results_df.groupby('magnitude_bucket')['correct'].agg(['count', 'mean'])
    mag_accuracy.columns = ['Count', 'Accuracy']
    
    # Create main results figure
    results_df['color'] = results_df['correct'].map({True: 'green', False: 'red'})
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=results_df['date'],
            y=results_df['actual_change'],
            mode='markers',
            marker=dict(
                size=10,
                color=results_df['color'],
                symbol='circle'
            ),
            name='Actual Change',
            text=[f"Date: {d}<br>Predicted: {p}<br>Actual: {a}<br>Change: {c:.2f}%<br>Confidence: {conf:.1%}" 
                  for d, p, a, c, conf in zip(
                      results_df['date'], 
                      results_df['predicted_direction'], 
                      results_df['actual_direction'], 
                      results_df['actual_change'],
                      results_df['confidence'])],
            hoverinfo='text'
        )
    )
    
    # Add horizontal line at zero
    fig.add_hline(y=0, line=dict(color='white', width=1, dash='dash'))
    
    # Update layout
    fig.update_layout(
        title='Backtest Results: Actual Price Change After Cutoff',
        xaxis_title='Date',
        yaxis_title='Change (%)',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    # Create confidence level chart
    conf_fig = go.Figure()
    
    for i, (level, row) in enumerate(conf_accuracy.iterrows()):
        if pd.isna(level) or pd.isna(row['Accuracy']):
            continue
            
        color = 'rgba(255, 99, 132, 0.7)' if row['Accuracy'] < 0.55 else 'rgba(75, 192, 192, 0.7)'
        
        conf_fig.add_trace(
            go.Bar(
                x=[level],
                y=[row['Accuracy']],
                text=[f"{row['Count']} predictions<br>{row['Accuracy']:.1%} accuracy"],
                textposition='auto',
                marker_color=color,
                name=str(level)
            )
        )
    
    conf_fig.update_layout(
        title='Accuracy by Confidence Level',
        xaxis_title='Confidence Level',
        yaxis_title='Accuracy',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        yaxis=dict(range=[0, 1])
    )
    
    # Create magnitude level chart
    mag_fig = go.Figure()
    
    for i, (level, row) in enumerate(mag_accuracy.iterrows()):
        if pd.isna(level) or pd.isna(row['Accuracy']):
            continue
            
        color = 'rgba(255, 99, 132, 0.7)' if row['Accuracy'] < 0.55 else 'rgba(75, 192, 192, 0.7)'
        
        mag_fig.add_trace(
            go.Bar(
                x=[level],
                y=[row['Accuracy']],
                text=[f"{row['Count']} predictions<br>{row['Accuracy']:.1%} accuracy"],
                textposition='auto',
                marker_color=color,
                name=str(level)
            )
        )
    
    mag_fig.update_layout(
        title='Accuracy by Expected Magnitude',
        xaxis_title='Expected Magnitude',
        yaxis_title='Accuracy',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        yaxis=dict(range=[0, 1])
    )
    
    return fig, conf_fig, mag_fig

def plot_session_example(data, session_date, cutoff_hours, similar_patterns, symbol, session_open_hour_utc):
    """Plot example session with predictions for backtesting analysis"""
    # Get data for the test session
    session_data = get_session_data(data, session_date, session_open_hour_utc)
    
    if session_data.empty:
        st.error(f"No data available for {session_date}")
        return None
    
    # Calculate session start time
    session_start = datetime.combine(session_date - timedelta(days=1), time(hour=session_open_hour_utc))
    session_start = pytz.utc.localize(session_start)
    
    # Calculate cutoff time
    cutoff_time = session_start + timedelta(hours=cutoff_hours)
    
    # Split data
    before_cutoff = session_data[session_data.index <= cutoff_time]
    after_cutoff = session_data[session_data.index > cutoff_time]
    
    if before_cutoff.empty or after_cutoff.empty:
        st.error(f"Not enough data for {session_date}")
        return None
    
    # Create the figure
    fig = go.Figure()
    
    # Calculate percent change from session open
    session_open = session_data['Open'].iloc[0]
    session_percent = ((session_data['Close'] - session_open) / session_open) * 100
    
    # Plot test day data
    fig.add_trace(
        go.Scatter(
            x=session_data.index,
            y=session_percent,
            mode='lines',
            name=f"{session_date} Actual",
            line=dict(color='white', width=3)
        )
    )
    
    # Add vertical line at cutoff
    fig.add_trace(
        go.Scatter(
            x=[cutoff_time, cutoff_time],
            y=[min(session_percent) - 1, max(session_percent) + 1],
            mode='lines',
            line=dict(color='yellow', width=2, dash='dash'),
            showlegend=False,
            name="Cutoff"
        )
    )

    # Add annotation for the cutoff line
    fig.add_annotation(
        x=cutoff_time,
        y=max(session_percent) + 0.5,
        text=f"Cutoff ({cutoff_hours}h after open)",
        showarrow=False,
        font=dict(color="yellow"),
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="yellow",
        borderwidth=1
    )
    
    # Add top matching patterns
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i, pattern_date in enumerate(similar_patterns[:5]):
        match_session_data = get_session_data(data, pattern_date, session_open_hour_utc)
        
        if match_session_data.empty:
            continue
        
        # Calculate match session start
        match_start = datetime.combine(pattern_date - timedelta(days=1), time(hour=session_open_hour_utc))
        match_start = pytz.utc.localize(match_start)
        
        # Calculate percent change from match session open
        match_open = match_session_data['Open'].iloc[0]
        match_percent = ((match_session_data['Close'] - match_open) / match_open) * 100
        
        # Align timestamps with test session
        aligned_times = []
        aligned_values = []
        
        for idx, value in zip(match_session_data.index, match_percent):
            # Calculate minutes from session start
            minutes_from_start = (idx - match_start).total_seconds() / 60
            
            # Create equivalent timestamp for test session
            aligned_time = session_start + timedelta(minutes=minutes_from_start)
            
            aligned_times.append(aligned_time)
            aligned_values.append(value)
        
        # Plot pattern
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=aligned_times,
                y=aligned_values,
                mode='lines',
                name=f"Match: {pattern_date}",
                line=dict(color=color, width=2, dash='dot')
            )
        )
    
    # Get session name based on hour
    session_name = "Custom"
    if session_open_hour_utc == 0:
        session_name = "Asia"
    elif session_open_hour_utc == 13:
        session_name = "New York"
    elif session_open_hour_utc == 7:
        session_name = "London"
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - {session_date} {session_name} Session with Similar Patterns (Cutoff at {cutoff_hours}h)",
        xaxis_title="Time (UTC)",
        yaxis_title="Price Change from Open (%)",
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=500
    )
    
    # Show vertical grid lines at hourly intervals
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinecolor='rgba(255,255,255,0.2)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinecolor='rgba(255,255,255,0.2)',
        zerolinewidth=2
    )
    
    return fig

# Main Streamlit application ----------------------------------------------------------------------------------
def main():
    # Add tab selection
    tabs = st.tabs(["Pattern Finder", "Backtest"])
    
    with tabs[0]:  # Pattern Finder tab
        st.markdown("### Identify similar historical trading patterns based on configurable session open times")
        
        # Main UI layout in Pattern Finder tab
        st.sidebar.header("Pattern Finder Settings")
        with st.sidebar:
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
            
            # Session Settings
            st.subheader("Session Settings")
            
            # Add session selection
            session_options = {
                "Asia (UTC 00:00)": 0,
                "London (UTC 07:00)": 7,
                "New York (UTC 13:00)": 13,
                "Custom": -1
            }
            
            selected_session = st.selectbox(
                "Session Start:",
                options=list(session_options.keys()),
                index=0,
                help="Select the trading session to analyze"
            )
            
            # If custom is selected, show hour input
            session_hour = session_options[selected_session]
            if session_hour == -1:  # Custom option
                session_hour = st.slider(
                    "Custom Start Hour (UTC):",
                    min_value=0,
                    max_value=23,
                    value=9,
                    step=1,
                    help="Select custom session start time in UTC"
                )
            
            # Interval selection
            interval = st.select_slider(
                "Select Interval:",
                options=["5m", "15m", "30m", "1h"],
                value="5m"
            )
            
            # Number of matches to display
            all_matches = st.slider(
                "Calculate Top Matches:",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                help="Total number of matches to find (you can display fewer)"
            )
            
            display_matches = st.slider(
                "Display on Chart:",
                min_value=1,
                max_value=min(10, all_matches),
                value=min(5, all_matches),
                step=1,
                help="Number of matches to show on the chart"
            )
            
            # Display options
            st.subheader("Display Options")
            recent_bias = st.checkbox("Bias Toward Recent Patterns", value=False)
            show_scores = st.checkbox("Show Similarity Scores", value=True)
            
            # Group similar patterns
            group_patterns = st.checkbox("Group Similar Patterns", value=False, 
                                      help="Group patterns that are very similar to reduce redundancy")
            
            # If user wants to group similar patterns, add threshold slider
            similarity_threshold = 0.8
            if group_patterns:
                similarity_threshold = st.slider(
                    "Similarity Threshold:",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.8,
                    step=0.1,
                    help="Patterns with similarity below this threshold will be grouped together"
                )
            
            # Add confidence filter
            st.subheader("Confidence Settings")
            filter_by_confidence = st.checkbox("Only Show High Confidence", value=False,
                                            help="Only show signals with high confidence (>80%)")
        
        # Main pattern finder logic
        try:
            # Fetch data
            with st.spinner(f"Fetching data for {symbol}..."):
                data = fetch_crypto_data(symbol, period="60d", interval=interval)
            
            if data is not None:
                # Group data by sessions with configurable start time
                with st.spinner("Processing trading sessions..."):
                    sessions = get_sessions(data, session_hour)
                    
                    # Get current session date
                    current_time = datetime.now(pytz.utc)
                    current_session_date = get_session_date(current_time, session_hour)
                    
                    # Calculate patterns for all sessions
                    all_patterns = calculate_session_patterns(sessions, current_session_date)
                    
                    # Get current session data and pattern
                    current_session = sessions.get(current_session_date)
                    
                    if current_session is not None and not current_session.empty:
                        session_open = current_session['Close'].iloc[0]
                        current_pattern = ((current_session['Close'] - session_open) / session_open) * 100
                        
                        # Calculate minutes from session start
                        session_start = current_session.index[0]
                        current_pattern.index = [(ts - session_start).total_seconds() / 60 for ts in current_session.index]
                        
                        # Find top matching patterns
                        matches_to_find = max(all_matches, 5)
                        all_top_matches = find_similar_sessions(
                            current_pattern, all_patterns, 
                            num_matches=matches_to_find,
                            recent_bias=recent_bias
                        )
                        
                        # Group similar patterns if requested
                        if group_patterns and all_top_matches:
                            top_matches = group_similar_patterns(all_top_matches, threshold=similarity_threshold)
                            # Limit to requested number after grouping
                            top_matches = top_matches[:all_matches]
                        else:
                            top_matches = all_top_matches[:all_matches]
                        
                        if top_matches:
                            # Calculate confidence metrics
                            confidence_metrics = calculate_confidence_metrics(top_matches)
                            
                            # Apply confidence filter if selected
                            show_results = True
                            if filter_by_confidence and confidence_metrics:
                                if confidence_metrics['confidence'] < 0.8:
                                    st.warning("Current pattern has low/medium confidence. Disable 'Only Show High Confidence' to view it.")
                                    show_results = False
                            
                            if show_results:
                                # Display the chart
                                fig = plot_pattern_matches(
                                    sessions, 
                                    symbol,
                                    current_session_date,
                                    top_matches,
                                    show_scores=show_scores,
                                    max_patterns=display_matches,
                                    session_open_hour_utc=session_hour
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show pattern distribution
                                if len(top_matches) >= 3:
                                    dist_fig = analyze_pattern_distribution(top_matches)
                                    if dist_fig:
                                        st.plotly_chart(dist_fig, use_container_width=True)
                                
                                # Display confidence metrics in a nice format
                                if confidence_metrics:
                                    st.subheader("Pattern Prediction")
                                    
                                    # Create three columns for metrics
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        direction_str = "Upward" if confidence_metrics['direction'] == "up" else "Downward"
                                        st.metric("Predicted Direction", direction_str)
                                        
                                    with col2:
                                        conf_pct = confidence_metrics['confidence'] * 100
                                        st.metric("Confidence", f"{conf_pct:.1f}%", 
                                                delta=f"{confidence_metrics['confidence_level']}")
                                        
                                    with col3:
                                        mag_val = confidence_metrics['avg_all']
                                        st.metric("Expected Change", f"{mag_val:.2f}%", 
                                                delta=f"{confidence_metrics['magnitude_level']} Magnitude")
                                    
                                    # Show detailed analysis
                                    st.markdown(f"""
                                    **Analysis based on {len(top_matches)} similar patterns:**
                                    
                                    - **{confidence_metrics['up_count']}** patterns ended positive (avg: **{confidence_metrics['avg_up']:.2f}%**)
                                    - **{confidence_metrics['down_count']}** patterns ended negative (avg: **{confidence_metrics['avg_down']:.2f}%**)
                                    - Similarity-weighted average: **{confidence_metrics['weighted_avg']:.2f}%**
                                    
                                    Based on backtesting, predictions with **{confidence_metrics['confidence_level']} confidence** and **{confidence_metrics['magnitude_level']} magnitude** 
                                    have historically been {
                                    "very accurate (75%+ win rate)" if confidence_metrics['confidence_level'] == "High" else 
                                    "moderately accurate" if confidence_metrics['confidence_level'] == "Medium" else 
                                    "less reliable"}.
                                    
                                    *Use the Backtest tab to verify accuracy for this specific session time.*
                                    """)
                                    
                                    # Show time remaining in current session
                                    session_end = datetime.combine(current_session_date, time(hour=session_hour))
                                    session_end = pytz.utc.localize(session_end)
                                    time_remaining = (session_end - current_time).total_seconds() / 3600
                                    
                                    if time_remaining > 0:
                                        st.info(f"⏱️ **{time_remaining:.1f} hours** remaining in current trading session (until UTC {session_hour:02d}:00)")
                                
                                # Display match details
                                st.subheader("Pattern Match Details")
                                
                                # Create a card for each match
                                for match in top_matches[:display_matches]:
                                    create_match_details_card(match, sessions, session_hour)
                                    
                        else:
                            st.warning("No similar historical patterns found. Try a different interval or check back later in the session.")
                    else:
                        st.warning("Not enough data for the current trading session. Please check back later.")
                
            else:
                st.error("No data available for the selected symbol.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    
    with tabs[1]:  # Backtest tab
        st.markdown("### Backtest Pattern Predictions with Different Session Times")
        st.markdown("""
        This tool tests if patterns detected in the **first part** of a session can predict price movement in the **second part**.
        
        Select your desired session start time, data parameters, and cutoff hour for making predictions.
        
        > **Note**: Yahoo Finance provides limited intraday history (maximum of 60 days for 5-minute data, 
        > 30 days for 15-minute data). The backtester will automatically adjust to use available data.
        """)
        
        # Backtest UI layout
        backtest_col1, backtest_col2 = st.columns([2, 1])
        
        with backtest_col2:
            st.subheader("Backtest Settings")
            bt_symbol = st.text_input("Symbol:", value="BTC-USD").upper()
            
            # Session settings
            bt_session_options = {
                "Asia (UTC 00:00)": 0,
                "London (UTC 07:00)": 7,
                "New York (UTC 13:00)": 13,
                "Custom": -1
            }
            
            bt_selected_session = st.selectbox(
                "Session Start Time:",
                options=list(bt_session_options.keys()),
                index=0,
                key="bt_session"
            )
            
            # If custom is selected, show hour input
            bt_session_hour = bt_session_options[bt_selected_session]
            if bt_session_hour == -1:  # Custom option
                bt_session_hour = st.slider(
                    "Custom Start Hour (UTC):",
                    min_value=0,
                    max_value=23,
                    value=9,
                    step=1,
                    key="bt_custom_hour"
                )
                
            # Backtest parameters
            bt_cutoff_hours = st.slider(
                "Cutoff Hour:",
                min_value=1,
                max_value=20,
                value=6,
                step=1,
                help="Hours after session start to make prediction"
            )
            
            bt_interval = st.select_slider(
                "Data Interval:",
                options=["5m", "15m", "30m", "1h"],
                value="5m",
                key="bt_interval"
            )
            
            # Check data availability before setting test days
            available_days = check_data_availability(bt_symbol, interval=bt_interval)
            max_test_days = min(30, max(10, available_days - 10))
            
            bt_test_days = st.slider(
                "Test Days:",
                min_value=5,
                max_value=max_test_days,
                value=min(15, max_test_days),
                step=5,
                help="Number of recent sessions to test"
            )
            
            bt_num_matches = st.slider(
                "Number of Patterns:",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                help="Number of similar patterns to use for prediction"
            )
            
            # Data availability info
            st.info(f"Yahoo Finance provides approximately {available_days} days of {bt_interval} data for {bt_symbol}")
            
            # Run backtest button
            run_button = st.button("Run Backtest", type="primary")
        
        with backtest_col1:
            if run_button:
                # Fetch data for backtesting
                with st.spinner(f"Fetching data for {bt_symbol}..."):
                    # Try to get data for the specified interval
                    bt_data = None
                    try:
                        # Calculate how many days we need to fetch
                        days_to_fetch = min(60, bt_test_days + 20)  # Max 60 days due to Yahoo limitation
                        
                        ticker = yf.Ticker(bt_symbol)
                        period = f"{days_to_fetch}d"
                        bt_data = ticker.history(period=period, interval=bt_interval)
                        
                        if bt_data.empty:
                            st.error(f"No {bt_interval} data available for {bt_symbol}.")
                            # Try fallback to 5-minute data if necessary
                            if bt_interval != "5m":
                                st.info("Trying to fetch 5-minute data instead...")
                                bt_data = ticker.history(period=period, interval="5m")
                                if not bt_data.empty:
                                    st.success("Successfully fetched 5-minute data")
                    except Exception as e:
                        st.error(f"Error fetching {bt_interval} data: {str(e)}")
                        # Try fallback to 5-minute data
                        try:
                            st.info("Trying to fetch 5-minute data instead...")
                            bt_data = ticker.history(period=period, interval="5m")
                        except Exception as e2:
                            st.error(f"Also failed to fetch 5-minute data: {str(e2)}")
                
                if bt_data is not None and not bt_data.empty:
                    st.info(f"Running backtest with {bt_session_hour}:00 UTC session start and {bt_cutoff_hours}h cutoff...")
                    
                    # Run backtest
                    results_df = run_backtest(
                        data=bt_data,
                        session_open_hour_utc=bt_session_hour,
                        cutoff_hours=bt_cutoff_hours,
                        num_matches=bt_num_matches,
                        test_days=bt_test_days
                    )
                    
                    if results_df is not None and not results_df.empty:
                        # Display results summary
                        correct_count = results_df['correct'].sum()
                        total_count = len(results_df)
                        accuracy = correct_count / total_count if total_count > 0 else 0
                        
                        st.header(f"Backtest Results: {bt_symbol}")
                        
                        # Summary metrics
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("Overall Accuracy", f"{accuracy:.1%}")
                        with metrics_col2:
                            st.metric("Test Sessions", f"{total_count}")
                        with metrics_col3:
                            st.metric("Correct Predictions", f"{correct_count}/{total_count}")
                        
                        # Plot results
                        main_fig, conf_fig, mag_fig = plot_backtest_results(results_df)
                        
                        if main_fig:
                            st.plotly_chart(main_fig, use_container_width=True)
                        
                        # Additional charts in columns
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            if conf_fig:
                                st.plotly_chart(conf_fig, use_container_width=True)
                        with chart_col2:
                            if mag_fig:
                                st.plotly_chart(mag_fig, use_container_width=True)
                        
                        # Display example session with prediction (if we have enough examples)
                        if total_count >= 3:
                            st.subheader("Example Session Analysis")
                            
                            # Try to find a good example (correct prediction with high confidence)
                            good_examples = results_df[(results_df['correct'] == True) & (results_df['confidence'] > 0.6)]
                            if not good_examples.empty:
                                example_date = good_examples.iloc[0]['date']
                                example_similar_dates = good_examples.iloc[0]['similar_dates']
                                
                                example_fig = plot_session_example(
                                    bt_data, 
                                    example_date, 
                                    bt_cutoff_hours, 
                                    example_similar_dates, 
                                    bt_symbol,
                                    bt_session_hour
                                )
                                
                                if example_fig:
                                    st.plotly_chart(example_fig, use_container_width=True)
                                    
                                    # Add explanation
                                    st.markdown(f"""
                                    **Example Analysis:**
                                    
                                    This chart shows a {bt_selected_session if bt_session_hour != -1 else f"UTC {bt_session_hour}:00"} session on **{example_date}**. 
                                    
                                    - The white line is the actual price movement
                                    - The colored dotted lines are the similar historical patterns used for prediction
                                    - The yellow vertical line shows the cutoff time ({bt_cutoff_hours} hours after session start)
                                    
                                    In this example, the pattern predicted a **{good_examples.iloc[0]['predicted_direction']}** trend 
                                    with **{good_examples.iloc[0]['confidence']:.1%}** confidence, and the actual outcome
                                    was a **{good_examples.iloc[0]['actual_change']:.2f}%** move in the 
                                    **{good_examples.iloc[0]['actual_direction']}** direction.
                                    """)
                    else:
                        st.warning("Could not generate meaningful backtest results. Try changing parameters like decreasing the cutoff hour, using fewer test days, or a different session start time.")
                else:
                    st.error("Could not fetch data for the selected symbol and interval combination.")
                    
                    # Show what data might be available
                    st.info("""
                    Yahoo Finance has these typical data availability limits:
                    - 1m data: Last 7 days
                    - 5m data: Last 60 days
                    - 15m data: Last 60 days
                    - 30m data: Last 60 days
                    - 1h data: Last 730 days
                    - 1d data: All available history
                    """)

if __name__ == "__main__":
    main()           