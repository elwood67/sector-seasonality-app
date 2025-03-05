import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np

# Set page to wide mode
st.set_page_config(layout="wide")

# Initialize all session state variables at the start
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None
if 'seasonal_pattern' not in st.session_state:
    st.session_state.seasonal_pattern = None

# Helper functions for older pandas versions
def get_iso_week_from_index(index):
    """
    Extract ISO week numbers from DatetimeIndex for older pandas versions.
    Handles cases where items might be strings or other non-datetime objects.
    """
    iso_weeks = []
    for date in index:
        try:
            # Convert to Python datetime if it's a pandas Timestamp or already a datetime
            if hasattr(date, 'isocalendar'):
                iso_weeks.append(date.isocalendar()[1])
            # Try to parse as string if it's not a datetime
            elif isinstance(date, str):
                dt_obj = pd.to_datetime(date)
                iso_weeks.append(dt_obj.isocalendar()[1])
            else:
                # If we can't determine the week, use a placeholder
                iso_weeks.append(None)
        except:
            # In case of any error, use a placeholder
            iso_weeks.append(None)
    return iso_weeks

def get_year_from_index(index):
    """
    Extract years from DatetimeIndex for older pandas versions.
    Handles cases where items might be strings or other non-datetime objects.
    """
    years = []
    for date in index:
        try:
            # Check if it's a pandas Timestamp object (which has year attribute)
            if hasattr(date, 'year'):
                years.append(date.year)
            # Try to parse as string if it's not a datetime
            elif isinstance(date, str):
                dt_obj = pd.to_datetime(date)
                years.append(dt_obj.year)
            else:
                # If we can't determine the year, use a placeholder
                years.append(None)
        except:
            # In case of any error, use a placeholder
            years.append(None)
    return years

def get_quarter_from_index(index):
    """
    Extract quarters from DatetimeIndex for older pandas versions.
    """
    quarters = []
    for date in index:
        try:
            if hasattr(date, 'month'):
                month = date.month
                quarters.append((month - 1) // 3 + 1)  # Convert month to quarter (1-4)
            elif isinstance(date, str):
                dt_obj = pd.to_datetime(date)
                month = dt_obj.month
                quarters.append((month - 1) // 3 + 1)
            else:
                quarters.append(None)
        except:
            quarters.append(None)
    return quarters

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
    """Get current ISO week number"""
    today = datetime.now()
    return today.isocalendar()[1]

def handle_symbol_change(new_symbol):
    st.session_state.symbol = new_symbol

def create_iso_calendar_view(year):
    """
    Create a DataFrame showing the ISO calendar for a year.
    """
    # Start with first day of the year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    calendar_data = []
    current_date = start_date
    
    while current_date <= end_date:
        iso_year, iso_week, iso_day = current_date.isocalendar()
        is_monday = iso_day == 1
        
        calendar_data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'day': current_date.strftime('%a'),
            'iso_week': iso_week,
            'is_week_start': is_monday
        })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(calendar_data)

def calculate_directional_consistency(data, num_years):
    """
    Calculate the percentage of years where price moves in the same direction for each ISO week.
    """
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(years=num_years)
    filtered_data = data[data.index >= start_date].copy()
    
    # Resample to weekly data (week ending on Sunday)
    weekly_data = filtered_data.resample('W-SUN').last()
    
    # Calculate weekly returns
    weekly_returns = weekly_data.pct_change().values  # Convert to numpy array
    
    # Add ISO week numbers using helper function for older pandas versions
    iso_weeks = get_iso_week_from_index(weekly_data.index)
    
    # Get years using helper function for older pandas versions
    years = get_year_from_index(weekly_data.index)
    
    # Check if all have the same length - if not, truncate to shortest
    min_length = min(len(weekly_returns), len(iso_weeks), len(years))
    
    # Create numpy arrays of same length
    years_array = np.array(years[:min_length])
    iso_weeks_array = np.array(iso_weeks[:min_length])
    returns_array = np.array(weekly_returns[:min_length])
    
    # Dictionary to store results
    consistency_dict = {}
    
    # Get unique weeks
    unique_weeks = set(w for w in iso_weeks_array if w is not None)
    
    # Calculate directional consistency for each ISO week
    for week in unique_weeks:
        # Find indices for this week
        week_indices = [i for i, w in enumerate(iso_weeks_array) if w == week]
        
        if week_indices:
            week_returns = returns_array[week_indices]
            
            # Count positive and negative moves
            positive_moves = sum(1 for r in week_returns if r > 0 and not np.isnan(r))
            negative_moves = sum(1 for r in week_returns if r < 0 and not np.isnan(r))
            total_moves = sum(1 for r in week_returns if not np.isnan(r))
            
            if total_moves > 0:
                max_consistent = max(positive_moves, negative_moves)
                consistency = (max_consistent / total_moves) * 100
                direction = "Bullish" if positive_moves >= negative_moves else "Bearish"
                consistency_dict[week] = {
                    'consistency': consistency,
                    'direction': direction,
                    'positive_count': positive_moves,
                    'negative_count': negative_moves,
                    'total_moves': total_moves
                }
            else:
                # No data for this week
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
    
    # Prepare data for chart
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(years=num_years)
    filtered_data = data[data.index >= start_date]
    
    # Resample to weekly data (week ending on Sunday)
    weekly_data = filtered_data.resample('W-SUN').last()
    
    # Get ISO weeks
    iso_weeks = get_iso_week_from_index(weekly_data.index)
    
    # Get years
    years = get_year_from_index(weekly_data.index)
    
    # Create a dictionary to store normalized data by year and week
    normalized_data = {}
    
    # Group by year
    unique_years = set(years)
    for year in unique_years:
        year_indices = [i for i, y in enumerate(years) if y == year]
        year_data = [weekly_data.iloc[i] for i in year_indices]
        year_weeks = [iso_weeks[i] for i in year_indices]
        
        # Create week to price dictionary for this year
        week_prices = {}
        for i, week in enumerate(year_weeks):
            if week is not None:
                week_prices[week] = year_data[i]
        
        # Normalize for this year
        if week_prices:
            # Get min and max prices
            prices = list(week_prices.values())
            min_price = min(prices)
            max_price = max(prices)
            price_range = max_price - min_price
            
            if price_range > 0:  # Avoid division by zero
                # Normalize each price to 0-100 scale
                for week, price in week_prices.items():
                    normalized_price = ((price - min_price) / price_range) * 100
                    if week not in normalized_data:
                        normalized_data[week] = []
                    normalized_data[week].append(normalized_price)
    
    # Calculate average normalized value for each week
    seasonal_pattern = {}
    for week, values in normalized_data.items():
        seasonal_pattern[week] = sum(values) / len(values)
    
    # Create x and y values for our chart (all weeks 1-53)
    all_weeks = list(range(1, 54))
    all_pattern_values = []
    
    for week in all_weeks:
        if week in seasonal_pattern:
            all_pattern_values.append(seasonal_pattern[week])
        else:
            # Use None for missing weeks
            all_pattern_values.append(None)
    
    # Fill in missing values with interpolation
    # Replace None with NaN for interpolation
    all_pattern_np = np.array(all_pattern_values, dtype=float)
    
    # Find indices of NaN values
    nan_indices = np.isnan(all_pattern_np)
    
    # Perform simple linear interpolation if any NaNs
    if np.any(nan_indices):
        valid_indices = ~nan_indices
        x_valid = np.where(valid_indices)[0]
        y_valid = all_pattern_np[valid_indices]
        
        # Interpolate missing values
        if len(x_valid) > 1:  # Need at least 2 points for interpolation
            x_nan = np.where(nan_indices)[0]
            all_pattern_np[nan_indices] = np.interp(x_nan, x_valid, y_valid)
    
    # Create the figure
    fig = go.Figure()
    
    # Create a single connected line with colored segments between points
    x_values = all_weeks
    y_values = all_pattern_np
    
    # Add the main marker trace (nearly invisible)
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
    for i in range(len(x_values) - 1):
        week = x_values[i]
        next_week = x_values[i+1]
        
        # Get consistency information for this week
        if week in directional_consistency:
            consistency = directional_consistency[week]['consistency']
            direction = directional_consistency[week]['direction']
        else:
            consistency = 0
            direction = 'Unknown'
        
        # Determine color based on consistency
        if consistency == 100:
            color = 'red'
            line_width = 4
            group = '100% Consistent'
        elif consistency >= 90:
            color = 'purple'
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
            color = 'blue'
            line_width = 2
            group = '60-69% Consistent'
        else:
            color = 'white'
            line_width = 1.5
            group = '<60% Consistent'
        
        # Create hover text
        hover_text = f"Week {week}<br>Pattern Value: {y_values[i]:.1f}<br>Direction: {direction}<br>Consistency: {consistency:.1f}%"
        
        # Add detailed info for weeks with consistency data
        if week in directional_consistency:
            info = directional_consistency[week]
            hover_text += f"<br>Bullish Years: {info['positive_count']}<br>Bearish Years: {info['negative_count']}<br>Total Years: {info['total_moves']}"
        
        # Add the line segment
        fig.add_trace(go.Scatter(
            x=[week, next_week],
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
        {'name': '100% Consistent', 'color': 'red', 'width': 4},
        {'name': '90-99% Consistent', 'color': 'purple', 'width': 3.5},
        {'name': '80-89% Consistent', 'color': 'yellow', 'width': 3},
        {'name': '70-79% Consistent', 'color': 'green', 'width': 2.5},
        {'name': '60-69% Consistent', 'color': 'blue', 'width': 2},
        {'name': '<60% Consistent', 'color': 'white', 'width': 1.5}
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
    
    # Filter data for current year
    current_year_indices = [i for i, y in enumerate(get_year_from_index(data.index)) if y == current_year]
    if current_year_indices:
        current_year_data = pd.Series([data.iloc[i] for i in current_year_indices], 
                                      index=[data.index[i] for i in current_year_indices])
        
        # Resample to weekly
        current_year_weekly = current_year_data.resample('W-SUN').last()
        current_weeks = get_iso_week_from_index(current_year_weekly.index)
        
        # Get min and max for normalization
        min_val = min(current_year_weekly)
        max_val = max(current_year_weekly)
        value_range = max_val - min_val
        
        if value_range > 0:  # Avoid division by zero
            # Normalize values to 0-100
            normalized_values = [(val - min_val) / value_range * 100 for val in current_year_weekly]
            
            # Create week to normalized value mapping
            week_values = {}
            for i, week in enumerate(current_weeks):
                if week is not None:
                    week_values[week] = normalized_values[i]
            
            # Extract x and y for plotting
            curr_x = list(week_values.keys())
            curr_y = list(week_values.values())
            
            # Add to chart
            fig.add_trace(go.Scatter(
                x=curr_x,
                y=curr_y,
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
        (40, 53, 'Q4', 'rgba(176, 224, 230, 0.3)')
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
    valid_y = [y for y in y_values if y is not None and not np.isnan(y)]
    if valid_y:
        seasonal_min = min(valid_y)
        seasonal_max = max(valid_y)
        seasonal_range = seasonal_max - seasonal_min
        
        y_min = max(0, seasonal_min - (seasonal_range * 0.4))
        y_max = seasonal_max + (seasonal_range * 0.4)
    else:
        # Default range if no valid data
        y_min = 0
        y_max = 100
    
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
            ticktext=[f'W{i}' for i in range(1, 54)],  # Shortened to just W1, W2, etc.
            tickvals=list(range(1, 54)),
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
        margin=dict(l=50, r=50, t=120, b=80)  # Increased top margin for legend
    )
    
    # Update sidebar stats
    sidebar_stats = {
        '100% Consistent': [w for w, c in directional_consistency.items() if c['consistency'] == 100],
        '90-99% Consistent': [w for w, c in directional_consistency.items() if 90 <= c['consistency'] < 100],
        '80-89% Consistent': [w for w, c in directional_consistency.items() if 80 <= c['consistency'] < 90]
    }
    
    return fig, seasonal_pattern, directional_consistency, sidebar_stats

def backtest_seasonality_strategy(data, directional_consistency, test_year, threshold=60):
    """
    Backtest a seasonality-based trading strategy for a specific year using ISO weeks.
    """
    # Filter data for the test year - using our helper function
    years = get_year_from_index(data.index)
    test_indices = [i for i, y in enumerate(years) if y == test_year]
    
    if not test_indices:
        return f"No data available for {test_year}"
    
    # Create test data DataFrame
    test_data = pd.DataFrame({
        'Close': [data.iloc[i] for i in test_indices]
    }, index=[data.index[i] for i in test_indices])
    
    # Resample to weekly data (using Sunday as week end)
    test_data = test_data.resample('W-SUN').last()
    
    # Add ISO week numbers using helper function
    test_data['iso_week'] = get_iso_week_from_index(test_data.index)
    
    # Add returns
    test_data['return'] = test_data['Close'].pct_change()
    
    # Add signal based on seasonality
    signals = []
    for week in test_data['iso_week']:
        if week in directional_consistency:
            consistency = directional_consistency[week]['consistency']
            direction = directional_consistency[week]['direction']
            if consistency >= threshold:
                signals.append(1 if direction == 'Bullish' else -1)
            else:
                signals.append(0)
        else:
            signals.append(0)
    
    test_data['signal'] = signals
    
    # Calculate expected direction based on signal
    test_data['expected_direction'] = ['up' if s == 1 else ('down' if s == -1 else 'neutral') for s in test_data['signal']]
    
    # Calculate actual direction based on next week's returns
    # This simulates entering at Sunday close/Monday open and exiting next Sunday close
    test_data['next_week_return'] = test_data['return'].shift(-1)
    
    actual_directions = []
    for ret in test_data['next_week_return']:
        if pd.isna(ret):
            actual_directions.append('neutral')
        else:
            actual_directions.append('up' if ret > 0 else 'down')
    
    test_data['actual_direction'] = actual_directions
    
    # Determine if the trade was correct (signal matched actual direction)
    is_correct = []
    for i in range(len(test_data)):
        exp_dir = test_data['expected_direction'].iloc[i]
        act_dir = test_data['actual_direction'].iloc[i]
        is_correct.append((exp_dir == 'up' and act_dir == 'up') or (exp_dir == 'down' and act_dir == 'down'))
    
    test_data['is_correct'] = is_correct
    
    # Only count trades where we had a signal
    test_data['trade_taken'] = test_data['signal'] != 0
    
    # Calculate strategy returns:
    # Use next week's return to simulate entering at this week's close
    # and exiting at next week's close
    strategy_returns = []
    for i in range(len(test_data)):
        if pd.isna(test_data['next_week_return'].iloc[i]) or pd.isna(test_data['signal'].iloc[i]):
            strategy_returns.append(0)
        else:
            strategy_returns.append(test_data['signal'].iloc[i] * test_data['next_week_return'].iloc[i])
    
    test_data['strategy_return'] = strategy_returns
    
    # Calculate cumulative returns
    cum_return = 0
    cum_returns = []
    for ret in test_data['return']:
        if not pd.isna(ret):
            cum_return += ret
        cum_returns.append(cum_return)
    
    test_data['cumulative_return'] = cum_returns
    
    # For strategy returns, calculate cumulative returns
    cum_strategy = 0
    cum_strategies = []
    for ret in test_data['strategy_return']:
        if not pd.isna(ret):
            cum_strategy += ret
        cum_strategies.append(cum_strategy)
    
    test_data['cumulative_strategy'] = cum_strategies
    
    # Calculate trade statistics (exclude the last week since we don't have next week's return)
    trades = test_data[test_data['trade_taken']].iloc[:-1].copy() if len(test_data) > 1 else test_data[test_data['trade_taken']]
    winning_trades = trades[trades['is_correct'] == True]
    
    stats = {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'win_rate': len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0,
        'total_return': test_data['cumulative_strategy'].iloc[-1] * 100 if len(test_data) > 0 else 0,
        'buy_hold_return': test_data['cumulative_return'].iloc[-1] * 100 if len(test_data) > 0 else 0,
        'long_trades': len(trades[trades['signal'] == 1]),
        'short_trades': len(trades[trades['signal'] == -1]),
        'avg_trade_return': trades['strategy_return'].mean() * 100 if len(trades) > 0 else 0
    }
    
    return test_data, stats

def display_backtest_results(test_data, stats, test_year, symbol, threshold):
    """
    Display backtest results in Streamlit.
    """
    st.subheader(f"Seasonality Strategy Backtest for {symbol} ({test_year})")
    st.write(f"Trading on weeks with directional consistency >= {threshold}%")
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
        st.metric("Total Trades", stats['total_trades'])
    
    with col2:
        st.metric("Strategy Return", f"{stats['total_return']:.1f}%")
        st.metric("Buy & Hold Return", f"{stats['buy_hold_return']:.1f}%")
    
    with col3:
        st.metric("Long/Short Ratio", f"{stats['long_trades']}/{stats['short_trades']}")
        st.metric("Avg Trade Return", f"{stats['avg_trade_return']:.2f}%")
    
    # Plot equity curves
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=test_data['cumulative_strategy'] * 100,
        mode='lines',
        name='Seasonality Strategy',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=test_data['cumulative_return'] * 100,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='gray', width=2)
    ))
    
    # Add markers for trades
    long_entries = test_data[test_data['signal'] == 1]
    short_entries = test_data[test_data['signal'] == -1]
    
    # Add markers for correct and incorrect predictions
    correct_trades = test_data[(test_data['trade_taken'] == True) & (test_data['is_correct'] == True)]
    incorrect_trades = test_data[(test_data['trade_taken'] == True) & (test_data['is_correct'] == False)]
    
    fig.add_trace(go.Scatter(
        x=long_entries.index,
        y=long_entries['cumulative_strategy'] * 100,
        mode='markers',
        name='Long Signal',
        marker=dict(color='blue', size=8, symbol='triangle-up')
    ))
    
    fig.add_trace(go.Scatter(
        x=short_entries.index,
        y=short_entries['cumulative_strategy'] * 100,
        mode='markers',
        name='Short Signal',
        marker=dict(color='red', size=8, symbol='triangle-down')
    ))
    
    fig.add_trace(go.Scatter(
        x=correct_trades.index,
        y=correct_trades['cumulative_strategy'] * 100,
        mode='markers',
        name='Correct Prediction',
        marker=dict(color='green', size=12, symbol='circle-open')
    ))
    
    fig.add_trace(go.Scatter(
        x=incorrect_trades.index,
        y=incorrect_trades['cumulative_strategy'] * 100,
        mode='markers',
        name='Incorrect Prediction',
        marker=dict(color='red', size=12, symbol='x-open')
    ))
    
    fig.update_layout(
        title=f"{symbol} Seasonality Strategy vs Buy & Hold ({test_year})",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show quarterly returns
    quarters = get_quarter_from_index(test_data.index)
    test_data['quarter'] = quarters
    
    quarterly_returns = {}
    for q in range(1, 5):  # Quarters 1-4
        q_data = test_data[test_data['quarter'] == q]
        if not q_data.empty:
            quarterly_returns[q] = q_data['strategy_return'].sum() * 100
    
    # Create quarterly returns figure
    fig2 = go.Figure()
    
    q_x = list(quarterly_returns.keys())
    q_y = list(quarterly_returns.values())
    
    fig2.add_trace(go.Bar(
        x=[f"Q{q}" for q in q_x],
        y=q_y,
        marker_color=['green' if y > 0 else 'red' for y in q_y]
    ))
    
    fig2.update_layout(
        title=f"Quarterly Returns - {test_year}",
        xaxis_title="Quarter",
        yaxis_title="Return (%)",
        template="plotly_dark",
        height=300
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Show debug information about ISO weeks
    with st.expander("Week Number Debug Information"):
        st.write("This shows how ISO week numbers map to calendar dates:")
        debug_calendar = create_iso_calendar_view(test_year)
        st.dataframe(debug_calendar.head(40))  # Show first 40 days
    
    # Show the actual trades with more detailed information
    st.subheader("Trade Log")
    
    if stats['total_trades'] > 0:
        trades = test_data[test_data['trade_taken']].iloc[:-1].copy() if len(test_data) > 1 else test_data[test_data['trade_taken']]
        
        # Create a DataFrame for display
        trade_log = []
        for i in range(len(trades)):
            try:
                trade_log.append({
                    'Date': trades.index[i].strftime('%Y-%m-%d'),
                    'Week #': trades['iso_week'].iloc[i],
                    'Direction': 'Long' if trades['signal'].iloc[i] == 1 else 'Short',
                    'Weekly Return (%)': trades['next_week_return'].iloc[i] * 100 if not pd.isna(trades['next_week_return'].iloc[i]) else 0,
                    'Expected Move': trades['expected_direction'].iloc[i],
                    'Actual Move': trades['actual_direction'].iloc[i],
                    'Result': 'Win' if trades['is_correct'].iloc[i] else 'Loss'
                })
            except:
                continue  # Skip any errors in processing trades
        
        # Convert to DataFrame
        if trade_log:
            trade_table = pd.DataFrame(trade_log)
            
            # Display the dataframe with colored cells
            st.dataframe(trade_table.style.applymap(
                lambda x: 'background-color: green; color: white' if x == 'Win' else 'background-color: red; color: white',
                subset=['Result']
            ))
            
            # Add summary statistics about prediction accuracy
            st.write("### Prediction Accuracy Analysis")
            
            # Group by expected and actual directions
            direction_summary = {}
            for trade in trade_log:
                exp_dir = trade['Expected Move']
                act_dir = trade['Actual Move']
                
                if exp_dir not in direction_summary:
                    direction_summary[exp_dir] = {'up': 0, 'down': 0, 'neutral': 0}
                
                direction_summary[exp_dir][act_dir] = direction_summary[exp_dir].get(act_dir, 0) + 1
            
            # Calculate accuracy
            accuracy = {}
            for direction, outcomes in direction_summary.items():
                if direction != 'neutral':  # Skip neutral predictions
                    total = sum(outcomes.values())
                    correct = outcomes.get(direction, 0)
                    accuracy[direction] = (correct / total * 100) if total > 0 else 0
            
            # Display accuracy metrics
            col1, col2 = st.columns(2)
            with col1:
                if 'up' in accuracy:
                    st.metric("Bullish Prediction Accuracy", f"{accuracy['up']:.1f}%")
            with col2:
                if 'down' in accuracy:
                    st.metric("Bearish Prediction Accuracy", f"{accuracy['down']:.1f}%")
        else:
            st.write("No valid trades to display")
    else:
        st.write("No trades were taken during this period")

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
                
                tab1, tab2, tab3 = st.tabs(["100%", "90-99%", "80-89%"])
                
                with tab1:
                    high_weeks = sidebar_stats['100% Consistent']
                    if high_weeks:
                        week_details = []
                        for w in sorted(high_weeks):
                            direction = directional_consistency[w]['direction']
                            week_details.append(f"Week {w} ({direction})")
                        
                        # Split into columns for better display if there are many weeks
                        if len(week_details) > 10:
                            col1, col2 = st.columns(2)
                            half = len(week_details) // 2
                            col1.markdown("<br>".join(week_details[:half]), unsafe_allow_html=True)
                            col2.markdown("<br>".join(week_details[half:]), unsafe_allow_html=True)
                        else:
                            st.markdown("<br>".join(week_details), unsafe_allow_html=True)
                    else:
                        st.write("No weeks with 100% consistency")
                
                with tab2:
                    med_weeks = sidebar_stats['90-99% Consistent']
                    if med_weeks:
                        week_details = []
                        for w in sorted(med_weeks):
                            consistency = directional_consistency[w]['consistency']
                            direction = directional_consistency[w]['direction']
                            week_details.append(f"Week {w} ({consistency:.1f}%, {direction})")
                        
                        # Split into columns for better display if there are many weeks
                        if len(week_details) > 10:
                            col1, col2 = st.columns(2)
                            half = len(week_details) // 2
                            col1.markdown("<br>".join(week_details[:half]), unsafe_allow_html=True)
                            col2.markdown("<br>".join(week_details[half:]), unsafe_allow_html=True)
                        else:
                            st.markdown("<br>".join(week_details), unsafe_allow_html=True)
                    else:
                        st.write("No weeks with 90-99% consistency")
                
                with tab3:
                    low_weeks = sidebar_stats['80-89% Consistent']
                    if low_weeks:
                        week_details = []
                        for w in sorted(low_weeks):
                            consistency = directional_consistency[w]['consistency']
                            direction = directional_consistency[w]['direction']
                            week_details.append(f"Week {w} ({consistency:.1f}%, {direction})")
                        
                        # Split into columns for better display if there are many weeks
                        if len(week_details) > 15:
                            col1, col2 = st.columns(2)
                            half = len(week_details) // 2
                            col1.markdown("<br>".join(week_details[:half]), unsafe_allow_html=True)
                            col2.markdown("<br>".join(week_details[half:]), unsafe_allow_html=True)
                        else:
                            st.markdown("<br>".join(week_details), unsafe_allow_html=True)
                    else:
                        st.write("No weeks with 80-89% consistency")
            
            # Add backtest section
            st.markdown("---")
            st.header("Seasonality Strategy Backtest")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Choose year to backtest
                current_year = datetime.now().year
                test_year = st.selectbox(
                    "Select year to backtest:",
                    options=list(range(current_year-5, current_year+1)),
                    index=1  # Default to last year
                )
            
            with col2:
                # Choose consistency threshold
                threshold = st.slider(
                    "Minimum consistency threshold (%):",
                    min_value=50,
                    max_value=100,
                    value=60,
                    step=5
                )
            
            # Run backtest
            if st.button("Run Backtest"):
                with st.spinner(f"Running backtest for {st.session_state.symbol} in {test_year}..."):
                    # When backtesting a specific year, we need to exclude that year's data
                    # from the seasonal pattern calculation to avoid lookahead bias
                    backtest_data = data.copy()
                    
                    # We should calculate directional consistency based on data up to the test year
                    years = get_year_from_index(backtest_data.index)
                    historical_indices = [i for i, y in enumerate(years) if y < test_year]
                    
                    if historical_indices:
                        historical_data = pd.Series(
                            [backtest_data.iloc[i] for i in historical_indices],
                            index=[backtest_data.index[i] for i in historical_indices]
                        )
                        
                        # Calculate how many years of history we have before the test year
                        available_years = len(set(get_year_from_index(historical_data.index)))
                        
                        # If we have at least 3 years of data, proceed with backtesting
                        if available_years >= 3:
                            # Recalculate directional consistency with only data prior to test year
                            backtest_years = min(num_years, available_years)
                            historical_consistency = calculate_directional_consistency(
                                historical_data, 
                                backtest_years
                            )
                            
                            test_data, stats = backtest_seasonality_strategy(
                                backtest_data, 
                                historical_consistency, 
                                test_year,
                                threshold
                            )
                            
                            # Display results
                            if isinstance(test_data, str):
                                st.error(test_data)  # Show error message
                            else:
                                display_backtest_results(test_data, stats, test_year, st.session_state.symbol, threshold)
                        else:
                            st.error(f"Not enough historical data before {test_year} to perform backtesting. Need at least 3 years of data.")
                    else:
                        st.error(f"No historical data available before {test_year}.")
                        
        else:
            st.error(f"No data available for {st.session_state.symbol}. Please check the symbol and try again.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())  # Print full traceback for debugging
        st.error("Please try another symbol or check your internet connection.")

if __name__ == "__main__":
    main()