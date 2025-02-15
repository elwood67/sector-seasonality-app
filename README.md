# Industry & Sector Seasonality Analysis

A Streamlit web application that analyzes and visualizes seasonal patterns in stock market sectors and industries. The app shows historical performance patterns on a week-by-week basis, highlighting strong seasonal tendencies.

## Features
- Analysis by sector or industry
- Adjustable time period (1-25 years)
- Weekly seasonality visualization
- Pattern strength indicators
- Quarterly breakdown
- Current week marker

## Setup
1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Place your stock_sectors.csv file in the same directory

3. Run the Streamlit app:
```bash
streamlit run sector_seasonality.py
```

## Data Requirements
The app expects a `stock_sectors.csv` file with the following columns:
- symbol
- sector
- industry

## Max history vs. 25 Years
For seasonality analysis, the choice between 25 years and max history involves some interesting tradeoffs:
Advantages of Max History:

More data points means more statistical significance
Can identify very long-term patterns
Particularly useful for well-established industries

Reasons to Stick with 25 Years:

Market structures and technology have changed dramatically
Many modern semiconductor companies didn't exist 30+ years ago
Trading patterns from the 1970s/80s might not be relevant today
The industry itself has evolved (e.g., semiconductors are much more critical to the economy now)

For semiconductors specifically, I'd actually lean toward keeping it at 25 years because:

The industry has changed so fundamentally since the 1990s
Modern trading patterns (post-internet) are quite different
Many current major players (like NVIDIA) weren't significant before 2000