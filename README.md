# Industry & Sector Seasonality Analysis

A Streamlit web application that analyzes and visualizes seasonal patterns in stock market sectors and industries. The app shows historical performance patterns on a week-by-week basis, highlighting strong seasonal tendencies.

## Features
- Analysis by sector or industry
- Adjustable time period (1-max years)
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

