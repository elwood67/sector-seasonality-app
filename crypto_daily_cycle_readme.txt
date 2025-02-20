
I've developed a tool that analyzes intraday price patterns across multiple timeframes to identify recurring market behavior. Here's what it's doing under the hood:

Pattern Analysis:

Takes historical price data (currently up to 60 days) in custom intervals (5m to 1h)
Normalizes each day's price movement as percentage change from the open
Calculates and overlays patterns from 5 different timeframes (5, 10, 20, 30, and 60-day lookbacks)
Generates a composite pattern (thick white line) that averages all timeframes to identify high-probability zones

Volume Analysis:

Plots normalized volume distribution throughout the trading day
Shows historical average volume (gray bars) vs. today's volume (red bars)
Helps identify whether pattern moves have volume confirmation

Key Features:

Real-time comparison of current day's price action (white dashed line) against historical patterns
Interactive toggles to isolate specific timeframe patterns
24-hour coverage for crypto markets with proper EST time alignment
Full grid system for precise time/price readings

The primary value comes from seeing how different timeframe patterns converge or diverge. When multiple timeframes show similar patterns at specific times (visible when lines cluster), it can indicate stronger support/resistance zones or higher probability directional moves.
You can customize:

Trading pairs (default setup for major cryptos)
Time intervals (5m, 15m, 30m, 1h)
Pattern visibility (toggle individual timeframes)
Analysis period (up to 60 days of lookback)

The tool is particularly useful for:

Identifying high-probability entry/exit times
Understanding when volume typically spikes
Spotting divergences between current price action and historical patterns
Finding times of day when patterns are most reliable