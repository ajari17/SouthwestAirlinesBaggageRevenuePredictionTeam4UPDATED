import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

# Load the data
unemployment_data = """date,unemplyment_rate
2015-01,5.7
2015-02,5.5
2015-03,5.4
2015-04,5.4
2015-05,5.6
2015-06,5.3
2015-07,5.2
2015-08,5.1
2015-09,5.0
2015-10,5.0
2015-11,5.1
2015-12,5.0
2016-01,4.8
2016-02,4.9
2016-03,5.0
2016-04,5.1
2016-05,4.8
2016-06,4.9
2016-07,4.8
2016-08,4.9
2016-09,5.0
2016-10,4.9
2016-11,4.7
2016-12,4.7
2017-01,4.7
2017-02,4.6
2017-03,4.4
2017-04,4.4
2017-05,4.4
2017-06,4.3
2017-07,4.3
2017-08,4.4
2017-09,4.3
2017-10,4.2
2017-11,4.2
2017-12,4.1
2018-01,4.0
2018-02,4.1
2018-03,4.0
2018-04,4.0
2018-05,3.8
2018-06,4.0
2018-07,3.8
2018-08,3.8
2018-09,3.7
2018-10,3.8
2018-11,3.8
2018-12,3.9
2019-01,4.0
2019-02,3.8
2019-03,3.8
2019-04,3.7
2019-05,3.6
2019-06,3.6
2019-07,3.7
2019-08,3.6
2019-09,3.5
2019-10,3.6
2019-11,3.6
2019-12,3.6
2021-05,5.8
2021-06,5.9
2021-07,5.4
2021-08,5.1
2021-09,4.7
2021-10,4.5
2021-11,4.2
2021-12,3.9
2022-02,3.8
2022-03,3.7
2022-04,3.7
2022-05,3.6
2022-06,3.6
2022-07,3.5
2022-08,3.6
2022-09,3.5
2022-10,3.6
2022-11,3.6
2022-12,3.5
2023-01,3.5
2023-02,3.6
2023-03,3.5
2023-04,3.4
2023-05,3.6
2023-06,3.6
2023-07,3.5
2023-08,3.7
2023-09,3.8
2023-10,3.9
2023-11,3.7
2023-12,3.8
2024-01,3.7
2024-02,3.9
2024-03,3.9
2024-04,3.9
2024-05,4.0
2024-06,4.1
2024-07,4.2
2024-08,4.2
2024-09,4.1
2024-10,4.1
2024-11,4.2
2024-12,4.1
2025-01,4.0
2025-02,4.1
2025-03,4.2
2025-04,4.2
2025-05,4.2
2025-06,4.1
2025-07,4.2
2025-08,4.3"""

fedfunds_data = """Date,fed_fund_rate
2015-01,0.11
2015-02,0.11
2015-03,0.11
2015-04,0.12
2015-05,0.12
2015-06,0.13
2015-07,0.13
2015-08,0.14
2015-09,0.14
2015-10,0.12
2015-11,0.12
2015-12,0.24
2016-01,0.34
2016-02,0.38
2016-03,0.36
2016-04,0.37
2016-05,0.37
2016-06,0.38
2016-07,0.39
2016-08,0.40
2016-09,0.40
2016-10,0.40
2016-11,0.41
2016-12,0.54
2017-01,0.65
2017-02,0.66
2017-03,0.79
2017-04,0.90
2017-05,0.91
2017-06,1.04
2017-07,1.15
2017-08,1.16
2017-09,1.15
2017-10,1.15
2017-11,1.16
2017-12,1.30
2018-01,1.41
2018-02,1.42
2018-03,1.51
2018-04,1.69
2018-05,1.70
2018-06,1.82
2018-07,1.91
2018-08,1.91
2018-09,1.95
2018-10,2.19
2018-11,2.20
2018-12,2.27
2019-01,2.40
2019-02,2.40
2019-03,2.41
2019-04,2.42
2019-05,2.39
2019-06,2.38
2019-07,2.40
2019-08,2.13
2019-09,2.04
2019-10,1.83
2019-11,1.55
2019-12,1.55
2021-05,0.06
2021-06,0.08
2021-07,0.10
2021-08,0.09
2021-09,0.08
2021-10,0.08
2021-11,0.08
2021-12,0.08
2022-02,0.08
2022-03,0.20
2022-04,0.33
2022-05,0.77
2022-06,1.21
2022-07,1.68
2022-08,2.33
2022-09,2.56
2022-10,3.08
2022-11,3.78
2022-12,4.10
2023-01,4.33
2023-02,4.57
2023-03,4.65
2023-04,4.83
2023-05,5.06
2023-06,5.08
2023-07,5.12
2023-08,5.33
2023-09,5.33
2023-10,5.33
2023-11,5.33
2023-12,5.33
2024-01,5.33
2024-02,5.33
2024-03,5.33
2024-04,5.33
2024-05,5.33
2024-06,5.33
2024-07,5.33
2024-08,5.33
2024-09,5.13
2024-10,4.83
2024-11,4.64
2024-12,4.48
2025-01,4.33
2025-02,4.33
2025-03,4.33
2025-04,4.33
2025-05,4.33
2025-06,4.33
2025-07,4.33
2025-08,4.33
2025-09,4.22
2025-10,4.09"""

from io import StringIO

# Parse data
df_unemployment = pd.read_csv(StringIO(unemployment_data))
df_unemployment['date'] = pd.to_datetime(df_unemployment['date'])
df_unemployment = df_unemployment.set_index('date')

df_fedfunds = pd.read_csv(StringIO(fedfunds_data))
df_fedfunds['Date'] = pd.to_datetime(df_fedfunds['Date'])
df_fedfunds = df_fedfunds.set_index('Date')


# Function to forecast using SARIMA
def forecast_sarima(data, column_name, periods=24, order=(2, 1, 2), seasonal_order=(1, 0, 1, 12)):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    forecast = results.forecast(steps=periods)
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                   periods=periods, freq='MS')

    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        column_name: forecast.values
    })

    return results, forecast_df


# Forecast both series
unemp_model, unemp_forecast = forecast_sarima(
    df_unemployment['unemplyment_rate'],
    'unemplyment_rate',
    periods=24
)

fedfunds_model, fedfunds_forecast = forecast_sarima(
    df_fedfunds['fed_fund_rate'],
    'fed_fund_rate',
    periods=24
)

# Create combined datasets (historical + forecast)
unemp_historical = df_unemployment.reset_index()
unemp_historical.columns = ['date', 'unemplyment_rate']
unemp_historical['type'] = 'historical'
unemp_forecast['type'] = 'forecast'
unemp_combined = pd.concat([unemp_historical, unemp_forecast], ignore_index=True)

fedfunds_historical = df_fedfunds.reset_index()
fedfunds_historical.columns = ['date', 'fed_fund_rate']
fedfunds_historical['type'] = 'historical'
fedfunds_forecast['type'] = 'forecast'
fedfunds_combined = pd.concat([fedfunds_historical, fedfunds_forecast], ignore_index=True)

# Save combined files (historical + forecast)
unemp_combined[['date', 'unemplyment_rate']].to_csv('unemployment_rate_historical_and_forecast.csv', index=False)
fedfunds_combined[['date', 'fed_fund_rate']].to_csv('fed_funds_rate_historical_and_forecast.csv', index=False)

# ============= NEW: Save prediction-only files in same format as original =============

# Format unemployment predictions to match original format (YYYY-MM)
unemp_predictions_only = unemp_forecast[['date', 'unemplyment_rate']].copy()
unemp_predictions_only['date'] = unemp_predictions_only['date'].dt.strftime('%Y-%m')
unemp_predictions_only.to_csv('unemployment_rate_predicted.csv', index=False)

# Format fed funds predictions to match original format (YYYY-MM, with capital Date column)
fedfunds_predictions_only = fedfunds_forecast[['date', 'fed_fund_rate']].copy()
fedfunds_predictions_only['date'] = fedfunds_predictions_only['date'].dt.strftime('%Y-%m')
fedfunds_predictions_only.columns = ['Date', 'fed_fund_rate']  # Match original column name
fedfunds_predictions_only.to_csv('fed_funds_rate_predicted.csv', index=False)

print("=" * 60)
print("CSV FILES SAVED")
print("=" * 60)
print("\n✓ Combined files (historical + forecast):")
print("  - unemployment_rate_historical_and_forecast.csv")
print("  - fed_funds_rate_historical_and_forecast.csv")
print("\n✓ Prediction-only files (same format as original):")
print("  - unemployment_rate_predicted.csv")
print("  - fed_funds_rate_predicted.csv")

# Create visualizations with visible gaps
fig, axes = plt.subplots(2, 1, figsize=(16, 10))


# Split data into segments to show gaps
def split_by_gaps(df, date_col='date'):
    """Split dataframe where there are gaps > 2 months"""
    df = df.sort_values(date_col)
    df['date_diff'] = df[date_col].diff().dt.days
    gap_indices = df[df['date_diff'] > 60].index

    segments = []
    start_idx = 0
    for gap_idx in gap_indices:
        segments.append(df.iloc[start_idx:gap_idx])
        start_idx = gap_idx
    segments.append(df.iloc[start_idx:])

    return [seg for seg in segments if len(seg) > 0]


# Plot Unemployment Rate
ax1 = axes[0]
historical_unemp = unemp_combined[unemp_combined['type'] == 'historical'].copy()
forecast_unemp = unemp_combined[unemp_combined['type'] == 'forecast'].copy()

unemp_segments = split_by_gaps(historical_unemp)

for i, segment in enumerate(unemp_segments):
    label = 'Historical' if i == 0 else None
    ax1.plot(segment['date'], segment['unemplyment_rate'],
             color='#2E86AB', linewidth=2.5, label=label, marker='o', markersize=3)

ax1.plot(forecast_unemp['date'], forecast_unemp['unemplyment_rate'],
         label='Forecast (24 months)', color='#A23B72', linewidth=2.5,
         linestyle='--', marker='s', markersize=3)

# Add shaded regions for gaps
ax1.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-04-30'),
            alpha=0.2, color='gray', label='Data Gap (COVID period)')
ax1.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-31'),
            alpha=0.2, color='gray')
ax1.axvline(x=historical_unemp['date'].iloc[-1], color='red', linestyle=':',
            linewidth=2, alpha=0.7, label='Forecast Start')

ax1.set_title('Unemployment Rate: Historical Data & 24-Month Forecast\n(COVID-impacted periods excluded)',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Unemployment Rate (%)', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot Federal Funds Rate
ax2 = axes[1]
historical_fed = fedfunds_combined[fedfunds_combined['type'] == 'historical'].copy()
forecast_fed = fedfunds_combined[fedfunds_combined['type'] == 'forecast'].copy()

fed_segments = split_by_gaps(historical_fed)

for i, segment in enumerate(fed_segments):
    label = 'Historical' if i == 0 else None
    ax2.plot(segment['date'], segment['fed_fund_rate'],
             color='#2E86AB', linewidth=2.5, label=label, marker='o', markersize=3)

ax2.plot(forecast_fed['date'], forecast_fed['fed_fund_rate'],
         label='Forecast (24 months)', color='#A23B72', linewidth=2.5,
         linestyle='--', marker='s', markersize=3)

# Add shaded regions for gaps
ax2.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-04-30'),
            alpha=0.2, color='gray', label='Data Gap (COVID period)')
ax2.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-31'),
            alpha=0.2, color='gray')
ax2.axvline(x=historical_fed['date'].iloc[-1], color='red', linestyle=':',
            linewidth=2, alpha=0.7, label='Forecast Start')

ax2.set_title('Federal Funds Rate: Historical Data & 24-Month Forecast\n(COVID-impacted periods excluded)',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Federal Funds Rate (%)', fontsize=12)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('economic_indicators_forecast_with_gaps.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: economic_indicators_forecast_with_gaps.png")

# Display preview of prediction files
print("\n" + "=" * 60)
print("PREDICTION FILES PREVIEW")
print("=" * 60)

print("\nUnemployment Rate Predictions (unemployment_rate_predicted.csv):")
print(unemp_predictions_only.head(12))
print("...")
print(unemp_predictions_only.tail(6))

print("\nFederal Funds Rate Predictions (fed_funds_rate_predicted.csv):")
print(fedfunds_predictions_only.head(12))
print("...")
print(fedfunds_predictions_only.tail(6))

plt.show()

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)