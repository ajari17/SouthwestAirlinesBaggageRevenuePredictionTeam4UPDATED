import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Create the full dataset
dates = pd.date_range(start='2015-01-01', periods=129, freq='MS')
values = [
    283.152, 288.626, 287.362, 294.603, 319.401, 324.953, 297.324, 277.768, 274.897, 285.837, 294.143, 278.658,
    278.334, 283.520, 283.584, 295.909, 309.540, 309.679, 283.501, 268.040, 267.457, 270.922, 274.761, 265.436,
    269.241, 280.517, 283.583, 294.034, 300.609, 296.384, 276.308, 259.359, 259.143, 267.297, 267.970, 254.947,
    255.496, 265.272, 267.482, 273.817, 280.804, 278.937, 264.994, 255.877, 258.196, 265.930, 263.809, 248.290,
    248.433, 259.049, 259.698, 268.767, 283.275, 283.001, 268.314, 259.849, 263.149, 269.871, 268.994, 252.411,
    255.200, 265.142, 232.113, 203.342, 201.649, 206.066, 204.785, 199.496, 197.424, 215.993, 223.360, 205.983,
    200.825, 197.204, 197.134, 222.953, 250.209, 256.684, 243.613, 212.882, 198.975, 205.994, 215.159, 208.954,
    210.762, 222.227, 243.689, 297.143, 344.853, 344.101, 311.205, 283.911, 284.313, 294.340, 292.656, 268.519,
    264.629, 281.216, 286.814, 294.550, 298.489, 279.224, 253.345, 246.185, 246.151, 255.480, 257.222, 243.348,
    247.606, 263.952, 266.481, 277.450, 280.958, 265.061, 246.222, 243.011, 250.030, 265.939, 269.336, 262.556,
    265.273, 262.136, 252.620, 255.592, 260.319, 255.852, 247.859, 250.982, 258.027
]

df_full = pd.DataFrame({
    'date': dates,
    'airplane_fare_cpi': values
})

# CLEAN DATA: Remove 2020, first 4 months of 2021, and Jan 2022
print("="*80)
print("DATA CLEANING")
print("="*80)

# Create mask for periods to exclude BEFORE setting index
exclude_mask = (
    (df_full['date'].dt.year == 2020) |  # All of 2020
    ((df_full['date'].dt.year == 2021) & (df_full['date'].dt.month <= 4)) |  # Jan-Apr 2021
    ((df_full['date'].dt.year == 2022) & (df_full['date'].dt.month == 1))  # Jan 2022
)

# Keep only non-excluded data
df_cleaned = df_full[~exclude_mask].copy()

print(f"\nOriginal data points: {len(df_full)}")
print(f"Cleaned data points: {len(df_cleaned)}")
print(f"Removed data points: {len(df_full) - len(df_cleaned)}")

print("\nRemoved periods:")
print("  - All of 2020 (12 months)")
print("  - January - April 2021 (4 months)")
print("  - January 2022 (1 month)")
print(f"  Total removed: 17 months")

print(f"\nCleaned data range: {df_cleaned['date'].min().strftime('%Y-%m')} to {df_cleaned['date'].max().strftime('%Y-%m')}")

# Now set index after filtering
df = df_cleaned.set_index('date').copy()

# Split data - use last 12 months as test set
train = df[:-12]
test = df[-12:]

print(f"\nTraining data: {train.index[0].strftime('%Y-%m')} to {train.index[-1].strftime('%Y-%m')} ({len(train)} months)")
print(f"Test data: {test.index[0].strftime('%Y-%m')} to {test.index[-1].strftime('%Y-%m')} ({len(test)} months)")

# Visualize the cleaned data
plt.figure(figsize=(15, 10))

# Plot 1: Time series comparison (original vs cleaned)
plt.subplot(3, 1, 1)
plt.plot(df_full['date'], df_full['airplane_fare_cpi'], 
         label='Original Data', linewidth=1, alpha=0.5, color='gray')
plt.plot(df_cleaned['date'], df_cleaned['airplane_fare_cpi'], 
         label='Cleaned Data', linewidth=2, color='blue')

# Highlight removed periods
removed_data = df_full[exclude_mask]
plt.scatter(removed_data['date'], removed_data['airplane_fare_cpi'], 
           color='red', s=50, alpha=0.6, label='Removed (2020, Early 2021, Jan 2022)', zorder=5)

plt.axvline(x=train.index[-1], color='orange', linestyle='--', label='Train/Test Split')
plt.title('Airplane Fare CPI - Original vs Cleaned Data', fontsize=14, fontweight='bold')
plt.ylabel('CPI')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Seasonal decomposition
plt.subplot(3, 1, 2)
decomposition = seasonal_decompose(train['airplane_fare_cpi'], model='additive', period=12)
plt.plot(decomposition.seasonal, label='Seasonal Component', color='green')
plt.title('Seasonal Pattern (Cleaned Data)', fontsize=14, fontweight='bold')
plt.ylabel('Seasonal Effect')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Year-over-year comparison
plt.subplot(3, 1, 3)
for year in range(2015, 2026):
    year_data = df[df.index.year == year]
    if len(year_data) > 0:
        plt.plot(year_data.index.month, year_data['airplane_fare_cpi'], 
                marker='o', label=str(year), linewidth=2)
plt.xlabel('Month')
plt.ylabel('CPI')
plt.title('Seasonal Pattern by Year (Cleaned Data)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 13))

plt.tight_layout()
plt.savefig('airplane_fare_analysis_cleaned.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("MODEL TRAINING AND COMPARISON")
print("="*80)

# Model 1: SARIMA
print("\n1. SARIMA Model (Seasonal ARIMA)")
try:
    sarima_model = SARIMAX(train['airplane_fare_cpi'], 
                           order=(1, 1, 1), 
                           seasonal_order=(1, 1, 1, 12),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    sarima_fit = sarima_model.fit(disp=False)
    sarima_test_pred = sarima_fit.forecast(steps=12)
    sarima_mae = mean_absolute_error(test, sarima_test_pred)
    sarima_rmse = np.sqrt(mean_squared_error(test, sarima_test_pred))
    print(f"   Test MAE: {sarima_mae:.2f}")
    print(f"   Test RMSE: {sarima_rmse:.2f}")
except Exception as e:
    print(f"   Error: {e}")
    sarima_test_pred = None
    sarima_mae = float('inf')

# Model 2: Holt-Winters (Exponential Smoothing)
print("\n2. Holt-Winters Exponential Smoothing")
try:
    hw_model = ExponentialSmoothing(train['airplane_fare_cpi'], 
                                     seasonal_periods=12,
                                     trend='add',
                                     seasonal='add')
    hw_fit = hw_model.fit()
    hw_test_pred = hw_fit.forecast(steps=12)
    hw_mae = mean_absolute_error(test, hw_test_pred)
    hw_rmse = np.sqrt(mean_squared_error(test, hw_test_pred))
    print(f"   Test MAE: {hw_mae:.2f}")
    print(f"   Test RMSE: {hw_rmse:.2f}")
except Exception as e:
    print(f"   Error: {e}")
    hw_test_pred = None
    hw_mae = float('inf')

# Model 3: Seasonal Naive (Baseline)
print("\n3. Seasonal Naive (Baseline)")
seasonal_naive_pred = train['airplane_fare_cpi'][-12:].values
sn_mae = mean_absolute_error(test, seasonal_naive_pred)
sn_rmse = np.sqrt(mean_squared_error(test, seasonal_naive_pred))
print(f"   Test MAE: {sn_mae:.2f}")
print(f"   Test RMSE: {sn_rmse:.2f}")

# Select best model
models = {
    'SARIMA': (sarima_mae, sarima_fit if sarima_test_pred is not None else None),
    'Holt-Winters': (hw_mae, hw_fit if hw_test_pred is not None else None),
    'Seasonal Naive': (sn_mae, None)
}

best_model_name = min(models, key=lambda x: models[x][0])
print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name} (MAE: {models[best_model_name][0]:.2f})")
print(f"{'='*80}")

# Retrain best model on full cleaned dataset and forecast 24 MONTHS
print("\n" + "="*80)
print("FORECASTING NEXT 24 MONTHS (2 YEARS)")
print("="*80)

forecast_periods = 24  # Changed from 12 to 24

if best_model_name == 'SARIMA':
    final_model = SARIMAX(df['airplane_fare_cpi'], 
                          order=(1, 1, 1), 
                          seasonal_order=(1, 1, 1, 12),
                          enforce_stationarity=False,
                          enforce_invertibility=False)
    final_fit = final_model.fit(disp=False)
    forecast = final_fit.forecast(steps=forecast_periods)
    
elif best_model_name == 'Holt-Winters':
    final_model = ExponentialSmoothing(df['airplane_fare_cpi'], 
                                        seasonal_periods=12,
                                        trend='add',
                                        seasonal='add')
    final_fit = final_model.fit()
    forecast = final_fit.forecast(steps=forecast_periods)
    
else:  # Seasonal Naive
    # Repeat the last 12 months twice for 2 years
    forecast_values = np.tile(df['airplane_fare_cpi'][-12:].values, 2)
    forecast = pd.Series(forecast_values)

# Create forecast dates
forecast_dates = pd.date_range(start='2025-10-01', periods=forecast_periods, freq='MS')
forecast_df = pd.DataFrame({
    'date': forecast_dates,
    'airplane_fare_cpi': forecast.values,
    'type': 'forecast'
})

print("\nForecast for October 2025 - September 2027 (24 months):")
print(forecast_df[['date', 'airplane_fare_cpi']].to_string(index=False))

# CREATE COMBINED OUTPUT FILE: Actual + Predicted
print("\n" + "="*80)
print("CREATING COMBINED OUTPUT FILE")
print("="*80)

# Prepare actual data
actual_df = df.reset_index()
actual_df['type'] = 'actual'
actual_df = actual_df[['date', 'airplane_fare_cpi', 'type']]

# Combine actual and forecast
combined_df = pd.concat([actual_df, forecast_df], ignore_index=True)

# Save combined file
output_filename = 'airplane_fare_actual_and_forecast_2years.csv'
combined_df.to_csv(output_filename, index=False)

print(f"\n✓ Combined file saved: '{output_filename}'")
print(f"  - Actual data points: {len(actual_df)}")
print(f"  - Forecast data points: {len(forecast_df)} (24 months)")
print(f"  - Total data points: {len(combined_df)}")
print(f"  - Date range: {combined_df['date'].min().strftime('%Y-%m')} to {combined_df['date'].max().strftime('%Y-%m')}")

# Also save forecast only
forecast_df[['date', 'airplane_fare_cpi']].to_csv('airplane_fare_forecast_only_2years.csv', index=False)
print(f"\n✓ Forecast-only file saved: 'airplane_fare_forecast_only_2years.csv'")

# Display preview of combined file
print("\n" + "="*80)
print("PREVIEW OF COMBINED FILE")
print("="*80)
print("\nLast 5 actual values:")
print(combined_df[combined_df['type'] == 'actual'].tail(5).to_string(index=False))
print("\nFirst 12 forecast values (Year 1):")
print(combined_df[combined_df['type'] == 'forecast'].head(12).to_string(index=False))
print("\nLast 12 forecast values (Year 2):")
print(combined_df[combined_df['type'] == 'forecast'].tail(12).to_string(index=False))

# Visualization of forecast with cleaned data
plt.figure(figsize=(18, 8))

# Plot historical cleaned data
plt.plot(df.index, df['airplane_fare_cpi'], label='Actual Data (Cleaned)', 
         linewidth=2, color='blue', marker='o', markersize=3)

# Plot test predictions vs actual
if best_model_name == 'SARIMA' and sarima_test_pred is not None:
    plt.plot(test.index, sarima_test_pred, 'g--', label='Test Predictions', linewidth=2)
elif best_model_name == 'Holt-Winters' and hw_test_pred is not None:
    plt.plot(test.index, hw_test_pred, 'g--', label='Test Predictions', linewidth=2)

plt.plot(test.index, test['airplane_fare_cpi'], 'go', 
         label='Actual Test Data', markersize=8)

# Plot future forecast (24 months)
plt.plot(forecast_dates, forecast.values, 'r-', 
         label=f'24-Month Forecast ({best_model_name})', linewidth=2.5, marker='o', markersize=6)

# Add confidence interval (approximate)
if best_model_name in ['SARIMA', 'Holt-Winters']:
    std_error = np.std(df['airplane_fare_cpi'][-24:]) * 1.5
    # Increase uncertainty for year 2
    std_error_yr1 = std_error
    std_error_yr2 = std_error * 1.3
    
    plt.fill_between(forecast_dates[:12], 
                     forecast.values[:12] - std_error_yr1, 
                     forecast.values[:12] + std_error_yr1,
                     alpha=0.2, color='red', label='Confidence Interval (Year 1)')
    
    plt.fill_between(forecast_dates[12:], 
                     forecast.values[12:] - std_error_yr2, 
                     forecast.values[12:] + std_error_yr2,
                     alpha=0.15, color='orange', label='Confidence Interval (Year 2)')

plt.axvline(x=df.index[-1], color='orange', linestyle='--', 
            linewidth=2, label='Forecast Start')

# Mark removed periods with shading
plt.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'), 
            alpha=0.1, color='red', label='Excluded: 2020')
plt.axvspan(pd.Timestamp('2021-01-01'), pd.Timestamp('2021-04-30'), 
            alpha=0.1, color='red')
plt.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-31'), 
            alpha=0.1, color='red')

plt.xlabel('Date', fontsize=12)
plt.ylabel('Airplane Fare CPI', fontsize=12)
plt.title(f'Airplane Fare CPI - 24 Month Forecast (Oct 2025 - Sep 2027)\nUsing {best_model_name} Model', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=9, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('airplane_fare_forecast_2years_cleaned.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print("\n" + "="*80)
print("FORECAST SUMMARY STATISTICS")
print("="*80)

# Overall 24 months
print(f"\nOverall (24 months):")
print(f"Mean predicted CPI: {forecast.mean():.2f}")
print(f"Min predicted CPI: {forecast.min():.2f} (Month: {forecast_dates[forecast.argmin()].strftime('%B %Y')})")
print(f"Max predicted CPI: {forecast.max():.2f} (Month: {forecast_dates[forecast.argmax()].strftime('%B %Y')})")
print(f"Standard deviation: {forecast.std():.2f}")

# Year 1 vs Year 2
forecast_year1 = forecast.values[:12]
forecast_year2 = forecast.values[12:]

print(f"\nYear 1 (Oct 2025 - Sep 2026):")
print(f"  Mean CPI: {forecast_year1.mean():.2f}")
print(f"  Min CPI: {forecast_year1.min():.2f}")
print(f"  Max CPI: {forecast_year1.max():.2f}")

print(f"\nYear 2 (Oct 2026 - Sep 2027):")
print(f"  Mean CPI: {forecast_year2.mean():.2f}")
print(f"  Min CPI: {forecast_year2.min():.2f}")
print(f"  Max CPI: {forecast_year2.max():.2f}")

print(f"\nYear-over-Year Change: {((forecast_year2.mean() - forecast_year1.mean()) / forecast_year1.mean() * 100):+.2f}%")

# Compare with historical averages
recent_avg = df['airplane_fare_cpi'][-12:].mean()
historical_avg = df['airplane_fare_cpi'].mean()
print(f"\nComparison:")
print(f"Last 12 months average: {recent_avg:.2f}")
print(f"Historical average (cleaned data): {historical_avg:.2f}")
print(f"Forecast average (24 months): {forecast.mean():.2f}")
print(f"Change from recent average: {((forecast.mean() - recent_avg) / recent_avg * 100):+.2f}%")

print("\n" + "="*80)
print("SEASONAL PATTERN IN FORECAST")
print("="*80)

# Analyze both years
for year_num in [1, 2]:
    start_idx = (year_num - 1) * 12
    end_idx = year_num * 12
    year_data = forecast_df.iloc[start_idx:end_idx].copy()
    
    summer_months = year_data[year_data['date'].dt.month.isin([5, 6, 7])]['airplane_fare_cpi'].mean()
    winter_months = year_data[year_data['date'].dt.month.isin([12, 1, 2])]['airplane_fare_cpi'].mean()
    
    print(f"\nYear {year_num} ({'2025-2026' if year_num == 1 else '2026-2027'}):")
    print(f"  Summer average (May-Jul): {summer_months:.2f}")
    print(f"  Winter average (Dec-Feb): {winter_months:.2f}")
    print(f"  Summer premium: {summer_months - winter_months:.2f} ({((summer_months/winter_months - 1) * 100):.1f}%)")

print("\n" + "="*80)
print("OUTPUT FILES CREATED")
print("="*80)
print(f"1. {output_filename} - Actual + 24-month Forecast combined")
print(f"2. airplane_fare_forecast_only_2years.csv - 24-month Forecast only")
print(f"3. airplane_fare_analysis_cleaned.png - Data analysis visualization")
print(f"4. airplane_fare_forecast_2years_cleaned.png - 24-month forecast visualization")
print("\n✓ Analysis complete! Now forecasting through September 2027.")