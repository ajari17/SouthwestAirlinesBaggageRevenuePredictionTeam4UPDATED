import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import itertools
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: CREATE HISTORICAL DATA FILE
# ============================================================================
print("=" * 70)
print("STEP 1: PREPARING HISTORICAL DATA")
print("=" * 70)

# Load the combined historical + forecast files
df_unemployment = pd.read_csv('unemployment_rate_historical_and_forecast.csv')
df_unemployment['date'] = pd.to_datetime(df_unemployment['date'])
df_unemployment = df_unemployment.rename(columns={'unemplyment_rate': 'unemployment'})

df_fedfunds = pd.read_csv('fed_funds_rate_historical_and_forecast.csv')
df_fedfunds['date'] = pd.to_datetime(df_fedfunds['date'])

df_cpi = pd.read_csv('airplane_fare_actual_and_forecast_2years.csv')
df_cpi['date'] = pd.to_datetime(df_cpi['date'])
df_cpi = df_cpi.rename(columns={'airplane_fare_cpi': 'cpi'})
# Drop the type column
df_cpi = df_cpi.drop(columns=['type'])

# Load credit card interest scores
df_credit = pd.read_csv('creditcardinterstovrtimefinal.csv')
df_credit['Month'] = pd.to_datetime(df_credit['Month'])
df_credit = df_credit.rename(columns={'Month': 'date', 'Score': 'credit_card_interest'})

# Get the last historical date from credit card data
last_credit_date = df_credit['date'].max()
print(f"Last credit card interest date: {last_credit_date.strftime('%Y-%m')}")

# Split data into historical and future
# Historical: up to last credit card date
df_unemployment_hist = df_unemployment[df_unemployment['date'] <= last_credit_date].copy()
df_fedfunds_hist = df_fedfunds[df_fedfunds['date'] <= last_credit_date].copy()
df_cpi_hist = df_cpi[df_cpi['date'] <= last_credit_date].copy()

# Future: after last credit card date (for forecasting)
df_unemployment_future = df_unemployment[df_unemployment['date'] > last_credit_date].copy()
df_fedfunds_future = df_fedfunds[df_fedfunds['date'] > last_credit_date].copy()
df_cpi_future = df_cpi[df_cpi['date'] > last_credit_date].copy()

print(f"Historical unemployment records: {len(df_unemployment_hist)}")
print(f"Future unemployment records: {len(df_unemployment_future)}")
print(f"Historical fed funds records: {len(df_fedfunds_hist)}")
print(f"Future fed funds records: {len(df_fedfunds_future)}")
print(f"Historical CPI records: {len(df_cpi_hist)}")
print(f"Future CPI records: {len(df_cpi_future)}")

# Merge all historical data
historical = df_credit.merge(df_cpi_hist[['date', 'cpi']], on='date', how='left')
historical = historical.merge(df_unemployment_hist[['date', 'unemployment']], on='date', how='left')
historical = historical.merge(df_fedfunds_hist[['date', 'fed_fund_rate']], on='date', how='left')

# Format date as YYYY-MM
historical['date'] = historical['date'].dt.strftime('%Y-%m')

# Set credit_card_interest to NaN for COVID gap periods
gap_periods = []
# All of 2020
for month in range(1, 13):
    gap_periods.append(f'2020-{month:02d}')
# Jan-Apr 2021
for month in range(1, 5):
    gap_periods.append(f'2021-{month:02d}')
# Jan 2022
gap_periods.append('2022-01')

# Mark gaps
historical.loc[historical['date'].isin(gap_periods), 'credit_card_interest'] = np.nan

# Reorder columns
historical = historical[['date', 'credit_card_interest', 'cpi', 'unemployment', 'fed_fund_rate']]

# Save historical data
historical.to_csv('historical_data.csv', index=False)
print(f"\n✓ Created historical_data.csv with {len(historical)} rows")
print(f"✓ Set {len(gap_periods)} periods to NaN (COVID gap)")
print(f"\nHistorical data preview:")
print(historical.head(10))
print("...")
print(historical.tail(10))

# ============================================================================
# STEP 2: CREATE FUTURE PREDICTIONS FILE (24 months from last date)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: PREPARING FUTURE PREDICTIONS")
print("=" * 70)

# Merge future predictions from all three sources
future_predictions = df_cpi_future[['date', 'cpi']].copy()
future_predictions = future_predictions.merge(df_unemployment_future[['date', 'unemployment']], on='date', how='outer')
future_predictions = future_predictions.merge(df_fedfunds_future[['date', 'fed_fund_rate']], on='date', how='outer')
future_predictions = future_predictions.sort_values('date').reset_index(drop=True)

# **ADD THIS CODE HERE TO FIX THE NaN ERROR**
# Remove rows with any NaN values in the exogenous variables
future_predictions = future_predictions.dropna(subset=['cpi', 'unemployment', 'fed_fund_rate'])

# Limit to exactly 24 months
future_predictions = future_predictions.head(24)

# Reset index
future_predictions = future_predictions.reset_index(drop=True)

# Format date as YYYY-MM
future_predictions['date'] = future_predictions['date'].dt.strftime('%Y-%m')

# Save future predictions
future_predictions.to_csv('future_predictions.csv', index=False)
print(f"✓ Created future_predictions.csv with {len(future_predictions)} rows")
print(f"\nFuture predictions preview:")
print(future_predictions.head(12))
print("...")
print(future_predictions.tail(12))

# ============================================================================
# STEP 3: BUILD SARIMAX MODEL WITH GRID SEARCH
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: BUILDING SARIMAX MODEL")
print("=" * 70)

# Load historical data
hist_data = pd.read_csv('historical_data.csv')
hist_data['date'] = pd.to_datetime(hist_data['date'])
hist_data = hist_data.set_index('date')

# Separate endog and exog
endog = hist_data['credit_card_interest']
exog = hist_data[['cpi', 'unemployment', 'fed_fund_rate']]

print(f"\nTraining data:")
print(f"  Total periods: {len(endog)}")
print(f"  Valid (non-NaN) periods: {endog.notna().sum()}")
print(f"  NaN periods (COVID gap): {endog.isna().sum()}")

# Grid search for best parameters (limited range for speed)
print("\nPerforming grid search for optimal SARIMAX parameters...")
p_range = range(0, 3)
d_range = range(0, 2)
q_range = range(0, 3)
P_range = range(0, 2)
D_range = range(0, 2)
Q_range = range(0, 2)
s = 12

best_aic = np.inf
best_params = None
best_seasonal_params = None

total_combinations = len(list(itertools.product(p_range, d_range, q_range,
                                                P_range, D_range, Q_range)))
current = 0

for param in itertools.product(p_range, d_range, q_range):
    for seasonal_param in itertools.product(P_range, D_range, Q_range):
        try:
            current += 1
            if current % 10 == 0:
                print(f"  Testing combination {current}/{total_combinations}...", end='\r')

            model = SARIMAX(endog,
                            exog=exog,
                            order=param,
                            seasonal_order=seasonal_param + (s,),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            missing='drop')  # KEY: Handle NaN values

            results = model.fit(disp=False, maxiter=200)

            if results.aic < best_aic:
                best_aic = results.aic
                best_params = param
                best_seasonal_params = seasonal_param + (s,)
        except:
            continue

print(f"\n\n✓ Grid search complete!")
print(f"  Best order: {best_params}")
print(f"  Best seasonal order: {best_seasonal_params}")
print(f"  Best AIC: {best_aic:.2f}")

# Fit final model with best parameters
print(f"\nFitting final SARIMAX model...")
final_model = SARIMAX(endog,
                      exog=exog,
                      order=best_params,
                      seasonal_order=best_seasonal_params,
                      enforce_stationarity=False,
                      enforce_invertibility=False,
                      missing='drop')

final_results = final_model.fit(disp=False)
print("✓ Model fitted successfully!")

# Display model summary
print("\n" + "-" * 70)
print("MODEL SUMMARY")
print("-" * 70)
print(f"AIC: {final_results.aic:.2f}")
print(f"BIC: {final_results.bic:.2f}")
print(f"Log-Likelihood: {final_results.llf:.2f}")

# ============================================================================
# STEP 4: GENERATE FORECAST USING FUTURE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: GENERATING 24-MONTH FORECAST")
print("=" * 70)

# Load future predictions
future_pred = pd.read_csv('future_predictions.csv')
future_exog = future_pred[['cpi', 'unemployment', 'fed_fund_rate']]

print(f"Using future predictions for {len(future_exog)} months")
print(f"  CPI range: {future_exog['cpi'].min():.1f} - {future_exog['cpi'].max():.1f}")
print(f"  Unemployment range: {future_exog['unemployment'].min():.2f}% - {future_exog['unemployment'].max():.2f}%")
print(f"  Fed rate range: {future_exog['fed_fund_rate'].min():.2f}% - {future_exog['fed_fund_rate'].max():.2f}%")

# Generate forecast
forecast_result = final_results.get_forecast(steps=len(future_exog), exog=future_exog)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'date': pd.to_datetime(future_pred['date']),
    'forecast': forecast_mean.values,
    'lower_ci': forecast_ci.iloc[:, 0].values,
    'upper_ci': forecast_ci.iloc[:, 1].values
})

print("\n✓ Forecast generated successfully!")
print(f"\nForecast summary:")
print(f"  Mean forecast: {forecast_mean.mean():.1f}")
print(f"  Range: {forecast_mean.min():.1f} - {forecast_mean.max():.1f}")
print(f"\nFirst 12 months forecast:")
print(forecast_df.head(12)[['date', 'forecast', 'lower_ci', 'upper_ci']])


# Save forecast
forecast_df.to_csv('credit_card_interest_forecast.csv', index=False)
print("\n✓ Saved forecast to credit_card_interest_forecast.csv")

# ============================================================================
# STEP 5: CREATE VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: CREATING VISUALIZATION")
print("=" * 70)

fig, ax = plt.subplots(figsize=(16, 8))

# Prepare historical data (remove NaN values for plotting)
hist_valid = hist_data[hist_data['credit_card_interest'].notna()].copy()
hist_valid = hist_valid.reset_index()


# Split historical data into segments to show gaps
def split_by_gaps(df, date_col='date', max_gap_days=90):
    """Split dataframe where there are large time gaps"""
    df = df.sort_values(date_col).copy()
    df['date_diff'] = df[date_col].diff().dt.days
    gap_indices = df[df['date_diff'] > max_gap_days].index

    segments = []
    start_idx = 0
    for gap_idx in gap_indices:
        if start_idx < gap_idx:
            segments.append(df.iloc[start_idx:gap_idx])
        start_idx = gap_idx
    if start_idx < len(df):
        segments.append(df.iloc[start_idx:])

    return [seg for seg in segments if len(seg) > 0]


# Plot historical data in segments (to show gaps)
hist_segments = split_by_gaps(hist_valid)
for i, segment in enumerate(hist_segments):
    label = 'Historical Data' if i == 0 else None
    ax.plot(segment['date'], segment['credit_card_interest'],
            color='#2E86AB', linewidth=2.5, marker='o', markersize=4,
            label=label, zorder=3)

# Plot forecast
ax.plot(forecast_df['date'], forecast_df['forecast'],
        color='#A23B72', linewidth=2.5, linestyle='--', marker='s',
        markersize=4, label='24-Month Forecast', zorder=3)

# Plot confidence interval
ax.fill_between(forecast_df['date'],
                forecast_df['lower_ci'],
                forecast_df['upper_ci'],
                alpha=0.3, color='#A23B72', label='95% Confidence Interval', zorder=2)

# Add shaded regions for COVID gaps
ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-04-30'),
           alpha=0.15, color='gray', label='Data Gap (COVID)', zorder=1)
ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-31'),
           alpha=0.15, color='gray', zorder=1)

# Add vertical line at forecast start
last_historical_date = hist_valid['date'].max()
ax.axvline(x=last_historical_date, color='red', linestyle=':',
           linewidth=2, alpha=0.7, label='Forecast Start', zorder=2)

# Formatting
ax.set_title('Credit Card Interest Score: SARIMAX Forecast with Economic Predictors\n' +
             f'Model: SARIMAX{best_params}{best_seasonal_params} | AIC: {best_aic:.1f}',
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Credit Card Interest Score', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', labelsize=10)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('credit_card_interest_sarimax_forecast.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to credit_card_interest_sarimax_forecast.png")

plt.show()

# ============================================================================
# STEP 4: GENERATE FORECAST USING FUTURE PREDICTIONS
# ============================================================================
print("\n" + "="*70)
print("STEP 4: GENERATING 24-MONTH FORECAST")
print("="*70)

# Load future predictions
future_pred = pd.read_csv('future_predictions.csv')
future_exog = future_pred[['cpi', 'unemployment', 'fed_fund_rate']]

print(f"Using future predictions for {len(future_exog)} months")
print(f"  CPI range: {future_exog['cpi'].min():.1f} - {future_exog['cpi'].max():.1f}")
print(f"  Unemployment range: {future_exog['unemployment'].min():.2f}% - {future_exog['unemployment'].max():.2f}%")
print(f"  Fed rate range: {future_exog['fed_fund_rate'].min():.2f}% - {future_exog['fed_fund_rate'].max():.2f}%")

# Generate forecast
forecast_result = final_results.get_forecast(steps=len(future_exog), exog=future_exog)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'date': pd.to_datetime(future_pred['date']),
    'forecast': forecast_mean.values,
    'lower_ci': forecast_ci.iloc[:, 0].values,
    'upper_ci': forecast_ci.iloc[:, 1].values
})

print("\n✓ Forecast generated successfully!")
print(f"\nForecast summary:")
print(f"  Mean forecast: {forecast_mean.mean():.1f}")
print(f"  Range: {forecast_mean.min():.1f} - {forecast_mean.max():.1f}")
print(f"\nFirst 12 months forecast:")
print(forecast_df.head(12)[['date', 'forecast', 'lower_ci', 'upper_ci']])

# Save forecast
forecast_df.to_csv('credit_card_interest_forecast.csv', index=False)
print("\n✓ Saved forecast to credit_card_interest_forecast.csv")

# **NEW: Create combined historical + forecasted file**
# Prepare historical data
historical_interest = hist_valid[['date', 'credit_card_interest']].copy()
historical_interest = historical_interest.rename(columns={'credit_card_interest': 'credit_card_interest_score'})
historical_interest['type'] = 'historical'

# Prepare forecasted data
forecasted_interest = forecast_df[['date', 'forecast']].copy()
forecasted_interest = forecasted_interest.rename(columns={'forecast': 'credit_card_interest_score'})
forecasted_interest['type'] = 'forecast'

# Combine historical and forecasted
combined_interest = pd.concat([historical_interest, forecasted_interest], ignore_index=True)
combined_interest['date'] = combined_interest['date'].dt.strftime('%Y-%m')

# Reorder columns
combined_interest = combined_interest[['date', 'credit_card_interest_score', 'type']]

# Save combined file
combined_interest.to_csv('forecasted_interest_credit_card.csv', index=False)
print("✓ Saved combined historical + forecast to forecasted_interest_credit_card.csv")

print(f"\nCombined file preview:")
print(f"  Total records: {len(combined_interest)}")
print(f"  Historical records: {len(combined_interest[combined_interest['type'] == 'historical'])}")
print(f"  Forecast records: {len(combined_interest[combined_interest['type'] == 'forecast'])}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print("\n📁 Files Created:")
print("  1. historical_data.csv - Historical data with COVID gaps marked as NaN")
print("  2. future_predictions.csv - Your economic predictions for next 24 months")
print("  3. credit_card_interest_forecast.csv - Forecast results with confidence intervals")
print("  4. credit_card_interest_sarimax_forecast.png - Visualization")

print("\n📊 Model Performance:")
print(f"  Order: SARIMAX{best_params}{best_seasonal_params}")
print(f"  AIC: {best_aic:.2f}")
print(f"  Training samples (after removing gaps): {endog.notna().sum()}")

print("\n🔮 Forecast Summary:")
print(f"  Last historical value: {hist_valid['credit_card_interest'].iloc[-1]:.1f}")
print(f"  12-month forecast: {forecast_mean.iloc[11]:.1f}")
print(f"  24-month forecast: {forecast_mean.iloc[-1]:.1f}")
print(f"  Average forecast: {forecast_mean.mean():.1f}")

print("\n✅ Key Features Implemented:")
print("  ✓ COVID gaps properly handled with missing='drop'")
print("  ✓ Grid search for optimal SARIMAX parameters")
print("  ✓ External predictors (CPI, unemployment, fed rate) incorporated")
print("  ✓ 24-month forecast with confidence intervals")
print("  ✓ Visualization with gaps clearly shown")
print("  ✓ Using combined historical + forecast files")

print("\n" + "=" * 70)