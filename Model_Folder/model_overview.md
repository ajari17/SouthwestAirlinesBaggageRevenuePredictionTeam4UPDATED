# Project Overview

This project forecasts key drivers and converts those forecasts into baggage-fee revenue. It builds 24‑month forecasts for:
- Economic indicators: Airfare CPI, Unemployment Rate, and Federal Funds Rate 
- Credit Card Interest (Google Trends “interest score”), using the economic indicators as exogenous predictors 
- Net Passenger Volume, using the same economic indicators as exogenous predictors 

It then combines:
- Monthly bags-per-passenger ratios (BpP) 
- Forecasted passenger volume 
- Forecasted credit-card percentage (share of passengers with a qualifying card) 

to compute paid bags and baggage-fee revenue at a per-bag fee, and produces visuals and summary tables.

---

## Components and How They Fit Together

- Airfare CPI, Unemployment, and Fed Funds are forecast for 24 months and saved as combined historical+forecast files; they serve as exogenous inputs to other models.
- Credit Card Interest (Google Trends) is forecast with SARIMAX using those exogenous inputs; COVID-impacted months are handled as gaps, and outputs include forecast CSVs and a visualization.
- Passenger Volume is forecast with a SARIMAX model using standardized exogenous variables and a log-transformed target; outputs include a 24‑month forecast with confidence intervals.
- BpP ratios are derived from monthly bag and passenger counts, visualized for seasonality, and used in revenue calculations.
- Revenue is computed by merging passenger forecasts with BpP ratios and credit-card percentage, distinguishing free vs paid bags, and multiplying paid bags by the bag fee; outputs include monthly charts and yearly tables.

---

## File-by-File Overview

### 1) cpimodel2.py — Airfare CPI: Cleaning, Model Selection, and 24‑Month Forecast
- Removes COVID-affected months (all of 2020, Jan–Apr 2021, Jan 2022) before modeling and visualizes cleaned vs original data, seasonal patterns, and a train/test split.
- Trains and compares SARIMA, retrains on all cleaned data, and forecasts 24 months (Oct 2025–Sep 2027).
- Saves combined actual+forecast and forecast-only CSVs and an analysis/forecast visualization, with approximate confidence bands.

### 2) fed_funds_and_unempmodels.py — Unemployment and Federal Funds Rate: SARIMAX Forecasts
- Forecasts unemployment and fed funds for 24 months with SARIMA and produces combined historical+forecast datasets and prediction-only files with consistent date formats.
- Creates a visualization highlighting gaps and the forecast start for both series.

### 3) comprehensive_creditcardinterest_predictor.py — Credit Card Interest (Google Trends): SARIMAX with Exogenous Drivers
- Aligns Credit Card Interest with CPI, Unemployment, and Fed Funds; sets COVID gap months to NaN and uses missing='drop' in SARIMAX to handle gaps during estimation.
- Builds future_predictions.csv by merging future CPI, Unemployment, and Fed Funds and drops any rows with NaNs; limits the horizon to 24 months.
- Performs a grid search over SARIMAX orders to select the best model by AIC; fits the final model, forecasts 24 months with 95% confidence intervals, and exports results and a visualization with shaded gaps and a forecast-start marker.

### 4) passenger_forecaster_variables.py — Passenger Volume: SARIMAX with Standardized Exogenous Variables
- Merges BTS passenger data with Airfare CPI, Fed Funds, and Unemployment; drops rows with missing exogenous values; checks for NaNs and confirms 1:1 date alignment for the forecasting matrix.
- Standardizes exogenous variables (StandardScaler) and log-transforms the passenger target; fits a simplified SARIMAX with seasonal differencing only (order=(0,0,0), seasonal_order=(0,1,0,12)).
- Generates a forecast up to the minimum horizon across exogenous inputs (capped at 24), back-transforms predictions and confidence intervals, and saves a CSV.

### 5) final_model2.py — Baggage-Fee Revenue Pipeline
- Loads passenger data, computes month numbers, merges monthly BpP ratios by month, and merges credit-card percentage (ScaledPercent/100) by year-month.
- Applies the revenue formula to compute free vs paid bags and multiplies paid bags by the bag fee (35.00) to get revenue; produces a monthly line chart with color splits, a forecast-only bar chart, and a yearly revenue table image.

### 6) 02_hybrid_bpp_forecast.py — Bags per Passenger (BpP): Seasonality and Hybrid Forecast
- Reads final_bag_pass_count.csv, parses the Date column explicitly as '%m-%y', computes bags_per_passenger as Bag_count ÷ PassengerCount, sets a monthly index, and visualizes BpP seasonality over the year.
- Maps the ratios (bag per passenger) for each month
-*WE ONLY USED THE BAGS PER PASSENGER RATIO FROM THIS, WE DID NOT USE THE FORECASTED REVENUE FROM THIS FOR OUR FINAL MODEL*

---

## Data Flow

1) Forecast Economic Drivers
- Airfare CPI (cleaned and forecast 24 months) 
- Unemployment and Fed Funds (forecast 24 months) 

2) Build Exogenous Matrices and Targets
- Credit Card Interest: Align historical target with exogenous drivers, mark COVID gaps NaN, and build future exogenous inputs (CPI, Unemployment, Fed Funds) for 24 months.
- Passengers: Merge BTS passenger totals with the three exogenous drivers; standardize exogenous variables; log-transform target.

3) Forecast Models
- Credit Card Interest: SARIMAX with grid-searched parameters; 24-month forecast + 95% CI.
- Passenger Volume: SARIMAX (seasonal differencing only); forecast up to 24 months; back-transform and save CI bands.

4) Revenue Computation
- Merge passenger forecast with monthly BpP ratios (by month number) and credit-card percentage (ScaledPercent/100), then compute monthly revenue.
- Optional hybrid path: Merge SARIMA 2026 passenger forecast with BpP ratios and a chosen per-bag fee to produce a standalone annual revenue view.

---

## Handling COVID-Affected Periods and Gaps

- Airfare CPI modeling excludes all of 2020, Jan–Apr 2021, and Jan 2022, and highlights those ranges in visuals.
- Unemployment and Fed Funds visuals shade COVID-impacted periods and mark forecast starts.
- Credit Card Interest sets those months to NaN in the target and uses missing='drop' in SARIMAX to avoid convergence issues while retaining aligned exogenous information.

---

## Bags per Passenger (BpP)

- Source and computation: From final_bag_pass_count.csv, parse Date as month-year ('%m-%y') and compute bags_per_passenger = Bag_count ÷ PassengerCount; visualize monthly seasonality across the year.
- Seasonal factors: Load bpp_seasonal_factors.csv and merge by month number with passenger forecasts to convert passengers to bag counts.
- Usage:
  - Final pipeline: Join BpP ratios and credit_card_pct with passenger forecasts to compute fee-bearing bags using the formula below.
  - Hybrid path (illustrative): Use SARIMA passenger forecast for 2026 and multiply by BpP ratios; apply a per-bag fee (example 40.00) to estimate revenue; visualize monthly and total revenue for 2026.

---

## Revenue Formula

For each month:
- total_bags = passengers × bag_ratio
- cc_pax = passengers × credit_card_pct
- free_bags = min(cc_pax, total_bags)
- paid_bags = total_bags − free_bags
- revenue = paid_bags × 35.00 

This logic converts passenger forecasts and BpP seasonality into fee-bearing bags after accounting for credit-card benefits, then multiplies by the per-bag fee to estimate monthly revenue.

---

## Notes and Assumptions

- Exogenous inputs must be aligned and complete for the forecast horizon; rows with any NaNs are dropped when building future_predictions.csv in the Credit Card Interest pipeline to ensure clean exog matrices.
- Passenger model standardizes exogenous variables and log-transforms the target; forecasts and CI bands are back-transformed to the original scale.
- The final revenue pipeline uses a bag fee of 35.00 in compute time; the hybrid BpP demonstration uses 40.00 for illustration only. Adjust these fees as needed.
- COVID gaps are explicitly handled across models and visuals to reduce bias and improve stability.