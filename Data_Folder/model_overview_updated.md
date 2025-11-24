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
- Passenger Volume is forecast with a SARIMAX model using standardized exogenous variables and a log-transformed target; outputs include a 24‑month forecast.
- BpP (Bag per passenger) ratios are derived from monthly bag and passenger counts (from Southwest Data), visualized for seasonality, and used in revenue calculations.
- Revenue is computed by merging passenger forecasts with BpP ratios and credit-card percentage, distinguishing free vs paid bags, and multiplying paid bags by the bag fee; outputs include monthly charts and yearly tables.

---

## File-by-File Overview

### 1) cpimodel2.py — Airfare CPI: Cleaning, Model Selection, and 24‑Month Forecast
- Removes COVID-affected months (all of 2020, Jan–Apr 2021, Jan 2022) before modeling and visualizes cleaned vs original data, seasonal patterns, and a train/test split.
- Trains and compares SARIMA, retrains on all cleaned data, and forecasts 24 months (Oct 2025–Sep 2027).
- Saves combined actual+forecast and forecast-only CSVs and an analysis/forecast visualization.

### 2) fed_funds_and_unempmodels.py — Unemployment and Federal Funds Rate: SARIMAX Forecasts
- Removes COVID-affected months (all of 2020, Jan–Apr 2021, Jan 2022) before modeling.
- Forecasts unemployment and fed funds for 24 months with SARIMA and produces combined historical+forecast datasets and prediction-only files with consistent date formats.
- Creates a visualization highlighting gaps and the forecast start for both series.

### 3) comprehensive_creditcardinterest_predictor.py — Credit Card Interest (Google Trends): SARIMAX with Exogenous Drivers
- Removes COVID-affected months (all of 2020, Jan–Apr 2021, Jan 2022) before modeling.
- Takes the baseline 6% credit card percentage and scales it up and down using Google Trends "Interest Score", coupled with the exogenous variables.
- Aligns Credit Card Interest with CPI, Unemployment, and Fed Funds; sets COVID gap months to NaN and uses missing='drop' in SARIMAX to handle gaps during estimation.
- Builds future_predictions.csv by merging future CPI, Unemployment, and Fed Funds and drops any rows with NaNs; limits the horizon to 24 months.
- Performs a grid search over SARIMAX orders to select the best model by AIC; fits the final model, forecasts 24 months with 95% confidence intervals, and exports results and a visualization with shaded gaps and a forecast-start marker.

### 4) passenger_forecaster_variables.py — Passenger Volume: SARIMAX with Standardized Exogenous Variables
- Removes COVID-affected months (all of 2020, Jan–Apr 2021, Jan 2022) before modeling.
- Merges BTS passenger data with Airfare CPI, Fed Funds, and Unemployment; drops rows with missing exogenous values; checks for NaNs and confirms 1:1 date alignment for the forecasting matrix.
- Standardizes exogenous variables (StandardScaler) and log-transforms the passenger target; fits a simplified SARIMAX with seasonal differencing only (order=(0,0,0), seasonal_order=(0,1,0,12)).
- Generates a forecast up to the minimum horizon across exogenous inputs (capped at 24), back-transforms predictions and confidence intervals, and saves a CSV.

### 5) final_model2.py — Baggage-Fee Revenue Pipeline
- Loads passenger data, computes month numbers, merges monthly BpP ratios by month, and merges credit-card percentage (ScaledPercent/100) by year-month.
- Applies the revenue formula to compute free vs paid bags and multiplies paid bags by the bag fee (35.00) to get revenue; produces a monthly line chart with color splits, a forecast-only bar chart, and a yearly revenue table image.

### 6) 02_hybrid_bpp_forecast.py — Bags per Passenger (BpP): Seasonality and Hybrid Forecast
- Reads final_bag_pass_count.csv, parses the Date column explicitly as '%m-%y', computes bags_per_passenger as Bag_count ÷ PassengerCount, sets a monthly index, and visualizes BpP seasonality over the year.
- Maps the ratios (bag per passenger) for each month
*IMPORTANT NOTE*
-*Checked in bags and checked in passengers were NOT provided for February so we decided to take January's data and decrease it by 15%. This was done to have a reasonable estimate for February, and 15% was used since in the passenger volume data February usually had around 15% less passengers.*
-*WE ONLY USED THE BAGS PER PASSENGER RATIO (total bags for said month (jan-dec) / total passengers from said month; given by Southwest Data) FROM THIS MODEL, WE DID NOT USE THE FORECASTED REVENUE FROM THIS FOR OUR FINAL MODEL!*

---

## Data Flow

1) Forecast Economic Drivers
- Airfare CPI (cleaned and forecast 24 months) 
- Unemployment (cleaned and forecast 24 months) 
- Fed Funds (cleaned and forecast 24 months) 

2) Build Exogenous Matrices and Targets
- Credit Card Interest: Align historical target with exogenous drivers, mark COVID gaps NaN, and build future exogenous inputs (CPI, Unemployment, Fed Funds) for 24 months.
- Passengers: Merge BTS passenger totals with the three exogenous drivers; standardize exogenous variables; log-transform target.

3) Forecast Models
- Credit Card Interest: SARIMAX with grid-searched parameters; 24-month forecast + 95% CI.
- Passenger Volume: SARIMAX (seasonal differencing only); forecast up to 24 months; back-transform.

4) Revenue Computation
- Merge passenger forecast with monthly BpP ratios (by month number) and credit-card percentage (ScaledPercent/100), then compute monthly revenue using the formula below.

- Revenue = Passengers x Bags
- total_bags = passengers × bag_ratio 
- credit_card_pass = passengers x credit_card_percentage
- free_bags = min(credit_card_pass, total_bags)
- paid_bags = total_bags − free_bags 
- Revenue = paid_bags × fee ($35)

---

## Handling COVID-Affected Periods and Gaps

- All 3 economic drivers, Net Volume of Passengers, Google Trends Data exclude all of 2020, Jan–Apr 2021, and Jan 2022 because of COVID (note - Jan 2022 had a COVID OMICRON Variant surge thus it was removed).
- Credit Card Interest sets those months to NaN in the target and uses missing='drop' in SARIMAX to avoid convergence issues while retaining aligned exogenous information.

---

## Notes and Assumptions

Credit baseline percentage assumption methodology:
- The average American Adult takes about 1.4 plane trips per year. (https://news.gallup.com/poll/388484/air-travel-remains-down-employed-adults-fly-less.aspx#:~:text=Americans%20as%20a%20whole%20took,historical%20range%20for%20both%20groups.)
- A round trip is roughly 2 passenger trips.
- Southwest from various sources carries around 140-170+ million people, so we chose 160 mil. as an estimate
- 160 mil / 2 mil = 80 mil unique passengers
- 30 mil. Total credit cards for all airlines (https://www.airlines.org/news-update/new-airline-industry-analysis-reveals-popularity-of-airline-credit-cards-loyalty-programs-impact-to-local-economies/)
- Southwest's market share is 17% 
- 17 % of 30 = 5.1
- 5.1/80 = 6.3% --> 6% baseline

- Exogenous inputs must be aligned and complete for the forecast horizon; rows with any NaNs are dropped when building future_predictions.csv in the Credit Card Interest pipeline to ensure clean exog matrices.
- Passenger model standardizes exogenous variables and log-transforms the target; forecasts and CI bands are back-transformed to the original scale.
- The final revenue pipeline uses a bag fee of 35.00 in compute time.
- COVID gaps are explicitly handled across models and visuals to reduce bias and improve stability.