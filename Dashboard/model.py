import numpy as np
import pandas as pd
import streamlit as st

# Try to import matplotlib, fallback if not available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


# ---------------------------------------------------------
# 1. Load real data from final_model.ipynb
# ---------------------------------------------------------
def load_data():
    """
    Load and combine real Southwest baggage prediction data from final_model.ipynb.
    Returns monthly-level DataFrame with historical + predicted data.
    Columns: month, month_dt, passengers, bag_ratio, credit_card_pct, revenue
    """
    # Credit card percentage data (from Cell 1)
    cc_data = {
        'Month': ['2015-01-01', '2015-02-01', '2015-03-01', '2015-04-01', '2015-05-01', '2015-06-01',
                  '2015-07-01', '2015-08-01', '2015-09-01', '2015-10-01', '2015-11-01', '2015-12-01',
                  '2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01',
                  '2016-07-01', '2016-08-01', '2016-09-01', '2016-10-01', '2016-11-01', '2016-12-01',
                  '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01', '2017-06-01',
                  '2017-07-01', '2017-08-01', '2017-09-01', '2017-10-01', '2017-11-01', '2017-12-01',
                  '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01',
                  '2018-07-01', '2018-08-01', '2018-09-01', '2018-10-01', '2018-11-01', '2018-12-01',
                  '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01',
                  '2019-07-01', '2019-08-01', '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01',
                  '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01',
                  '2021-11-01', '2021-12-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01',
                  '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01',
                  '2022-12-01', '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01',
                  '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01',
                  '2023-12-01', '2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01',
                  '2024-06-01', '2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01', '2024-11-01',
                  '2024-12-01', '2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01',
                  '2025-06-01', '2025-07-01', '2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01',
                  '2025-12-01', '2026-01-01', '2026-02-01', '2026-03-01', '2026-04-01', '2026-05-01',
                  '2026-06-01', '2026-07-01', '2026-08-01', '2026-09-01', '2026-10-01', '2026-11-01',
                  '2026-12-01', '2027-01-01', '2027-02-01', '2027-03-01', '2027-04-01', '2027-05-01',
                  '2027-06-01', '2027-07-01', '2027-08-01'],
        'ScaledPercent': [9.12, 8.16, 8.736, 8.832, 8.544, 9.6, 9.312, 8.256, 6.816, 6.336, 6.336, 6.72,
                          6.72, 4.992, 5.952, 5.472, 5.376, 5.952, 5.184, 4.992, 4.416, 4.32, 4.128, 3.744,
                          4.992, 4.608, 5.568, 5.28, 5.184, 6.24, 6.144, 5.664, 5.088, 5.952, 6.72, 5.856,
                          8.064, 6.816, 6.528, 7.008, 6.624, 7.2, 7.392, 6.72, 6.144, 5.952, 5.568, 5.088,
                          9.504, 8.352, 6.624, 6.24, 6.624, 7.68, 6.72, 6.912, 6.336, 6.24, 5.664, 5.472,
                          4.8, 6.336, 5.568, 4.032, 4.032, 4.896, 4.416, 4.032, 6.72, 7.968, 6.816, 6.528,
                          8.352, 7.584, 7.008, 6.24, 6.144, 6.144, 6.048, 6.336, 7.296, 6.72, 5.952, 5.856,
                          6.24, 5.664, 5.088, 5.088, 4.608, 5.088, 4.992, 5.856, 6.144, 5.568, 4.896, 4.704,
                          4.992, 4.8, 4.32, 4.224, 3.744, 3.648, 3.936, 4.608, 4.512, 5.376, 4.224, 4.32,
                          5.568, 7.008, 6.144, 5.184, 5.28, 5.568, 5.348171514272502, 5.642350079796097,
                          5.8080725628276175, 6.002725382515154, 5.086230134392167, 5.020448856730207,
                          6.284910223030227, 6.716272000516867, 6.1947943427926875, 5.581295189112827,
                          5.432122066377441, 5.768729666061314, 5.586168689486723, 5.899583013582311,
                          6.162851117016169, 6.2710254010493145, 5.365276430004657, 5.272299494158216,
                          6.533880968074202, 6.861063256497248, 6.362009380738982]
    }
    cc_df = pd.DataFrame(cc_data)
    cc_df['month'] = pd.to_datetime(cc_df['Month']).dt.strftime('%Y-%m')
    cc_df['credit_card_pct'] = cc_df['ScaledPercent'] / 100  # Convert to decimal

    # Passenger volume data (from Cell 7)
    passenger_df = pd.DataFrame({
        'month': [
            '2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06',
            '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12',
            '2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06',
            '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12',
            '2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
            '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
            '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
            '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12',
            '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06',
            '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12',
            '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06',
            '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
            '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06',
            '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
            '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
            '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12',
            '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06',
            '2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12',
            '2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06',
            '2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12',
            '2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06',
            '2026-07', '2026-08', '2026-09', '2026-10', '2026-11', '2026-12',
            '2027-01', '2027-02', '2027-03', '2027-04', '2027-05', '2027-06',
            '2027-07'
        ],
        'passenger_volume': [
            5770630, 5429968, 7206551, 7045383, 7237810, 7482379,
            7925763, 7370297, 6635536, 7158854, 6894798, 6998894,
            6237276, 5988872, 7454865, 7257890, 7633273, 7857398,
            7878543, 7563794, 7093346, 7459018, 7327211, 7257233,
            6480166, 6029203, 7727929, 7721121, 7826615, 8160322,
            8430946, 7964564, 6853488, 7682273, 7622654, 7501432,
            6849512, 6307883, 8088401, 7817082, 8148699, 8272990,
            8408848, 7901142, 7040204, 7924241, 7780976, 7558688,
            6653632, 6296136, 7975715, 7658132, 7953610, 7912686,
            8065676, 7682974, 6884020, 7704291, 7086288, 7639831,
            2788785, 2733099, 4743556, 5342999, 6207268, 6810118,
            7480735, 6804789, 6117501, 6674775, 6550969, 6426249,
            4902332, 5406496, 6990609, 7083773, 7615679, 7741202,
            8077123, 7731683, 7461429, 8060431, 7371186, 6394055,
            6378477, 6231060, 7766645, 7732912, 8213045, 8318888,
            8629998, 7899014, 7733689, 8543931, 7852507, 7916270,
            6836059, 6790113, 8618239, 8163288, 8688973, 8829882,
            9115661, 7874499, 7379827, 7888966, 7345050, 7769142,
            6205868, 6018193, 7819842, 7530333, 8037753, 8312609,
            8493689, 7497589.390960731, 7027459.500517201, 7500072.303507872,
            6972652.670936624, 7396863.646006387,
            5876352.912594783, 5675999.67285237, 7471432.934878376,
            7153171.312872374, 7656264.599755639, 7911938.261772902,
            8116473.870688262, 7014586.558510848, 6601956.245337194,
            7142450.2118823, 6645573.854437236, 7075500.137648732,
            5567553.365532229, 5391628.5271559665, 7198493.784760071,
            6890472.283210236, 7400300.090629524, 7661266.254453541,
            7871004.379585627
        ]
    })
    passenger_df['month'] = passenger_df['month'].astype(str)
    passenger_df['passengers'] = passenger_df['passenger_volume'].astype(float)
    passenger_df['month_dt'] = pd.to_datetime(passenger_df['month'])
    passenger_df['month_num'] = passenger_df['month_dt'].dt.month

    # BpP_Ratio data (bag ratio by month, from Cell 5)
    bpp_data = {
        'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'BpP_Ratio': [0.8321792310614882, 0.8305381403956789, 0.7310026368579755,
                      0.8099278447362636, 0.7191662733690993, 0.7007624633431085,
                      0.8136658615371184, 0.7644541255281421, 0.5736807119542031,
                      0.638858717515488, 0.7134765513704484, 0.6356559438549426]
    }
    bpp_df = pd.DataFrame(bpp_data)

    # Merge bag ratio and credit-card pct
    passenger_df = passenger_df.merge(bpp_df, left_on='month_num', right_on='Month', how='left')
    passenger_df.rename(columns={'BpP_Ratio': 'bag_ratio'}, inplace=True)
    passenger_df = passenger_df.merge(cc_df[['month', 'credit_card_pct']], on='month', how='left')

    # Revenue calculation (from Cell 7)
    def calculate_revenue(row):
        passengers = row['passengers']
        bag_ratio = row['bag_ratio']
        credit_card_pct = row['credit_card_pct']
        total_bags = passengers * bag_ratio
        cc_pax = passengers * credit_card_pct
        free_bags = min(cc_pax, total_bags)
        paid_bags = total_bags - free_bags
        revenue = paid_bags * 35.0
        return revenue

    passenger_df['revenue'] = passenger_df.apply(calculate_revenue, axis=1)

    # Return final DataFrame with key columns
    final_df = passenger_df[['month', 'month_dt', 'passengers', 'bag_ratio', 'credit_card_pct', 'revenue']].copy()
    
    return final_df


# ---------------------------------------------------------
# 2. Build yearly view from monthly data
# ---------------------------------------------------------
def build_yearly_view(df):
    """
    Aggregate monthly data to yearly level.
    Returns DataFrame with year, total_revenue, total_passengers, avg_bag_ratio, avg_credit_card_pct
    """
    df = df.copy()
    df['year'] = df['month_dt'].dt.year
    
    yearly_df = df.groupby('year', as_index=False).agg({
        'revenue': 'sum',
        'passengers': 'sum',
        'bag_ratio': 'mean',
        'credit_card_pct': 'mean'
    }).rename(columns={
        'revenue': 'total_revenue',
        'passengers': 'total_passengers',
        'bag_ratio': 'avg_bag_ratio',
        'credit_card_pct': 'avg_credit_card_pct'
    })
    
    return yearly_df


# ---------------------------------------------------------
# 3. Get model evaluation metrics (from notebook analysis)
# ---------------------------------------------------------
def get_model_metrics():
    """
    Return model evaluation metrics from the notebook analysis.
    These are based on the credit card percentage model and passenger volume model.
    Actual values from final_model.ipynb output.
    """
    # From Cell 1: Walk-forward validation MAPE for credit card model
    # From Cell 3: In-sample MAPE for passenger volume model
    return {
        'credit_card_mape': 10.91,  # From Cell 1: Walk-Forward Validation MAPE
        'passenger_mape': 15.16,     # From Cell 3: In-sample MAPE
        'credit_card_mae': 0.61,     # From Cell 1: Walk-Forward Validation MAE
    }


# ---------------------------------------------------------
# 4. Calculate Year-over-Year changes
# ---------------------------------------------------------
def calculate_yoy_changes(yearly_df):
    """
    Calculate year-over-year changes for all metrics.
    Returns DataFrame with YoY percentage changes.
    """
    df = yearly_df.copy().sort_values('year')
    df['prev_year'] = df['year'] - 1
    
    # Merge with previous year data
    df = df.merge(
        df[['year', 'total_revenue', 'total_passengers', 'avg_bag_ratio', 'avg_credit_card_pct']],
        left_on='prev_year',
        right_on='year',
        suffixes=('', '_prev'),
        how='left'
    )
    
    # Calculate YoY changes
    df['revenue_yoy_pct'] = ((df['total_revenue'] - df['total_revenue_prev']) / df['total_revenue_prev'] * 100).round(2)
    df['passengers_yoy_pct'] = ((df['total_passengers'] - df['total_passengers_prev']) / df['total_passengers_prev'] * 100).round(2)
    df['bag_ratio_yoy_pct'] = ((df['avg_bag_ratio'] - df['avg_bag_ratio_prev']) / df['avg_bag_ratio_prev'] * 100).round(2)
    df['cc_pct_yoy_pct'] = ((df['avg_credit_card_pct'] - df['avg_credit_card_pct_prev']) / df['avg_credit_card_pct_prev'] * 100).round(2)
    
    return df[['year', 'total_revenue', 'total_passengers', 'avg_bag_ratio', 'avg_credit_card_pct',
               'revenue_yoy_pct', 'passengers_yoy_pct', 'bag_ratio_yoy_pct', 'cc_pct_yoy_pct']]


# ---------------------------------------------------------
# 5. Generate year insights
# ---------------------------------------------------------
def generate_year_insights(yearly_df, year):
    """
    Generate plain English insights for a specific year compared to previous year.
    """
    df = yearly_df.sort_values('year')
    year_data = df[df['year'] == year]
    
    if len(year_data) == 0:
        return "Year not found in data."
    
    if year == df['year'].min():
        return f"{year} marked the first year in our dataset. Southwest generated ${year_data.iloc[0]['total_revenue']:,.0f} in baggage revenue while serving {year_data.iloc[0]['total_passengers']:,.0f} passengers."
    
    prev_year = year - 1
    prev_data = df[df['year'] == prev_year]
    
    if len(prev_data) == 0:
        return f"In {year}, Southwest generated ${year_data.iloc[0]['total_revenue']:,.0f} in baggage revenue with {year_data.iloc[0]['total_passengers']:,.0f} passengers."
    
    curr = year_data.iloc[0]
    prev = prev_data.iloc[0]
    
    revenue_change = ((curr['total_revenue'] - prev['total_revenue']) / prev['total_revenue'] * 100)
    passengers_change = ((curr['total_passengers'] - prev['total_passengers']) / prev['total_passengers'] * 100)
    bag_ratio_change = ((curr['avg_bag_ratio'] - prev['avg_bag_ratio']) / prev['avg_bag_ratio'] * 100)
    cc_pct_change = ((curr['avg_credit_card_pct'] - prev['avg_credit_card_pct']) / prev['avg_credit_card_pct'] * 100)
    
    direction = "grew" if revenue_change > 0 else "declined"
    abs_change = abs(revenue_change)
    
    insights = f"Baggage revenue {direction} by {abs_change:.1f}% compared to {prev_year}, reaching ${curr['total_revenue']:,.0f}."
    
    # Add contributing factors with more natural language
    factors = []
    if abs(passengers_change) > 1:
        if passengers_change > 0:
            factors.append(f"passenger volume grew {abs(passengers_change):.1f}%")
        else:
            factors.append(f"passenger volume dropped {abs(passengers_change):.1f}%")
    
    if abs(bag_ratio_change) > 2:
        if bag_ratio_change > 0:
            factors.append(f"more passengers checked bags (bag ratio up {abs(bag_ratio_change):.1f}%)")
        else:
            factors.append(f"fewer passengers checked bags (bag ratio down {abs(bag_ratio_change):.1f}%)")
    
    if abs(cc_pct_change) > 5:
        if cc_pct_change > 0:
            factors.append(f"more credit card holders (up {abs(cc_pct_change):.1f}%)")
        else:
            factors.append(f"fewer credit card holders (down {abs(cc_pct_change):.1f}%)")
    
    if factors:
        if len(factors) == 1:
            insights += f" This change was primarily due to {factors[0]}."
        elif len(factors) == 2:
            insights += f" Key drivers were {factors[0]} and {factors[1]}."
        else:
            insights += f" Main factors included {', '.join(factors[:-1])}, and {factors[-1]}."
    
    return insights


# ---------------------------------------------------------
# 6. Identify best and worst years by growth
# ---------------------------------------------------------
def get_best_worst_years(yearly_df, n=5):
    """
    Get top N best and worst years by revenue growth.
    Returns tuple of (best_years_df, worst_years_df)
    """
    df = calculate_yoy_changes(yearly_df)
    df = df[df['revenue_yoy_pct'].notna()].copy()
    
    best = df.nlargest(n, 'revenue_yoy_pct')[['year', 'total_revenue', 'revenue_yoy_pct']].copy()
    worst = df.nsmallest(n, 'revenue_yoy_pct')[['year', 'total_revenue', 'revenue_yoy_pct']].copy()
    
    return best, worst


# ---------------------------------------------------------
# 7. Separate historical vs predicted data
# ---------------------------------------------------------
def get_historical_predicted_split(yearly_df, cutoff_year=2024):
    """
    Split data into historical (actual) and predicted (forecast) periods.
    Returns tuple of (historical_df, predicted_df)
    """
    historical = yearly_df[yearly_df['year'] <= cutoff_year].copy()
    predicted = yearly_df[yearly_df['year'] > cutoff_year].copy()
    return historical, predicted


# ---------------------------------------------------------
# 8. Scenario simulation
# ---------------------------------------------------------
def simulate_scenario(base_passengers, base_bag_ratio, base_cc_pct,
                     pct_change_passengers, pct_change_bag_ratio, pct_change_cc_pct):
    """
    Simulate a scenario based on percentage changes from base values.
    Returns predicted revenue.
    """
    passengers = base_passengers * (1 + pct_change_passengers / 100)
    bag_ratio = base_bag_ratio * (1 + pct_change_bag_ratio / 100)
    credit_card_pct = base_cc_pct * (1 + pct_change_cc_pct / 100)
    
    # Clamp values to reasonable ranges
    passengers = max(0, passengers)
    bag_ratio = max(0.3, min(1.0, bag_ratio))
    credit_card_pct = max(0.0, min(0.15, credit_card_pct))
    
    total_bags = passengers * bag_ratio
    cc_pax = passengers * credit_card_pct
    free_bags = min(cc_pax, total_bags)
    paid_bags = total_bags - free_bags
    revenue = paid_bags * 35.0
    
    return {
        'revenue': revenue,
        'passengers': passengers,
        'bag_ratio': bag_ratio,
        'credit_card_pct': credit_card_pct
    }


# ---------------------------------------------------------
# 9. Monthly and Seasonal Analysis Functions
# ---------------------------------------------------------
def get_season(month_num):
    """Map month number to season"""
    if month_num in [12, 1, 2]:
        return 'Winter'
    elif month_num in [3, 4, 5]:
        return 'Spring'
    elif month_num in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def analyze_monthly_patterns(raw_df, selected_year=None):
    """Analyze monthly and seasonal revenue patterns based on actual data"""
    df = raw_df.copy()
    df['year'] = df['month_dt'].dt.year
    df['month_num'] = df['month_dt'].dt.month
    df['month_name'] = df['month_dt'].dt.strftime('%B')
    df['season'] = df['month_num'].apply(get_season)
    
    # Filter by selected year if provided
    if selected_year and selected_year != "All years":
        # Ensure selected_year is an integer for comparison
        year_int = int(selected_year) if isinstance(selected_year, (str, float)) else selected_year
        df_filtered = df[df['year'] == year_int].copy()
        # Debug: ensure we actually have data for this year
        if len(df_filtered) == 0:
            # If no data found, fall back to all data
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    # Monthly averages across all years (for pattern visualization)
    monthly_avg = df.groupby('month_num').agg({
        'revenue': 'mean',
        'passengers': 'mean'
    }).reset_index()
    monthly_avg['month_name'] = pd.to_datetime(monthly_avg['month_num'], format='%m').dt.strftime('%B')
    monthly_avg = monthly_avg.sort_values('month_num')
    
    # Seasonal averages (across all years) - for "All years" view
    seasonal_avg = df.groupby('season').agg({
        'revenue': 'mean',
        'passengers': 'mean'
    }).reset_index()
    # Order seasons properly
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_avg['season_order'] = seasonal_avg['season'].apply(lambda x: season_order.index(x))
    seasonal_avg = seasonal_avg.sort_values('season_order')
    
    # Actual seasonal data for selected year (or all years)
    if selected_year and selected_year != "All years":
        # Ensure selected_year is an integer for comparison (already done above, but be safe)
        year_int = int(selected_year) if isinstance(selected_year, (str, float)) else selected_year
        # Make sure df_filtered is actually filtered correctly
        df_filtered_check = df[df['year'] == year_int].copy()
        if len(df_filtered_check) > 0:
            seasonal_actual = df_filtered_check.groupby('season').agg({
                'revenue': 'sum',  # Sum for the year
                'passengers': 'sum'
            }).reset_index()
        else:
            # Fallback to df_filtered if check fails
            seasonal_actual = df_filtered.groupby('season').agg({
                'revenue': 'sum',
                'passengers': 'sum'
            }).reset_index()
        seasonal_actual['season_order'] = seasonal_actual['season'].apply(lambda x: season_order.index(x))
        seasonal_actual = seasonal_actual.sort_values('season_order')
    else:
        # For "All years", calculate total revenue per season across all years
        seasonal_actual = df.groupby('season').agg({
            'revenue': 'sum',  # Total across all years
            'passengers': 'sum'
        }).reset_index()
        seasonal_actual['season_order'] = seasonal_actual['season'].apply(lambda x: season_order.index(x))
        seasonal_actual = seasonal_actual.sort_values('season_order')
    
    # Best/worst months based on ACTUAL data (not averages)
    if selected_year and selected_year != "All years":
        # Use actual monthly data for the selected year
        monthly_actual = df_filtered.groupby('month_num').agg({
            'revenue': 'sum',  # Sum if multiple months (shouldn't happen, but safe)
            'passengers': 'sum'
        }).reset_index()
        if len(monthly_actual) > 0:
            best_month_row = monthly_actual.loc[monthly_actual['revenue'].idxmax()]
            worst_month_row = monthly_actual.loc[monthly_actual['revenue'].idxmin()]
            best_month = {
                'month_num': int(best_month_row['month_num']),
                'month_name': pd.to_datetime(int(best_month_row['month_num']), format='%m').strftime('%B'),
                'revenue': float(best_month_row['revenue']),
                'passengers': float(best_month_row['passengers'])
            }
            worst_month = {
                'month_num': int(worst_month_row['month_num']),
                'month_name': pd.to_datetime(int(worst_month_row['month_num']), format='%m').strftime('%B'),
                'revenue': float(worst_month_row['revenue']),
                'passengers': float(worst_month_row['passengers'])
            }
        else:
            # Fallback to averages if no data
            best_month_row = monthly_avg.loc[monthly_avg['revenue'].idxmax()]
            worst_month_row = monthly_avg.loc[monthly_avg['revenue'].idxmin()]
            best_month = {
                'month_num': int(best_month_row['month_num']),
                'month_name': best_month_row['month_name'],
                'revenue': float(best_month_row['revenue']),
                'passengers': float(best_month_row['passengers'])
            }
            worst_month = {
                'month_num': int(worst_month_row['month_num']),
                'month_name': worst_month_row['month_name'],
                'revenue': float(worst_month_row['revenue']),
                'passengers': float(worst_month_row['passengers'])
            }
    else:
        # For "All years", find the best/worst month across all actual data points
        # Get the month with highest and lowest revenue across all years
        best_month_row = df.loc[df['revenue'].idxmax()]
        worst_month_row = df.loc[df['revenue'].idxmin()]
        best_month = {
            'month_num': int(best_month_row['month_num']),
            'month_name': best_month_row['month_name'],
            'revenue': float(best_month_row['revenue']),
            'passengers': float(best_month_row['passengers'])
        }
        worst_month = {
            'month_num': int(worst_month_row['month_num']),
            'month_name': worst_month_row['month_name'],
            'revenue': float(worst_month_row['revenue']),
            'passengers': float(worst_month_row['passengers'])
        }
    
    # Best/worst seasons based on actual data
    if selected_year and selected_year != "All years":
        seasonal_actual = df_filtered.groupby('season').agg({
            'revenue': 'sum',
            'passengers': 'sum'
        }).reset_index()
        seasonal_actual['season_order'] = seasonal_actual['season'].apply(lambda x: season_order.index(x))
        seasonal_actual = seasonal_actual.sort_values('season_order')
        if len(seasonal_actual) > 0:
            best_season_row = seasonal_actual.loc[seasonal_actual['revenue'].idxmax()]
            worst_season_row = seasonal_actual.loc[seasonal_actual['revenue'].idxmin()]
            best_season = {
                'season': best_season_row['season'],
                'revenue': float(best_season_row['revenue']),
                'passengers': float(best_season_row['passengers'])
            }
            worst_season = {
                'season': worst_season_row['season'],
                'revenue': float(worst_season_row['revenue']),
                'passengers': float(worst_season_row['passengers'])
            }
        else:
            best_season_row = seasonal_avg.loc[seasonal_avg['revenue'].idxmax()]
            worst_season_row = seasonal_avg.loc[seasonal_avg['revenue'].idxmin()]
            best_season = {
                'season': best_season_row['season'],
                'revenue': float(best_season_row['revenue']),
                'passengers': float(best_season_row['passengers'])
            }
            worst_season = {
                'season': worst_season_row['season'],
                'revenue': float(worst_season_row['revenue']),
                'passengers': float(worst_season_row['passengers'])
            }
    else:
        # For "All years", use averages
        best_season_row = seasonal_avg.loc[seasonal_avg['revenue'].idxmax()]
        worst_season_row = seasonal_avg.loc[seasonal_avg['revenue'].idxmin()]
        best_season = {
            'season': best_season_row['season'],
            'revenue': float(best_season_row['revenue']),
            'passengers': float(best_season_row['passengers'])
        }
        worst_season = {
            'season': worst_season_row['season'],
            'revenue': float(worst_season_row['revenue']),
            'passengers': float(worst_season_row['passengers'])
        }
    
    return {
        'monthly_avg': monthly_avg,  # Averages across all years (for pattern visualization)
        'seasonal_avg': seasonal_avg,  # Averages across all years
        'seasonal_actual': seasonal_actual,  # Actual seasonal data for selected year or all years
        'best_month': best_month,
        'worst_month': worst_month,
        'best_season': best_season,
        'worst_season': worst_season,
        'monthly_data': df,
        'filtered_data': df_filtered
    }


# ---------------------------------------------------------
# 10. Streamlit UI - Polished Competition Dashboard
# ---------------------------------------------------------
st.set_page_config(
    page_title="Southwest Baggage Revenue Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Inter', 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .insight-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #262730;
        font-size: 1rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .insight-card strong {
        color: #1f77b4;
        font-weight: 600;
    }
    
    .model-summary-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        color: #262730;
        font-size: 0.95rem;
        line-height: 1.7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .model-summary-box h4 {
        color: #1f77b4;
        font-weight: 600;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .model-summary-box ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .model-summary-box li {
        margin: 0.4rem 0;
        color: #262730;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Prevent text truncation in metric components */
    [data-testid="stMetricValue"] {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 1.5rem !important;
    }
    
    [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">✈️ Southwest Baggage Revenue Dashboard</h1>', unsafe_allow_html=True)
st.caption("**Aggie Data Science Club × Southwest Airlines - Team 4** | Forecasting baggage revenue using real data and advanced time series models")

# Load data
with st.spinner("Loading real Southwest baggage prediction data..."):
    raw_df = load_data()
    yearly_df = build_yearly_view(raw_df)
    yearly_df_with_yoy = calculate_yoy_changes(yearly_df)
    historical_df, predicted_df = get_historical_predicted_split(yearly_df, cutoff_year=2024)
    model_metrics = get_model_metrics()
    best_years, worst_years = get_best_worst_years(yearly_df, n=5)

# Year selector in sidebar
st.sidebar.header("📅 Navigation")
year_options = ["All years"] + sorted(yearly_df['year'].unique().tolist(), reverse=True)
selected_year = st.sidebar.selectbox("Select Year", options=year_options, index=0)

# View toggle: Yearly vs Monthly
view_mode = st.sidebar.radio("View Mode", ["Yearly", "Monthly"], index=0)

# Get data for selected year
latest_year = yearly_df.iloc[-1]
if selected_year == "All years":
    display_df = yearly_df.copy()
    year_data = latest_year
else:
    display_df = yearly_df[yearly_df['year'] == selected_year].copy()
    year_data = display_df.iloc[0] if len(display_df) > 0 else latest_year

# Year Insights Card - only in Yearly view
if view_mode == "Yearly" and selected_year != "All years" and selected_year != yearly_df['year'].min():
    insights = generate_year_insights(yearly_df, selected_year)
    st.markdown(f'<div class="insight-card"><strong>📊 {selected_year} Year in Review</strong><br><br>{insights}</div>', unsafe_allow_html=True)

# Main KPI Row with YoY Comparisons - only in Yearly view
if view_mode == "Yearly":
    st.markdown("## 📈 Key Performance Indicators")
col1, col2, col3 = st.columns(3)

# Get YoY data
yoy_data = yearly_df_with_yoy[yearly_df_with_yoy['year'] == selected_year] if selected_year != "All years" else None
prev_year_data = None
if selected_year != "All years" and selected_year != yearly_df['year'].min():
    prev_year = selected_year - 1
    prev_year_data = yearly_df[yearly_df['year'] == prev_year]
    if len(prev_year_data) > 0:
        prev_year_data = prev_year_data.iloc[0]

with col1:
    revenue = year_data['total_revenue']
    if prev_year_data is not None:
        revenue_yoy = ((revenue - prev_year_data['total_revenue']) / prev_year_data['total_revenue'] * 100)
        # Use "normal" so negative values show red and positive show green
        delta_color = "normal"
        st.metric(
            "Baggage Revenue",
            f"${revenue:,.0f}",
            delta=f"{revenue_yoy:+.1f}% vs {selected_year-1}",
            delta_color=delta_color
        )
    else:
        st.metric("Baggage Revenue", f"${revenue:,.0f}")

with col2:
    passengers = year_data['total_passengers']
    if prev_year_data is not None:
        passengers_yoy = ((passengers - prev_year_data['total_passengers']) / prev_year_data['total_passengers'] * 100)
        # Use "normal" so negative values show red and positive show green
        delta_color = "normal"
        st.metric(
            "Total Passengers",
            f"{passengers:,.0f}",
            delta=f"{passengers_yoy:+.1f}% vs {selected_year-1}",
            delta_color=delta_color
        )
    else:
        st.metric("Total Passengers", f"{passengers:,.0f}")

with col3:
    cc_pct = year_data['avg_credit_card_pct']
    if prev_year_data is not None:
        cc_pct_yoy = ((cc_pct - prev_year_data['avg_credit_card_pct']) / prev_year_data['avg_credit_card_pct'] * 100)
        # Use "normal" so negative values show red and positive show green
        delta_color = "normal"
        st.metric(
            "Credit Card %",
            f"{cc_pct*100:.2f}%",
            delta=f"{cc_pct_yoy:+.1f}% vs {selected_year-1}",
            delta_color=delta_color
        )
    else:
        st.metric("Credit Card %", f"{cc_pct*100:.2f}%")

    st.markdown("---")

# Main Chart: Historical vs Predicted (only show in Yearly view)
chart_col1, chart_col2 = st.columns([3, 1])

with chart_col1:
    if MATPLOTLIB_AVAILABLE:
        # Create a matplotlib figure with different colors for historical vs predicted
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data as one continuous line (combining pre-2020 and post-2020)
        if len(historical_df) > 0:
            # Sort by year to ensure continuous plotting
            hist_sorted = historical_df.sort_values('year')
            # Plot all historical data as one continuous line
            ax.plot(hist_sorted['year'], hist_sorted['total_revenue'], 
                    color='#3498db', linewidth=3, marker='o', markersize=6, 
                    zorder=2, label='Historical')
        
        # Plot predicted data in red, connecting from last historical point
        if len(predicted_df) > 0:
            if len(historical_df) > 0:
                # Get the last historical point to connect seamlessly
                cutoff_year = historical_df['year'].max()
                last_hist_revenue = historical_df[historical_df['year'] == cutoff_year]['total_revenue'].iloc[0]
                
                # Create connecting data
                pred_years = [cutoff_year] + predicted_df['year'].tolist()
                pred_revenues = [last_hist_revenue] + predicted_df['total_revenue'].tolist()
                
                ax.plot(pred_years, pred_revenues, 
                        color='#e74c3c', linewidth=3, linestyle='--', marker='s', markersize=6,
                        zorder=2)
            else:
                ax.plot(predicted_df['year'], predicted_df['total_revenue'], 
                        color='#e74c3c', linewidth=3, linestyle='--', marker='s', markersize=6,
                        zorder=2)
        
        # Set labels
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Revenue (USD)', fontsize=12, fontweight='bold')
        ax.set_title('')  # Remove any title/caption
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add vertical line at cutoff if both exist (without text caption)
        if len(historical_df) > 0 and len(predicted_df) > 0:
            cutoff_year = historical_df['year'].max()
            ax.axvline(x=cutoff_year + 0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)
        
        # Explicitly remove/hide legend - ensure no legend is displayed
        # Remove any existing legend and prevent auto-legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        # Disable automatic legend creation
        ax.legend_ = None
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        # Fallback: Use Streamlit's native chart
        # Combine all data into a single series to avoid legend labels
        all_years = sorted(set(
            list(historical_df['year'].values if len(historical_df) > 0 else []) +
            list(predicted_df['year'].values if len(predicted_df) > 0 else [])
        ))
        
        # Create a single combined series to avoid separate legend items
        chart_data = pd.DataFrame({'year': all_years})
        chart_data = chart_data.set_index('year')
        
        # Combine historical and predicted into one column to avoid legend
        combined_revenue = []
        for year in all_years:
            if len(historical_df) > 0 and year in historical_df['year'].values:
                combined_revenue.append(historical_df[historical_df['year'] == year]['total_revenue'].iloc[0])
            elif len(predicted_df) > 0 and year in predicted_df['year'].values:
                combined_revenue.append(predicted_df[predicted_df['year'] == year]['total_revenue'].iloc[0])
            else:
                combined_revenue.append(None)
        
        chart_data['Revenue'] = combined_revenue
        
        # Display chart (Streamlit will still show a legend, but with just one item)
        st.line_chart(chart_data)

with chart_col2:
    st.markdown("### Quick Stats")
    if len(historical_df) > 0:
        st.metric("Historical Avg", f"${historical_df['total_revenue'].mean():,.0f}")
    if len(predicted_df) > 0:
        st.metric("Predicted Avg", f"${predicted_df['total_revenue'].mean():,.0f}")

    st.markdown("---")

# Monthly and Seasonal Analysis Section
if view_mode == "Monthly":
    st.markdown("## 📅 Monthly & Seasonal Analysis")
    
    # Analyze patterns based on actual data from notebook
    monthly_analysis = analyze_monthly_patterns(raw_df, selected_year)
    
    # Key insights
    st.markdown("### 🎯 Key Insights")
    insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
    
    with insight_col1:
        best_month_name = monthly_analysis['best_month'].get('month_name', 'N/A')
        best_month_rev = monthly_analysis['best_month'].get('revenue', 0)
        st.metric(
            "Best Month",
            best_month_name,
            help=f"Actual revenue: ${best_month_rev:,.0f}"
        )
    
    with insight_col2:
        worst_month_name = monthly_analysis['worst_month'].get('month_name', 'N/A')
        worst_month_rev = monthly_analysis['worst_month'].get('revenue', 0)
        st.metric(
            "Worst Month",
            worst_month_name,
            help=f"Actual revenue: ${worst_month_rev:,.0f}"
        )
    
    with insight_col3:
        best_season_name = monthly_analysis['best_season'].get('season', 'N/A')
        best_season_rev = monthly_analysis['best_season'].get('revenue', 0)
        st.metric(
            "Best Season",
            best_season_name,
            help=f"Revenue: ${best_season_rev:,.0f}"
        )
    
    with insight_col4:
        worst_season_name = monthly_analysis['worst_season'].get('season', 'N/A')
        worst_season_rev = monthly_analysis['worst_season'].get('revenue', 0)
        st.metric(
            "Worst Season",
            worst_season_name,
            help=f"Revenue: ${worst_season_rev:,.0f}"
        )
    
    # Monthly revenue pattern - show actual data for selected year or averages for all years
    if selected_year != "All years":
        st.markdown(f"### 📊 Monthly Revenue Pattern for {selected_year}")
        st.markdown(f"Shows actual monthly revenue data for **{selected_year}** from the model.")
        # Get actual monthly data for the selected year - use the filtered data directly
        # The filtered_data already has the correct year filtered
        monthly_actual_df = monthly_analysis['filtered_data'].copy()
        
        # Group by month_num to get monthly totals (in case there are any duplicates)
        monthly_actual = monthly_actual_df.groupby('month_num', as_index=False).agg({
            'revenue': 'sum',
            'passengers': 'sum'
        })
        # Add month_name column
        monthly_actual['month_name'] = pd.to_datetime(monthly_actual['month_num'], format='%m').dt.strftime('%B')
        monthly_actual = monthly_actual.sort_values('month_num')
        monthly_display = monthly_actual
        chart_title = f'Monthly Revenue for {selected_year} (Actual Data)'
    else:
        st.markdown("### 📊 Average Monthly Revenue Pattern")
        st.markdown("Shows the average revenue for each month across all years, revealing seasonal trends.")
        monthly_display = monthly_analysis['monthly_avg']
        chart_title = 'Average Monthly Revenue Across All Years'
    
    monthly_chart_col1, monthly_chart_col2 = st.columns([2, 1])
    
    with monthly_chart_col1:
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 6))
            # Ensure months are in correct order (January to December)
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            # Reorder the data to match month_order
            monthly_display_ordered = monthly_display.set_index('month_name').reindex(month_order).reset_index()
            monthly_display_ordered = monthly_display_ordered.dropna()  # Remove any missing months
            
            ax.bar(monthly_display_ordered['month_name'], 
                   monthly_display_ordered['revenue'],
                   color='#3498db', alpha=0.7, edgecolor='#2980b9', linewidth=1.5)
            ax.set_xlabel('Month', fontsize=12, fontweight='bold')
            ax.set_ylabel('Revenue (USD)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            # For fallback, ensure proper ordering
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            chart_data = monthly_display[['month_name', 'revenue']].set_index('month_name')
            chart_data = chart_data.reindex([m for m in month_order if m in chart_data.index])
            st.bar_chart(chart_data)
    
    with monthly_chart_col2:
        st.markdown("#### Monthly Rankings")
        # Sort by revenue descending for rankings, but display in order
        monthly_ranked = monthly_display.sort_values('revenue', ascending=False).reset_index(drop=True)
        for rank_num, (idx, row) in enumerate(monthly_ranked.iterrows(), start=1):
            if rank_num == 1:
                rank_emoji = "🥇"
            elif rank_num == 2:
                rank_emoji = "🥈"
            elif rank_num == 3:
                rank_emoji = "🥉"
            else:
                rank_emoji = f"{rank_num}."
            st.markdown(f"{rank_emoji} **{row['month_name']}**: ${row['revenue']:,.0f}")
    
    # Seasonal breakdown - use actual data based on selected year
    st.markdown("### 🍂 Seasonal Revenue Breakdown")
    if selected_year != "All years":
        st.markdown(f"Revenue patterns by season for **{selected_year}** (Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Fall: Sep-Nov)")
        seasonal_display = monthly_analysis['seasonal_actual']
        chart_title = f'Revenue by Season for {selected_year}'
        ylabel = 'Revenue (USD)'
    else:
        st.markdown("Total revenue by season across all years (Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Fall: Sep-Nov)")
        seasonal_display = monthly_analysis['seasonal_actual']
        chart_title = 'Total Revenue by Season (All Years)'
        ylabel = 'Total Revenue (USD)'
    
    season_col1, season_col2 = st.columns([2, 1])
    
    with season_col1:
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
            bars = ax.bar(seasonal_display['season'], 
                         seasonal_display['revenue'],
                         color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Season', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}',
                       ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            chart_data = seasonal_display[['season', 'revenue']].set_index('season')
            st.bar_chart(chart_data)
    
    with season_col2:
        st.markdown("#### Season Details")
        for idx, row in seasonal_display.iterrows():
            pct_of_total = (row['revenue'] / seasonal_display['revenue'].sum()) * 100
            st.markdown(f"**{row['season']}**")
            st.markdown(f"${row['revenue']:,.0f}")
            st.markdown(f"*{pct_of_total:.1f}% of total*")
            st.markdown("---")
    
    # Monthly trend over time
    st.markdown("### 📈 Monthly Revenue Over Time")
    if selected_year != "All years":
        st.markdown(f"Showing actual monthly revenue data for **{selected_year}** from the model.")
    else:
        st.markdown("View monthly revenue trends across all years to identify patterns and anomalies.")
    
    # Use filtered data from analysis (already filtered by selected_year)
    monthly_filtered = monthly_analysis['filtered_data'].copy()
    if selected_year != "All years":
        chart_title = f"Monthly Revenue for {selected_year} (Actual Data)"
    else:
        chart_title = "Monthly Revenue Over All Years"
    
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot each year as a line
        if selected_year == "All years":
            for year in sorted(monthly_filtered['year'].unique()):
                year_data = monthly_filtered[monthly_filtered['year'] == year].sort_values('month_num')
                ax.plot(year_data['month_name'], year_data['revenue'], 
                       marker='o', label=str(int(year)), alpha=0.6, linewidth=2)
            ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        else:
            year_data = monthly_filtered.sort_values('month_num')
            ax.plot(year_data['month_name'], year_data['revenue'], 
                   marker='o', linewidth=3, color='#3498db', markersize=8)
        
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Revenue (USD)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        if selected_year == "All years":
            # Show average by month
            monthly_summary = monthly_filtered.groupby('month_name')['revenue'].mean().reset_index()
            chart_data = monthly_summary.set_index('month_name')
        else:
            year_data = monthly_filtered.sort_values('month_num')
            chart_data = year_data[['month_name', 'revenue']].set_index('month_name')
        st.line_chart(chart_data)
    
    st.markdown("---")

st.markdown("---")

# Year-over-Year Comparison Panel (when specific year selected) - only in Yearly view
if view_mode == "Yearly" and selected_year != "All years" and prev_year_data is not None:
    st.markdown("## 🔄 Year-over-Year Comparison")
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown(f"### {selected_year - 1} (Previous Year)")
        st.metric("Revenue", f"${prev_year_data['total_revenue']:,.0f}")
        st.metric("Passengers", f"{prev_year_data['total_passengers']:,.0f}")
        st.metric("Bag Ratio", f"{prev_year_data['avg_bag_ratio']:.3f}")
        st.metric("Credit Card %", f"{prev_year_data['avg_credit_card_pct']*100:.2f}%")
    
    with comp_col2:
        st.markdown(f"### {selected_year} (Selected Year)")
        st.metric("Revenue", f"${year_data['total_revenue']:,.0f}")
        st.metric("Passengers", f"{year_data['total_passengers']:,.0f}")
        st.metric("Bag Ratio", f"{year_data['avg_bag_ratio']:.3f}")
        st.metric("Credit Card %", f"{year_data['avg_credit_card_pct']*100:.2f}%")
    
    st.markdown("---")

# Breakdown Panel: What's Driving Changes - only in Yearly view
if view_mode == "Yearly" and selected_year != "All years" and prev_year_data is not None:
    st.markdown("## 🔍 What's Driving the Change?")
    
    revenue_change = ((year_data['total_revenue'] - prev_year_data['total_revenue']) / prev_year_data['total_revenue'] * 100)
    passengers_change = ((year_data['total_passengers'] - prev_year_data['total_passengers']) / prev_year_data['total_passengers'] * 100)
    bag_ratio_change = ((year_data['avg_bag_ratio'] - prev_year_data['avg_bag_ratio']) / prev_year_data['avg_bag_ratio'] * 100)
    cc_pct_change = ((year_data['avg_credit_card_pct'] - prev_year_data['avg_credit_card_pct']) / prev_year_data['avg_credit_card_pct'] * 100)
    
    breakdown_df = pd.DataFrame({
        'Factor': ['Passenger Volume', 'Bag Ratio', 'Credit Card %'],
        'Change %': [passengers_change, bag_ratio_change, cc_pct_change],
        'Impact': ['High' if abs(x) > 5 else 'Medium' if abs(x) > 2 else 'Low' for x in [passengers_change, bag_ratio_change, cc_pct_change]]
    })
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(breakdown_df.set_index('Factor')['Change %'])
    with col2:
        st.dataframe(breakdown_df, use_container_width=True)
    
    st.markdown("---")

# Advanced Scenario Simulator with Pinning
st.markdown("## 🎯 Scenario Simulator")
st.markdown("Adjust key drivers to see how they impact predicted yearly baggage revenue. Pin scenarios to compare side-by-side.")

sim_col1, sim_col2 = st.columns(2)

# Get base values from latest year
base_passengers = int(latest_year['total_passengers'])
base_bag_ratio = float(latest_year['avg_bag_ratio'])
base_cc_pct = float(latest_year['avg_credit_card_pct'])

with sim_col1:
    st.markdown("### Scenario A")
    pct_passengers_a = st.slider(
        "% Change in Passengers",
        min_value=-30.0,
        max_value=30.0,
        value=0.0,
        step=1.0,
        key="passengers_a"
    )
    pct_bag_ratio_a = st.slider(
        "% Change in Bag Ratio",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=1.0,
        key="bag_ratio_a"
    )
    pct_cc_pct_a = st.slider(
        "% Change in Credit Card %",
        min_value=-30.0,
        max_value=30.0,
        value=0.0,
        step=1.0,
        key="cc_pct_a"
    )
    
    scenario_a = simulate_scenario(
        base_passengers, base_bag_ratio, base_cc_pct,
        pct_passengers_a, pct_bag_ratio_a, pct_cc_pct_a
    )
    
    st.success(f"**Scenario A Revenue:** ${scenario_a['revenue']:,.0f}")
    
    pin_a = st.button("📌 Pin Scenario A", key="pin_a")

with sim_col2:
    st.markdown("### Scenario B")
    pct_passengers_b = st.slider(
        "% Change in Passengers",
        min_value=-30.0,
        max_value=30.0,
        value=5.0,
        step=1.0,
        key="passengers_b"
    )
    pct_bag_ratio_b = st.slider(
        "% Change in Bag Ratio",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=1.0,
        key="bag_ratio_b"
    )
    pct_cc_pct_b = st.slider(
        "% Change in Credit Card %",
        min_value=-30.0,
        max_value=30.0,
        value=0.0,
        step=1.0,
        key="cc_pct_b"
    )
    
    scenario_b = simulate_scenario(
        base_passengers, base_bag_ratio, base_cc_pct,
        pct_passengers_b, pct_bag_ratio_b, pct_cc_pct_b
    )
    
    st.success(f"**Scenario B Revenue:** ${scenario_b['revenue']:,.0f}")
    
    pin_b = st.button("📌 Pin Scenario B", key="pin_b")

# Pinned scenarios comparison
if 'pinned_a' not in st.session_state:
    st.session_state.pinned_a = None
if 'pinned_b' not in st.session_state:
    st.session_state.pinned_b = None

if pin_a:
    st.session_state.pinned_a = scenario_a.copy()
    st.session_state.pinned_a['name'] = 'Scenario A (Pinned)'
if pin_b:
    st.session_state.pinned_b = scenario_b.copy()
    st.session_state.pinned_b['name'] = 'Scenario B (Pinned)'

if st.session_state.pinned_a or st.session_state.pinned_b:
    st.markdown("### 📊 Pinned Scenarios Comparison")
    comp_scenarios = []
    if st.session_state.pinned_a:
        comp_scenarios.append(st.session_state.pinned_a)
    if st.session_state.pinned_b:
        comp_scenarios.append(st.session_state.pinned_b)
    
    if len(comp_scenarios) > 0:
        comp_df = pd.DataFrame(comp_scenarios)
        comp_cols = st.columns(len(comp_scenarios))
        
        for idx, scenario in enumerate(comp_scenarios):
            with comp_cols[idx]:
                st.markdown(f"#### {scenario.get('name', f'Scenario {chr(65+idx)}')}")
                st.metric("Revenue", f"${scenario['revenue']:,.0f}")
                st.metric("Passengers", f"{scenario['passengers']:,.0f}")
                st.metric("Bag Ratio", f"{scenario['bag_ratio']:.3f}")
                st.metric("Credit Card %", f"{scenario['credit_card_pct']*100:.2f}%")
        
        # Comparison chart
        if len(comp_scenarios) == 2:
            diff = comp_scenarios[1]['revenue'] - comp_scenarios[0]['revenue']
            diff_pct = (diff / comp_scenarios[0]['revenue'] * 100)
            st.metric(
                "Revenue Difference",
                f"${diff:,.0f}",
                delta=f"{diff_pct:+.1f}%"
        )

st.markdown("---")

# Best and Worst Years
st.markdown("## 🏆 Top Performers & Challenges")
best_col, worst_col = st.columns(2)

with best_col:
    st.markdown("### 🟢 Top 5 Years by Growth")
    if len(best_years) > 0:
        best_chart_df = best_years[['year', 'revenue_yoy_pct']].set_index('year')
        st.bar_chart(best_chart_df)
        with st.expander("View Details"):
            st.dataframe(best_years[['year', 'total_revenue', 'revenue_yoy_pct']], use_container_width=True)

with worst_col:
    st.markdown("### 🔴 Bottom 5 Years by Growth")
    if len(worst_years) > 0:
        worst_chart_df = worst_years[['year', 'revenue_yoy_pct']].set_index('year')
        st.bar_chart(worst_chart_df)
        with st.expander("View Details"):
            st.dataframe(worst_years[['year', 'total_revenue', 'revenue_yoy_pct']], use_container_width=True)

st.markdown("---")

# Model Performance Section
st.markdown("## 🎯 Model Performance")
perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.markdown("### Credit Card Percentage Model")
    st.metric("MAPE", f"{model_metrics['credit_card_mape']:.2f}%", 
              help="Mean Absolute Percentage Error - lower is better. 10.91% means predictions are off by ~11% on average.")
    st.metric("MAE", f"{model_metrics['credit_card_mae']:.2f}",
              help="Mean Absolute Error - average prediction error in percentage points.")
    st.info("Our credit card confidence model achieves about 11% average prediction error, which is solid for business forecasting. This means we can reliably predict how many passengers will use credit cards for free baggage benefits.")

with perf_col2:
    st.markdown("### Passenger Volume Model")
    st.metric("MAPE", f"{model_metrics['passenger_mape']:.2f}%",
              help="Mean Absolute Percentage Error - lower is better. 15.16% means predictions are off by ~15% on average.")
    st.info("The passenger volume model has around 15% average error. It factors in economic conditions like airfare costs, interest rates, and unemployment levels to forecast how many passengers Southwest will carry each year.")

st.markdown("---")

# Model Summary
st.markdown("## 📋 How This Model Works")
st.markdown("""
<div class="model-summary-box">
<h4>Model Architecture & Methodology</h4>
<p><strong>What we're predicting:</strong> Yearly baggage revenue in USD</p>

<p><strong>Where the data comes from:</strong></p>
<ul>
  <li>Passenger volume forecasts using SARIMAX time series modeling, incorporating economic indicators like Airfare CPI, Federal Funds Rate, and Unemployment Rate</li>
  <li>Credit card confidence percentage predictions from our ScaledPercent model</li>
  <li>Bag-to-passenger ratios that follow monthly seasonal patterns</li>
</ul>

<p><strong>How revenue is calculated:</strong></p>
<ul>
  <li>Total bags = Passengers × Bag Ratio</li>
  <li>Free bags = minimum of (Passengers × Credit Card %) and Total Bags</li>
  <li>Paid bags = Total Bags - Free Bags</li>
  <li>Revenue = Paid Bags × $35 per bag fee</li>
</ul>

<p style="margin-top: 1rem; font-style: italic; color: #666;">This model combines multiple forecasting approaches to provide accurate yearly baggage revenue predictions for Southwest Airlines.</p>
</div>
""", unsafe_allow_html=True)

# Raw data expander
with st.expander("📊 View Raw Data"):
    st.dataframe(yearly_df, use_container_width=True)
