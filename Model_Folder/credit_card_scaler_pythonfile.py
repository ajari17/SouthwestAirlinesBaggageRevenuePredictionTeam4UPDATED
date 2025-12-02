#%%
import pandas as pd

# Load the CSV
import pandas as pd

df = pd.DataFrame({
    'Month': [
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
        '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10',
        '2021-11', '2021-12', '2022-02', '2022-03', '2022-04', '2022-05',
        '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11',
        '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05',
        '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11',
        '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05',
        '2024-06', '2024-07', '2024-08', '2024-09', '2024-10', '2024-11',
        '2024-12', '2025-01', '2025-02', '2025-03', '2025-04', '2025-05',
        '2025-06', '2025-07', '2025-08', '2025-09', '2025-10', '2025-11'
    ],
    'Score': [
        95, 85, 91, 92, 89, 100, 97, 86, 71, 66, 66, 70,
        70, 52, 62, 57, 56, 62, 54, 52, 46, 45, 43, 39,
        52, 48, 58, 55, 54, 65, 64, 59, 53, 62, 70, 61,
        84, 71, 68, 73, 69, 75, 77, 70, 64, 62, 58, 53,
        99, 87, 69, 65, 69, 80, 70, 72, 66, 65, 59, 57,
        50, 66, 58, 42, 42, 51, 46, 42, 70, 83, 71, 68,
        87, 79, 73, 65, 64, 64, 63, 66, 76, 70, 62, 61,
        65, 59, 53, 53, 48, 53, 52, 61, 64, 58, 51, 49,
        52, 50, 45, 44, 39, 38, 41, 48, 47, 56, 44, 45,
        58, 73, 64, 54, 55, 58
    ]
})


# Compute the midpoint (mean of the Score column)
midpoint = df['Score'].mean()
target_midpoint = 0.06   #
scale_factor = target_midpoint / midpoint

# Apply scaling (convert to percent)
df['ScaledPercent'] = df['Score'] * scale_factor * 100

# Output preview
print(df)

# If you want to save it:
df.to_csv('creditcard_scaled.csv', index=False)

#%%
import pandas as pd

# Load the CSV
df = pd.read_csv('confidence.csv')
df = df.drop(columns=['predicted_value','lower_confidence','upper_confidence'])
df = df.rename(columns={'date':'Month','avg_confidence':"Score"})
# Compute the midpoint (mean of the Score column)
midpoint = df['Score'].mean()

# Set midpoint to 6% → compute scaling factor
target_midpoint = 0.06   #
scale_factor = target_midpoint / midpoint

# Apply scaling (convert to percent)
df['ScaledPercent'] = df['Score'] * scale_factor * 100

# Output preview
print(df)

# If you want to save it:
df.to_csv('creditcard_scaled_forecasted.csv', index=False)

#%% md
# PRINTING THE GRAPH (HISTORICAL + PREDICTED)
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error

# Load the data
data = {
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
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'])

# Split into historical and predicted
cutoff_date = pd.to_datetime('2025-11-01')
df_historical = df[df['Month'] <= cutoff_date]
df_predicted = df[df['Month'] > cutoff_date]

# Create the plot
fig, ax = plt.subplots(figsize=(20, 8))

# Define COVID-affected periods to gray out
covid_periods = [
    (pd.to_datetime('2020-01-01'), pd.to_datetime('2020-12-31')),  # All of 2020
    (pd.to_datetime('2021-01-01'), pd.to_datetime('2021-04-30')),  # Jan-Apr 2021
    (pd.to_datetime('2022-01-01'), pd.to_datetime('2022-01-31'))   # Jan 2022
]

# Add gray shaded regions for COVID periods
y_min, y_max = 0, 12  # Adjust based on your data range
for start_date, end_date in covid_periods:
    ax.axvspan(start_date, end_date, alpha=0.3, color='gray', label='COVID-Affected Period' if start_date == pd.to_datetime('2020-01-01') else '')

# Plot historical data in blue
ax.plot(df_historical['Month'], df_historical['ScaledPercent'],
        marker='o', color='steelblue', label='Historical Credit Card Confidence',
        markersize=4, linewidth=2, zorder=3)

# Plot predicted data in red
ax.plot(df_predicted['Month'], df_predicted['ScaledPercent'],
        marker='s', color='crimson', label='Predicted Credit Card Confidence',
        markersize=4, linewidth=2, linestyle='--', zorder=3)

# Add vertical line at the cutoff
ax.axvline(x=cutoff_date, color='darkgray', linestyle=':', linewidth=2, alpha=0.7,
           label='Historical/Predicted Split (2025-11)', zorder=2)

# Formatting
ax.set_title('Credit Card Confidence Score: Historical vs Predicted\n(Gray regions indicate COVID-affected periods)',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Scaled Confidence Score (%)', fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, zorder=1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
