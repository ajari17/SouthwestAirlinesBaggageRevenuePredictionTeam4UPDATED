## 1. Historical Passenger Volume Dataset (Training Data Overview)

### Source
- Data obtained from the Bureau of Transportation Statistics (BTS).
- Accessed through the BTS download interface.
- Filter applied prior to download: All major U.S. airports only.
- Link: https://www.transtats.bts.gov/Data_Elements.aspx?Data=4


## 2. Dataset Structure

| Column        | Description                         |
|----------------|-------------------------------------|
| Year          | YYYY                                |
| Month         | MM                                  |
| DOMESTIC      | Number of domestic passengers        |
| INTERNATIONAL | Number of international passengers   |
| TOTAL         | Sum of domestic + international      |


Format:
Year,Month,DOMESTIC,INTERNATIONAL,TOTAL

## 3. Pre-processing Steps
- Removed 2020, 2021 Jan–Apr and Jan 2022.
- Removed yearly-total rows.
- Removed 2002–2014.
- Final dataset contains fewer than 200 rows.
