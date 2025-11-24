Airline Fare CPI Dataset

1. Data Source

- Source: U.S. Bureau of Labor Statistics (BLS)  
- Series Title: Airline fares in U.S. city average, all urban consumers, not seasonally adjusted    
- URL: [https://data.bls.gov/timeseries/CUUR0000SETG01?output_view=data](https://data.bls.gov/timeseries/CUUR0000SETG01?output_view=data)

2. Data Access Method

- The data is downloadable as a xlsx  
- In the more formatting options, it is downloadable as a html table or xlsx

3. Breakdown of how the data is structured

- Data Format:  
- Time Coverage: January 2015 - September 2025  
- Frequency: Monthly  
- Column Structure:  
  - `Year`: Integer (2015-2025)  
  - `Jan` through `Dec`: Monthly CPI values (float)  
  - `HALF1`: Semi-annual average for first half (Jan-Jun) - available for some years  
  - `HALF2`: Semi-annual average for second half (Jul-Dec) - available for some years  
- Data Characteristics:  
  - Total observations: 129 monthly data points  
  -  Missing values: October-December 2025 (future dates at time of extraction)  
  - Value range: 197.134 (March 2021) to 344.853 (May 2022)

4. Preprocessing Steps

- Removed HALF1 and HALF2 columns (not needed for analysis)  
- COVID-19 Data Handling  
  - Bad data from the pandemic:  
  -  All of 2020 (12 months)  
  -  January - April 2021 (4 months)  
  -  January 2022 (1 month)  
  - Total removed:17 months of data  
  - Rationale: These periods showed high volatility that would distort our SARIMAX model

---
  
