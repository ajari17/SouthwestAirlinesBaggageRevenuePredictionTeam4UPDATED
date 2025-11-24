Dataset found on https://fred.stlouisfed.org/series/FEDFUNDS
Downloaded as a CSV file from the website using the ‘Download Data’ option.

### Processing and uses
- Changed FEDFUNDS → new column **fed_fund_rate**
- Changed observation_date → new column **Date**
- Removed: all of 2020, 2021 Jan–Apr, Jan 2022
- Removed 2002–2014
- was used as historical data to forecast future rate in order to add it as exogenous variable to both the net volume of passengers and credit card interest.

###Column meanings
- observation_date --> month of recording
- FEDFUNDS --> rate at said month 