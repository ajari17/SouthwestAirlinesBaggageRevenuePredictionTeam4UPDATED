## Google Trends Dataset (Southwest Airlines Credit Card)

### Source
- Downloaded directly from Google Trends
- Search term: “Southwest Airlines credit card”

### Structure
| Column | Description |
|--------|-------------|
| Month  | YYYY-MM     |
| Score  | Google Trends interest score |

### Preprocessing
- Combined Year + Month → new column **Month** (YYYY-MM format)
- Renamed “interest” → **Score**
- Removed: all of 2020, 2021 Jan–Apr, Jan 2022
- Removed 2002–2014
