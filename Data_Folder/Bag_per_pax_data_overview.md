# Southwest Airlines Checked Bag Analysis – 2023

## Overview

1. Dataset was provided to us by **Southwest Airlines**, downloaded via shared one-drive link.  
2. The dataset was very large, but we only needed the following columns:  
   - "total_checked_bag_count" --> represents the total number of checked bags
   - "checked_in_passenger_count" --> represents the total number of checked passengers

3. We were given data for the **1st operational day of every month in 2023**. For analysis, we **summed all the columns**.  

4. **Special handling for February:**  
   - February did not have the required columns.  
   - We inferred values using **net passenger volume**, which is approximately **15% less than January**.  
   - February’s data was taken as **85% of January’s values**.  

5. **Data cleaning:**  
   - NaN values were removed or ignored during processing.  
   - After summing both columns, we computed **bags per passenger** by dividing "total_checked_bag_count" by "checked_in_passenger_count".

---