# Interview and Presentation Guide

Use these answers as a starting point. Put them in your own words and practise aloud. Interviewers can tell when an answer is memorised, so know the numbers and the reasons behind each choice.

**Important:** always say the data is **simulated**. Honesty about the data is a strength. Hiding it is a risk.

---

## A. Resume bullet points (3)

1. **Built an end-to-end retail demand forecasting and replenishment system in Python** on 35,000+ simulated store-item-day records across 4 stores and 12 products, including data validation, exploratory analysis, feature engineering and a Streamlit dashboard.

2. **Reduced forecast error from 51.9% to 36.9% WAPE** against a seasonal-naive baseline, using a hybrid of gradient boosting for regular demand and Croston/SBA for intermittent demand, validated with rolling-origin backtests that exclude stockout days.

3. **Automated stocking decisions with safety stock, reorder points and economic order quantities** at category-specific service levels (90–98%), generating purchase orders worth INR 2.42 lakh for a 28-day horizon.

*Adjust the numbers if you re-run the project with different settings.*

---

## B. LinkedIn project descriptions (3)

**Short (for a project card)**
Built an end-to-end retail demand forecasting and inventory replenishment system in Python. It simulates store sales with stockouts and promotions, forecasts four weeks ahead, cuts forecast error from 52% to 37% against a naive baseline, and recommends reorder points and order quantities. Includes a Streamlit dashboard and tests. Built on synthetic data.

**Medium (for the Featured or Experience section)**
I wanted to understand how retailers avoid running out of stock without overbuying. So I built a complete system that forecasts sales for every store and product, then turns those forecasts into purchasing decisions. It handles real-world messiness: hidden demand on stockout days, intermittent sales for slow items, weekend and festive patterns, and promotions. Accuracy is tested with rolling backtests so the results reflect how the system would be used. The project includes a planner dashboard and CSV purchase-order export. Tools: Python, pandas, scikit-learn, SciPy, Streamlit.

**Long (for a post about lessons learned)**
Three things surprised me building a retail forecasting project. First, the most common mistake is leakage: a feature that quietly uses tomorrow's information makes results look brilliant and then fail. I wrote a test to catch it. Second, stockout days look like low demand, so a model learns the wrong lesson unless you treat them as hidden rather than zero. Third, one model rarely fits every product. Slow-moving items need a different method from daily staples. The project compares methods honestly, and one result surprised me: blending in a simple baseline made accuracy worse, so I removed it. Code and documentation are on GitHub.

---

## C. Ten interview questions and strong answers

**1. What is the objective of your project?**
To forecast daily sales for each store and product four weeks ahead, and to convert those forecasts into reorder points and order quantities. The goal is fewer stockouts and less cash tied up in stock. I built it on simulated data because I do not have access to a retailer's systems, so I designed the simulator to behave like real point-of-sale data.

**2. What real-world problem does it solve?**
Retailers constantly decide how much to order. Order too little and customers leave without buying. Order too much and money sits on shelves, or products expire. This project replaces guesswork with a forecast, a safety buffer sized to the forecast's uncertainty, and a clear reorder rule.

**3. What data did you use?**
A daily sales table covering 4 stores, 12 products and two years. Each row has the quantity sold, price, discount, promotion and festival flags, footfall, end-of-day stock, stock on order and a stockout flag. Separate master tables hold each product's unit cost, shelf life and supplier lead time. The data is synthetic, and I designed it with known patterns, such as weekend peaks and summer drinks, so I could check that the model learns them.

**4. Which model did you use and why?**
I used a gradient-boosting regressor with Poisson loss, trained once across all series. It handles missing values, learns from lag and calendar features, and uses one model for many products. Poisson loss suits unit counts. For intermittent items, where most days have zero sales, I used Croston's method, which is designed for that pattern. I also tested a simple seasonal-naive baseline, which every model must beat. I did not use ARIMA or Prophet for every series because one global model was simpler to maintain across 48 series.

**5. What is the workflow?**
Load and validate the data, clean it, explore it, build a complete daily calendar for each series, create features using only past information, classify each series as regular or intermittent, forecast 28 days ahead, backtest the methods on past periods, calculate safety stock and reorder points from the forecast and its error, and export purchase orders and charts.

**6. How does inventory optimisation work in your project?**
For each product in each store I forecast demand over the supplier's lead time. I add safety stock, calculated as a service-level z-score times the typical forecast error times the square root of the lead time. That sum is the reorder point. When stock on hand plus stock on order falls to the reorder point, I order enough to bring it up to the reorder point plus an economic order quantity. The EOQ balances the fixed cost of placing an order against the cost of holding stock. I also set service levels by category, so perishable dairy gets a higher target than personal care.

**7. Which evaluation metrics did you use and why?**
MAE, which is the average miss in units, and WAPE, which is the total miss as a percentage of total sales, so products of different size can be compared. RMSE shows whether occasional large misses matter. MASE compares the method to a naive forecast, so a value below one means it beats the baseline. I did not use R² as the main measure. On count data with trend and seasonality, R² can look good even when daily forecasts are poor. I evaluated with rolling backtests and skipped stockout days, because true demand is hidden on those days.

**8. What challenges did you face?**
Three main ones. First, stockouts hide true demand. I excluded those days from training and scoring, so the model would not learn that a shelf running empty means low demand. Second, leakage: it is easy to accidentally use information from the day being predicted, so I wrote a test that changes the last day's value and checks that no earlier feature moves. Third, intermittent demand: most daily values are zero, so the usual regression models behave badly. I used Croston's method for those series. I also found that blending in a seasonal-naive baseline made the results worse, which I measured and then removed.

**9. How would you improve this project?**
Forecast at weekly level for intermittent items, which usually improves accuracy a lot. Use shelf life to cap order sizes for perishables. Add price elasticity and weather or local event features. Add anomaly detection to flag unusual sales. Expose the replenishment logic through an API so a purchasing system can call it. Track each model run with MLflow.

**10. How would you explain this project to a non-technical interviewer?**
I built a system that looks at past sales and predicts what a shop will sell over the next month. Then it tells the buyer when to reorder each product and how many to order, with a small safety buffer in case sales spike. That helps the shop avoid empty shelves and avoids tying up cash in too much stock. I tested it on simulated data, and it cut forecasting error by roughly 30% compared with a simple "same as last week" rule.

---

## D. Explaining the project to HR (simple language, 30 seconds)

"I built a project that helps a shop decide what to order and when. It learns from past sales to predict the next four weeks, then sets a reorder point for every product so shelves do not run empty and money is not wasted on too much stock. I used simulated data because I cannot access a real company's system. It's on GitHub with a dashboard anyone can try."

---

## E. Explaining the project to a technical interviewer (2 minutes)

"The pipeline has five stages. First, a simulator generates store-item-day sales with weekly, yearly and festival seasonality, promotions and supplier delays, and it hides demand on stockout days the way real point-of-sale data does. Second, I clean the data and build a complete daily panel, so lags mean what they say. Third, features are strictly backward-looking, and a test enforces that. Fourth, each series is classified by its share of zero-sales days. Regular series go to a Poisson gradient-boosting model that I run recursively for 28 days. Intermittent series go to Croston with the SBA correction. Fifth, I backtest over three rolling origins, score with MAE, RMSE, MASE and WAPE, and use the per-series residual RMSE as sigma in the safety-stock formula. The reorder point is lead-time demand plus safety stock, and the order-up-to level adds an EOQ. One result I'd highlight: blending in seasonal naive hurt accuracy, so the final model uses the ML model alone for regular items."

---

## F. Explaining the business value confidently

- **Lost sales:** every stockout is revenue a customer took elsewhere. A better reorder point reduces that loss directly.
- **Working capital:** every unit held is cash not available for other uses. A right-sized order cuts that.
- **Operational time:** planners spend less time on spreadsheets and more on exceptions, such as the nine items flagged this month.
- **Transparency:** every recommendation can be traced to a forecast, an error figure and a formula, so a planner can challenge it.

Use real numbers from the project: 36.9% WAPE against 51.9% for the naive baseline, 9 items flagged for reorder, INR 242,213 in proposed orders. Say "on simulated data" each time.

---

## G. Questions to ask yourself before the interview

- Can I draw the pipeline from memory?
- Can I reproduce the worked example (ROP 248.67, order 2,135 units) by hand?
- Can I explain why stockout days are excluded?
- Can I say what leakage is and how the test catches it?
- Can I explain why I dropped the blend with seasonal naive?
- Can I name one thing I would do differently with real data?
