# Empirical-Exercise-in-Value-at-Risk
Assume that you are the risk manager of a retirement fund. The fund has
€500 million in cash to invest in fixed-income markets. You decide to allocate this
amount equally across 50 bond ETFs of your choice. Your objective is to estimate
the market risk of this portfolio using alternative Value-at-Risk methods and to
determine which method provides the best performance.
Data Requirement
Choose 50 bond ETFs and form an equally weighted portfolio. Download two
years of daily historical data for all selected ETFs. In order to compute returns, use
adjusted closing prices whenever available. Assume that the regulator requires the
computation of the 90% VaR.

Question A
Estimate the next-day VaR of the portfolio at the 90% confidence level using the
following methods:
1. Historical simulation.
2. Parametric VaR under the assumption of normally distributed returns.
3. Parametric VaR under the assumption of Student-t distributed returns.
You must write the corresponding Python code for all methods and include it
in an appendix to your essay.

Question B
Use backtesting to evaluate the forecasting performance of the three VaR methods in Question A and determine which method should be preferred.

Question C
Estimate the next-week VaR (5 trading days) at the 90% confidence level using:
1. Historical simulation.
2. Parametric VaR under the assumption of normally distributed returns.
3. Parametric VaR under the assumption of Student-t distributed returns.
