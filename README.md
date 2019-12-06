# competitive data science predict future sales

## Plan
* Benchmark - previous month sales
* Generate lag-based features with sales from previous month and months
* Generate lag-based features not only for item/shop pairs but for total shop or total item sales
* Try to tune hyper parameters for model
* Try ensembling. Start with simple average of linear model and gradient boosted trees. And then try to use stacking.
* Explore new features! There is a lot useful information in the data: text descriptions, item categories and seasonal trends.


## Feature Ideas


## Validation
Let's start with 3 previous months data. Each row in train is a data for 3 previous months and target from the forth month. In train we take from 3 to 22 months, in valid 23 to 32, in test 33 (the last month).
