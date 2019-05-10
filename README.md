# Data698
# Report
- https://rawcdn.githack.com/fung1091/Data698/master/final/finalreport1.html
- https://nbviewer.jupyter.org/github/fung1091/Data698/blob/master/final/finalreport.ipynb




Pdf final report at Github and uploaded:

https://github.com/fung1091/Data698/blob/master/final/finalreport_final.pdf

Full code of ipynb at Github:

https://github.com/fung1091/Data698/blob/master/final/finalreport.ipynb

or use the follow link to open:

https://nbviewer.jupyter.org/github/fung1091/Data698/blob/master/final/finalreport.ipynb

# Present
- https://github.com/fung1091/Data698/blob/master/final/presentation.pdf
- https://youtu.be/DZJXHo-h6XU



# Data 698 - Final report

## Tze Fung Lung, Jim

## April 17, 2019

### Topic: Portfolio optimization and Machine learning with visualization analysis for S&P 500


## 10 Keyword: 
- S&P500 
- Stocks 
- Return 
- Risk 
- Strategic 
- Linear 
- k-Nearest 
- Moving-Average 
- ARIMA 
- LSTM

# Table of contents

1. [Introduction](#1)<br>
   1.1 [Objective](#2)<br>
   1.2 [Description of the Problem](#3)<br>
   1.3 [Why the problem is interesting](#4)<br>
   1.4 [What other approaches](#5)<br>
   1.5 [Discussion on your hypothesis](#6)<br>
2. [Literature review](#7)<br>
   2.1 [S&P 500 Index](#8)<br>
   2.2 [Time series forecasting](#9)<br>
   2.3 [portfolio construction framework](#10)<br>
   2.4 [Monte Carlo Simulation](#11)<br>
   2.5 [Machine learning - Moving average](#12)<br>
   2.6 [Machine learning - Linear regression](#13)<br>
   2.7 [Machine learning - K-Nearest - Neighbors](#14)<br>
   2.8 [Machine learning - ARIMA](#15)<br>
   2.9 [Machine learning - Prophet](#16)<br>
   2.10 [Machine learning - Long Short-Term Memory](#17)<br>
3. [Methodology section](#18)<br>
   3.1 [Data Exploration](#19)<br>
   3.2 [Higher monthly return](#20)<br>
   3.3 [Portfolio Optimization](#21)<br>
   3.4 [Machine Learning](#22)<br>
4. [Definition of data collection method](#23)<br>
   4.1 [Data Exploration](#24)<br>
   4.2 [Data Preparation - Stock Tickers and Dataset](#25)<br>
   4.3 [Visualization and Correlation](#26)<br>
5. [Experimentation and Results](#27)<br>
   5.1 [The highest Correlation (Sort out top 30 stocks from all 500 shares)](#28)<br>
   5.2 [The highest monthly return (Sort out top 10 from 30 stocks share)](#29)<br>
   5.3 [portfolio optimization - Random Portfolios from top 10 stocks share](#30)<br>
   5.4 [Portfolio Optimization - Efficient Frontier from top 10 stocks share](#31)<br>
6. [Machine Learning](#32)<br>
   6.1 [Moving average](#33)<br>
   6.2 [Linear regression](#34)<br>
   6.3 [K-Nearest - Neighbors](#35)<br>
   6.4 [ARIMA](#36)<br>
   6.5 [Prophet](#37)<br>
   6.6 [Long Short-Term Memory](#38)<br>
   6.7 [Model Comparsion](#39)<br>
7. [Conclusion](#40)<br>

## 1. Introduction<a id="1"></a>

### 1.1 Objectives<a id="2"></a>

Predicting how the stock market will perform is one of the most difficult things to do. There are so many factors involved in the prediction – physical factors vs. physiological, rational and irrational behavior, etc. All these aspects combine to make share prices volatile and very difficult to predict with a high degree of accuracy.

The S&P 500 is widely regarded as the best single gauge of large-cap U.S. equities. The index includes 500 leading companies and captures approximately 80% coverage of available market capitalization.

### 1.2 Description of the Problem<a id="3"></a>

We’ll look at the S&P 500, an index of the largest US companies. The S&P 500 is an American stock market index based on the market capitalization of 500 large companies having common stock listed on the NYSE, NASDAQ Exchange.

I will load all 500 dataset in S&P 500 for analysis by using portfolio optimization to get the possible several stocks with higher return and lower risk. And using the machine learning predict the investment trend for S&P 500 index.

- What are the top 20 higher monthly return among all 500 number of stocks in S&P500 by Mathematical programming? The target is to **find out the top valuable, higher return with lower risk of stocks**.

- Could I invest these top 20 stocks now by analysis for the **trend of S&P500 index by Machine learning**? It is to determine if I could **invest these stocks by choosing the most accuracy model with the trend**.

### 1.3 Why the problem is interesting <a id="4"></a>

**Automatic trading without anyone involved will be the trend of stock market near future**. I would like to use the data science methods to make a strategic for investment.

I will study which method of machine learning would be more accurate, suitable for prediction by using root-mean-squared error, that the prediction will be more meaningful in use.

### 1.4 What other approaches have been tried <a id="5"></a>

First of all, I will construct the portfolio optimization in order to achieve a maximum expected return given their risk preferences due to the fact that the returns of a portfolio are greatly affected by nature of the relationship between assets and their weights in the portfolio.

The top 20 monthly return of stocks will be get into the portfolio optimization. Then in order to get the higher return and lower risk of stocks, the portfolio optimization will be conducted to find out which are the best choose of investment and generate the visualization for returns and volatility.

For the next part, I will work with historical data about the S&P500 price index to understand if I can invest in market this moment. I will implement a mix of machine learning algorithms to predict the future stock price of this company, starting with simple algorithms like averaging and linear regression, and then moving on to advanced techniques like Auto ARIMA and LSTM.

And I will compare the models by using root-mean-squared error (RMSE) to measure of how model performed and measure difference between predicted values and the actual values.

### 1.5 Discussion on your hypothesis is and how you specific solution will improve <a id="6"></a>

Stock market analysis is divided into two parts – Fundamental Analysis and Technical Analysis.

Fundamental Analysis involves analyzing the company’s future profitability on the basis of its current business environment and financial performance. Technical Analysis, on the other hand, includes reading the charts and using statistical figures to identify the trends in the stock market.

We’ll scrape all S&P 500 tickers from Wiki and load all 500 dataset to be in cleaning and appending the adjusted closing price from 2008 to 2018.

- **Moving Average** - The predicted closing price for each day will be the average of a set of previously observed values. Instead of using the simple average, we will be using the moving average technique which uses the latest set of values for each prediction.

- **Linear Regression** - The most basic machine learning algorithm that can be implemented on this data is linear regression. The linear regression model returns an equation that determines the relationship between the independent variables and the dependent variable.

- **K-Nearest** - Neighbors Another interesting ML algorithm that one can use here is kNN (k nearest neighbours). Based on the independent variables, kNN finds the similarity between new data points and old data points.

- **ARIMA** - ARIMA is a very popular statistical method for time series forecasting. ARIMA models take into account the past values to predict the future values.

- **Long Short Term Memory (LSTM)** - LSTMs are widely used for sequence prediction problems and have proven to be extremely effective.

## 2. Literature review <a id="7"></a>

### 2.1 S&P 500 Index <a id="8"></a>

The S&P 500 or Standard & Poor's 500 Index is a market-capitalization-weighted index of the 500 largest U.S. publicly traded companies. The index is widely regarded as the best gauge of large-cap U.S. equities.

The S&P 500 uses a market capitalization weighting method, giving a higher percentage allocation to companies with the largest market capitalizations. The market capitalization of a company is calculated by taking the current stock price and multiplying it by the outstanding shares.

Reference: 
- https://www.investopedia.com/terms/s/sp500.asp

### 2.2 Time series forecasting <a id="9"></a>

**Time series** is a collection of data points collected at constant time intervals. These are analyzed to determine the long term trend so as to forecast the future or perform some other form of analysis. But what makes a TS different from say a regular regression problem? There are 2 things:

It is time dependent. So the basic assumption of a linear regression model that the observations are independent doesn’t hold in this case.
Along with an increasing or decreasing trend, most TS have some form of seasonality trends, i.e. variations specific to a particular time frame. For example, if you see the sales of a woolen jacket over time, you will invariably find higher sales in winter seasons.

Reference:
- https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

### 2.3 portfolio construction framework <a id="10"></a>

**Modern portfolio theory (MPT)** provides investors with a portfolio construction framework that maximizes returns for a given level of risk, through diversification. MPT reasons that investors should not concern themselves with an individual investment’s expected return, but rather the weighted average of the expected returns of a portfolio’s component securities as well as how individual securities move together. Markowitz consequently introduced the concept of covariance to quantify this co-movement.

It proposed that investors should instead consider variances of return, along with expected returns, and choose portfolios offering the highest expected return for a given level of variance. These portfolios were deemed “efficient.”  For given levels of risk, there are multiple combinations of asset classes (portfolios) that maximize expected return. Markowitz displayed these portfolios across a two-dimensional plane showing expected return and standard deviation, which we now call the efficient frontier. 

Reference: 
- https://www.windhamlabs.com/insights/modern-portfolio-theory/

### 2.4 Monte Carlo Simulation <a id="11"></a>

Monte Carlo simulations are used to model the probability of different outcomes in a process that cannot easily be predicted due to the intervention of random variables. It is a technique used to understand the impact of risk and uncertainty in prediction and forecasting models.

The technique was first developed by Stanislaw Ulam, a mathematician who worked on the Manhattan Project. After the war, while recovering from brain surgery, Ulam entertained himself by playing countless games of solitaire. He became interested in plotting the outcome of each of these games in order to observe their distribution and determine the probability of winning. After he shared his idea with John Von Neumann, the two collaborated to develop the Monte Carlo simulation.

Reference: 
- https://www.investopedia.com/terms/m/montecarlosimulation.asp

### 2.5 Machine learning - Moving average <a id="12"></a>

Smoothing is a technique applied to time series to remove the fine-grained variation between time steps.

The hope of smoothing is to remove noise and better expose the signal of the underlying causal processes. Moving averages are a simple and common type of smoothing used in time series analysis and time series forecasting.

Calculating a moving average involves creating a new series where the values are comprised of the average of raw observations in the original time series.

Reference:
- https://machinelearningmastery.com/moving-average-smoothing-for-time-series-forecasting-python/

### 2.6 Machine learning - Linear regression <a id="13"></a>

Linear regression is a very simple approach for supervised learning. Though it may seem somewhat dull compared to some of the more modern algorithms, linear regression is still a useful and widely used statistical learning method. Linear regression is used to predict a quantitative response Y from the predictor variable X.

Linear Regression is made with an assumption that there’s a linear relationship between X and Y.

Reference: 
- https://medium.com/simple-ai/linear-regression-intro-to-machine-learning-6-6e320dbdaf06
- https://www.datascience.com/blog/time-series-forecasting-machine-learning-differences

### 2.7 Machine learning - K-Nearest - Neighbors <a id="14"></a>

K-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:

In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification. The k-NN algorithm is among the simplest of all machine learning algorithms.

Reference:
- https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/
- https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
- https://tomaszgolan.github.io/introduction_to_machine_learning/markdown/introduction_to_machine_learning_01_knn/introduction_to_machine_learning_01_knn/

### 2.8 Machine learning - ARIMA <a id="15"></a>

An ARIMA model is a class of statistical models for analyzing and forecasting time series data.

It explicitly caters to a suite of standard structures in time series data, and as such provides a simple yet powerful method for making skillful time series forecasts.

ARIMA is an acronym that stands for AutoRegressive Integrated Moving Average. It is a generalization of the simpler AutoRegressive Moving Average and adds the notion of integration.

This acronym is descriptive, capturing the key aspects of the model itself. Briefly, they are:

- AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
- I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
- MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.

Each of these components are explicitly specified in the model as a parameter. A standard notation is used of ARIMA(p,d,q) where the parameters are substituted with integer values to quickly indicate the specific ARIMA model being used.

The parameters of the ARIMA model are defined as follows:

- p: The number of lag observations included in the model, also called the lag order.
- d: The number of times that the raw observations are differenced, also called the degree of differencing.
- q: The size of the moving average window, also called the order of moving average.

Reference:
- https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

### 2.9 Machine learning - Prophet <a id="16"></a>

Prophet is a procedure for forecasting time series data. It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays. It works best with daily periodicity data with at least one year of historical data. Prophet is robust to missing data, shifts in the trend, and large outliers.

Prophet is open source software released by Facebook's Core Data Science team.

The Prophet procedure is an additive regression model with four main components:

- A piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting changepoints from the data.
- A yearly seasonal component modeled using Fourier series.
- A weekly seasonal component using dummy variables.
- A user-provided list of important holidays.

Reference:
- https://research.fb.com/prophet-forecasting-at-scale/

### 2.10 Machine learning - Long Short-Term Memory <a id="17"></a>

Long Short-Term Memory usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is their default behavior. 

All recurrent neural networks have the form of a chain of repeating modules of a neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer. LSTMs also have this chain like structure, but the repeating module has a different structure. The key to LSTMs is the cell state which is acting like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to flow along it unchanged. 

Reference:
- https://machinelearningtutorials.com/long-short-term-memory-lstm/


## 3. Methodology section <a id="18"></a>

This project are separated 4 part of analysis from data exploration, visualization, correlation and monthly return for data extraction by mathematical programming, portfolio optimization and machine learning.  

The following strategic analysis as follow:

### 3.1 Data Exploration (Getting 500 stocks data) <a id="19"></a>

1. To use **Beautifulsoup** grap stocks symbol for S&P 500

2. To use API in the function of **"fix_yahoo_finance"** to load all dataset for 500 stocks share from January 2008 until Now, each stocks symbol will be used to load their own CSV dataset file. 

3. To select each stocks of adjusted closing price and use the **"join"** function and rename the columns to create the joined CSV

4. To use both **"plotly"** and **"matplotlib"** for data visualization, "iplot" can be used to compare the detail price in certain period of time. 

<a id="20"></a>
### 3.2 Higher monthly return  (Choosing top 10 stocks correlation and monthly return with index from 500 stocks data)

1. To use **corr()** function to find out the **top 30 stocks** which are higher correlation with S&P 500 index from **500 stocks share**.

2. The Last part of this project will conduct the prediction of machine learning for S&P 500 index, so the correlation between stocks and index is important reference for the trend affected by index fluctuation. And it will provide the buy and sell signal when the prediction are generated.

3. After getting the top 30 stocks of highest correlation with index, we will calculate and **compare the higher monthly return for this 30 stocks**. 

4. To calculate the **top 10 stocks of highest monthly return from the above 30**. we will sort and extract the joined dataset for the next part of Portfolio optimization.

### 3.3 Portfolio Optimization (Testing top 10 stocks from part 2 for portfolio optimization) <a id="21"></a>

1. After getting the top 10 stocks from part 2 by using Highest correlation and  monthly return without consideration the factor of risk, but this stage we would like to use the portfolio optimization method to decide the investment strategic.

2. Calculating the **top 10 higher monthly return** of stocks share as Medium or long term investment

3. Using **portfolio optimization** to calculate the **top 10 higher monthly return** of stocks share as Medium or long term investment which are higher monthly return and lower risk as investment strategic.

### 3.4 Machine Learning <a id="22"></a>

After portfolio optimization analysis, it assume we get the proportion of investment strategic, we will expect to know if it is time to invest this moment or when is the best momnent for buying or selling. Therefore, the machine learning for S&P 500 index will be conducted in machine learning process by comparing different methods and model. 

The command process as follow:
    - Dropping NAN with dropna in **pandas** 
    - splitting into train and validation
    - Measuring root mean square error (RMSE) for the standard deviation of residuals
    - Plot the prediction 

- 1. Moving average

- 2. Linear Regression
    - Using the function of **linear_model** in **sklearn**
    
- 3. k-Nearest Neighbours
    - Using the function of **neighbors, GridSearchCV, MinMaxScaler** in **sklearn** to find the best parameter

- 4. Auto ARIMA
    - Using the function of **auto_arima** in **pmdarima** which automatically selects the best combination of (p,q,d) that provides the least error.

- 5. Prophet
    - Using the function of **Prophet in fbprophet** which Prophet, designed by Facebook, is a time series forecasting library that The input for Prophet is a dataframe with two columns: date and target.
    
- 6. Long Short Term Memory
    - Using the function of **MinMaxScaler in sklearn** and **LSTM in keras** to create and fit the LSTM network

