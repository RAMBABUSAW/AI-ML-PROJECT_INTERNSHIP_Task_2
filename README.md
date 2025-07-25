📈 Stock Market Price Predictor
🧠 Objective
The main goal of this project is to predict the closing prices of stocks using historical data and machine learning techniques. The model is trained on past stock prices and can predict future trends based on learned patterns.

📊 Dataset Used
The stock data is collected using the Yahoo Finance API (yfinance Python library). It provides reliable and up-to-date stock market data for multiple companies.

Stocks Selected: 10 randomly chosen companies from different sectors:

AAPL, MSFT, GOOGL, AMZN, TSLA, META, NFLX, NVDA, INTC, IBM

Time Period: Last 5 years of daily stock prices (2019–2024)

Data Fields: Date, Open, High, Low, Close, Volume

🧹 Data Preprocessing
Before training the model, the following preprocessing steps are applied:

Select Feature: Only the Close price is used for prediction.

Target Variable: The Target is set as the next day's closing price.

Handling Missing Values: Rows with missing data are dropped.

Scaling: The Close prices are standardized using StandardScaler to improve model performance.

Train-Test Split: 80% of the data is used for training, 20% for testing (chronologically, no shuffle).

🧮 Machine Learning Model
The model used for this project is:

🔹 Linear Regression
A simple and fast model for predicting continuous values.

Fits a straight line that best captures the relationship between the input (Close) and output (Target).

(You can also try Random Forest or LSTM in advanced versions.)

📊 Evaluation Metrics
Two metrics are used to evaluate the model:

Metric	Description
RMSE	Root Mean Squared Error – measures prediction error
R² Score	Coefficient of Determination – measures how well the model fits the data

These values are printed for each stock after prediction.

📈 Visualizations
For each stock, a line graph is plotted comparing:

Actual closing prices (blue line)

Predicted closing prices (orange line)

The graphs help visualize how close the model’s predictions are to real prices.

The plots are saved as .png images for future reference.

🧰 Technologies and Libraries Used
Tool/Library	Purpose
Python	Programming language
pandas	Data loading and manipulation
yfinance	Downloading historical stock data
scikit-learn	Machine learning (Linear Regression, metrics)
matplotlib	Visualization of predictions
numpy	Numerical operations

📂 Project Structure
bash
Copy
Edit
stock-price-predictor/
│
├── stock_predictor.py       # Main Python script
├── summary_results.csv      # Evaluation summary for all stocks
├── AAPL_prediction.png      # Plot of predicted vs actual (example)
├── ...
├── README.md                # Project documentation (this file)
✅ Summary
This project demonstrates how machine learning can be used to forecast future stock prices using historical trends. The system:

Loads and processes 5 years of stock data for multiple companies,

Trains a Linear Regression model to predict the next day’s closing price,

Evaluates accuracy using RMSE and R² Score,

Shows results with plots of actual vs predicted values.

It is a great beginner-friendly project to understand stock data, regression, and how to apply ML to real-world problems.
