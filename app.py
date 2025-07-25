# Stock Price Predictor using Linear Regression for 10 Random Stocks

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Define 10 random stock symbols
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'INTC', 'IBM']

# Initialize results storage
results = []

for stock in stocks:
    print(f"\nProcessing stock: {stock}")

    # 2. Download 5 years of data
    data = yf.download(stock, start='2019-01-01', end='2024-01-01')

    if data.empty:
        print(f"Data not available for {stock}. Skipping...")
        continue

    # 3. Use only 'Close' prices and preprocess
    df = data[['Close']].copy()
    df.dropna(inplace=True)

    # Target is next day's closing price
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Close']].values
    y = df['Target'].values

    if len(X) == 0:
        print(f"Not enough data for {stock}. Skipping...")
        continue

    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # 5. Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Predict
    y_pred = model.predict(X_test)

    # 7. Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{stock} - RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

    # Save result
    results.append({'Stock': stock, 'RMSE': rmse, 'R2 Score': r2})

    # 8. Plot and save the result
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='orange')
    plt.title(f'{stock} - Actual vs Predicted')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{stock}_prediction.png")
    plt.show()
    plt.close()

# Summary Table
print("\nSummary of Results:")
summary_df = pd.DataFrame(results)
print(summary_df)

# Optional: Save summary as CSV
summary_df.to_csv("summary_results.csv", index=False)
print("\nSummary results saved to 'summary_results.csv'")