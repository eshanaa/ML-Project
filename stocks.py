import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def fetch_stock_data(ticker, start_date, end_date, interval='1d'):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data


def base_filter(user_stock_data, sp500_data, user_stock):
    stock_pct_change = user_stock_data['Close'].pct_change() * 100
    sp500_pct_change = sp500_data['Close'].pct_change() * 100

    # Calculate cumulative percentage change
    stock_cumulative = stock_pct_change.sum()
    sp500_cumulative = sp500_pct_change.sum()

    # Threshold set to 5 times the S&P fluctuation to be dynamic
    threshold = 5 * sp500_cumulative
    print(f"The change in {user_stock} was {round(stock_cumulative, 2)}% and the change in the S&P 500\n"
          f"was {round(sp500_cumulative, 2)}%.")
    if abs(stock_cumulative - sp500_cumulative) < threshold:
        print(f"{user_stock} follows the market trend.")
        return False
    else:
        print(f"{user_stock} does not follow the market trend.")

        return True


def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(data, span1=12, span2=26, signal=9):
    ema1 = data.ewm(span=span1, adjust=False).mean()
    ema2 = data.ewm(span=span2, adjust=False).mean()
    macd = ema1 - ema2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def display_anomalies(anomalies_data):
    if anomalies_data.empty:
        print("No anomalies detected.")
    else:
        print("Anomalies detected:")
        for index, row in anomalies_data.iterrows():
            print(f"Date: {row['Date']}, Close: {row['Close']}, Price Change: {round(row['Price_Change'],2)}%, RSI: {row['RSI']}, MACD: {row['MACD']}")


def stock_anomalies_detection(user_stock, user_chosen_start_date):
    start_date = user_chosen_start_date - timedelta(days=365)
    end_date = user_chosen_start_date - timedelta(days=1)
    formatted_start_date = start_date.strftime('%Y-%m-%d')
    formatted_end_date = end_date.strftime('%Y-%m-%d')

    # usr_pst_yr means user_stock_data_past_year_excluding_week
    usr_pst_yr = fetch_stock_data(user_stock, formatted_start_date, formatted_end_date, interval='1d')
    usr_pst_yr.reset_index(inplace=True)

    # Feature Engineering
    usr_pst_yr['Date'] = pd.to_datetime(usr_pst_yr['Date'])
    usr_pst_yr['Month'] = usr_pst_yr['Date'].dt.month
    usr_pst_yr['Day'] = usr_pst_yr['Date'].dt.day
    usr_pst_yr['DayOfWeek'] = usr_pst_yr['Date'].dt.dayofweek
    usr_pst_yr['SMA_20'] = usr_pst_yr['Close'].rolling(window=20).mean()
    usr_pst_yr['RSI'] = compute_rsi(usr_pst_yr['Close'])  # Function to calculate RSI
    usr_pst_yr['Price_Change'] = usr_pst_yr['Close'].pct_change() * 100
    usr_pst_yr['MACD'], usr_pst_yr['MACD_Signal'] = compute_macd(usr_pst_yr['Close'])

    # Prepare Data for Model, dropping NaN rows
    # X = Features extracted from user_stock_data, emphasis on price changes
    usr_pst_yr['Next_Close'] = usr_pst_yr['Close'].shift(-1)
    usr_pst_yr.dropna(inplace=True)
    training_data = usr_pst_yr[:-7].copy()

    # Prepare X only, y not used in Isolation Forest
    X_training = training_data[['Month', 'Day', 'DayOfWeek', 'Price_Change', 'SMA_20', 'RSI', 'MACD']]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_training)

    # Train the Isolation Forest model
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_train_scaled)

    # Predict and add anomalies
    anomalies = model.predict(X_train_scaled)
    training_data.loc[:, 'Anomaly'] = anomalies

    # Same Process for recent_week_data, using the same scaler
    recent_week_data = usr_pst_yr.tail(7).copy()
    usr_pst_yr.loc[usr_pst_yr.index[:-7], 'Anomaly'] = anomalies

    X_recent_week = recent_week_data[['Month', 'Day', 'DayOfWeek', 'Price_Change', 'SMA_20', 'RSI', 'MACD']]
    X_recent_week_scaled = scaler.transform(X_recent_week)

    recent_week_anomalies = model.predict(X_recent_week_scaled)
    recent_week_data.loc[:, 'Predicted_Anomaly'] = recent_week_anomalies

    anomalies_detected = recent_week_data[recent_week_data['Predicted_Anomaly'] == -1]

    display_anomalies(anomalies_detected)


def get():
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    formatted_start_date = start_date.strftime('%Y-%m-%d')
    formatted_end_date = end_date.strftime('%Y-%m-%d')

    user_stock_data = fetch_stock_data(user_stock, formatted_start_date, formatted_end_date, interval='1d')
    sp500_data = fetch_stock_data('^GSPC', formatted_start_date, formatted_end_date, interval='1d')
    print(user_stock_data)
    print(sp500_data)
    if base_filter(user_stock_data, sp500_data, user_stock) and learning_center(user_stock, datetime.now()):
        # Double-confirmed to not follow the market trend
        pass
    else:
        # Follows the market trend
        pass
    :return:
    """
    pass


def main():
    # 1. Data Changes
    # 2. Anomaly Detection
    # 3. Model Training and Evaluation
    user_stock = 'STLA' # input("Enter the stock ticker: ")

    stock_anomalies_detection(user_stock, datetime.now())


if __name__ == "__main__":
    main()
