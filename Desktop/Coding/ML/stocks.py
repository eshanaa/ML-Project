#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


import yfinance as yf


def fetch_stock_data(ticker, start_date, end_date, interval='1d'):
    """Fetch historical data for a given stock ticker."""
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data


def main():
    # Example: Fetch data for Apple Inc. from 2020-01-01 to 2021-01-01
    apple_data = fetch_stock_data('AAPL', '2020-01-01', '2021-01-01')
    print(apple_data)


if __name__ == "__main__":
    main()
