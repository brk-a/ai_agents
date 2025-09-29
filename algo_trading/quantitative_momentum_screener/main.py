"""
Momentum investing strategy
Build an investing strategy that selects
50 stocks with the highest price momentum
then calculate recommended trades for an
equal-weight portfolio of said stocks 
"""

import numpy as np
import pandas as pd
import requests
import math
import xlsxwriter
from scipy import stats
from secrets import IEX_CLOUD_API_KEY


class QuantitativeValueMomentumScreener:
    def __init__(self):
        self.df = None
        self.stocks = None

    def __str__(self):
        pass

    def __repr__(self):
        pass

    @classmethod
    def from_file(cls, path: str, file_type: str = "csv") -> "QuantitativeValueMomentumScreener":
        """Alternative constructor to create instance from CSV or Excel file."""
        instance = cls()
        if file_type.lower() == "csv":
            instance.load_csv(path)
        elif file_type.lower() == "excel":
            instance.load_excel(path)
        else:
            raise ValueError("Unsupported file_type. Use 'csv' or 'excel'.")
        return instance

    def create_initial_dataframe(self) -> pd.DataFrame:
        """Create an empty DataFrame with columns for stock data."""
        columns = ["Ticker", "Stock_price", "Market_cap", "Number_of_shares_to_buy"]
        self.df = pd.DataFrame(columns=columns)
        return self.df

    def load_stock_info_from_api(self):
        pass

    def load_stocks_list(self, path_to_file: str) -> None:
        self.stocks = pd.read_csv(path_to_file)

    def load_csv(self):
        pass

    def load_excel(self):
        pass

    def to_excel(self):
        pass

#1:43:34