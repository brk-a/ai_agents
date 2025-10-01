"""
Momentum investing strategy
Build an investing strategy that selects
50 stocks with the highest price momentum
then calculate recommended trades for an
equal-weight portfolio of said stocks 
"""

import math
from statistics import mean

import numpy as np
import pandas as pd
import requests
from scipy import stats

from secrets import IEX_CLOUD_API_KEY, BASE_URL


class QuantitativeValueMomentumScreener:
    """Momentum investing strategy selecting stocks with highest price momentum."""

    HQM_COLUMNS = [
        "Ticker",
        "Price",
        "Number_of_shares_to_buy",
        "One-year_price_return",
        "One-year_return_percentile",
        "Six-month_price_return",
        "Six-month_return_percentile",
        "Three-month_price_return",
        "Three-month_return_percentile",
        "One-month_price_return",
        "One-month_return_percentile",
        "HQM_score",
    ]

    def __init__(self):
        self.df = pd.DataFrame(columns=self.HQM_COLUMNS)
        self.stocks = []

    def __str__(self):
        return f"QuantitativeValueMomentumScreener with {len(self.df)} stocks"

    def __repr__(self):
        return f"{self.__class__.__name__}(stocks_count={len(self.stocks)})"

    @classmethod
    def from_file(cls, path: str, file_type: str = "csv") -> "QuantitativeValueMomentumScreener":
        """Create instance and load data from CSV or Excel file."""
        instance = cls()
        if file_type.lower() == "csv":
            instance.load_csv(path)
        elif file_type.lower() == "excel":
            instance.load_excel(path)
        else:
            raise ValueError("Unsupported file_type. Use 'csv' or 'excel'.")
        return instance

    def create_initial_dataframe(self) -> pd.DataFrame:
        """Create an empty DataFrame with HQM columns."""
        self.df = pd.DataFrame(columns=self.HQM_COLUMNS)
        return self.df

    def naive_load_stock_info_from_api(self) -> None:
        """Naively load stock info for each ticker via IEX Cloud API."""
        if self.df.empty:
            self.create_initial_dataframe()

        ticker_groups = list(self.chunks(self.stocks, 100))

        for group in ticker_groups:
            ticker_string = ",".join(group)
            url = f"{BASE_URL}?symbols={ticker_string}&types=stats,price&token={IEX_CLOUD_API_KEY}"
            data = requests.get(url).json()

            for ticker in group:
                try:
                    stats_data = data[ticker]["stats"]
                    price = data[ticker]["price"]
                    one_year_return = stats_data.get("year1ChangePercent", None)
                except KeyError:
                    price = np.nan
                    one_year_return = np.nan

                row = {
                    "Ticker": ticker,
                    "Price": price,
                    "One-year_price_return": one_year_return,
                    "Number_of_shares_to_buy": "N/A",
                }
                self.df = self.df.append(row, ignore_index=True)

    def more_practical_load_stock_info_from_api(self) -> None:
        """Load stock info with multiple periods and calculate percentiles and HQM score."""
        if self.df.empty or list(self.df.columns) != self.HQM_COLUMNS:
            self.create_initial_dataframe()

        ticker_groups = list(self.chunks(self.stocks, 100))

        for group in ticker_groups:
            ticker_string = ",".join(group)
            url = f"{BASE_URL}?symbols={ticker_string}&types=stats,price&token={IEX_CLOUD_API_KEY}"
            data = requests.get(url).json()

            for ticker in group:
                try:
                    stats_data = data[ticker]["stats"]
                    price = data[ticker]["price"]
                    one_year_return = stats_data.get("year1ChangePercent", None)
                    month_6_return = stats_data.get("month6ChangePercent", None)
                    month_3_return = stats_data.get("month3ChangePercent", None)
                    month_1_return = stats_data.get("month1ChangePercent", None)
                except KeyError:
                    price = np.nan
                    one_year_return = np.nan
                    month_6_return = np.nan
                    month_3_return = np.nan
                    month_1_return = np.nan

                row = {
                    "Ticker": ticker,
                    "Price": price,
                    "Number_of_shares_to_buy": "N/A",
                    "One-year_price_return": one_year_return,
                    "One-year_return_percentile": "N/A",
                    "Six-month_price_return": month_6_return,
                    "Six-month_return_percentile": "N/A",
                    "Three-month_price_return": month_3_return,
                    "Three-month_return_percentile": "N/A",
                    "One-month_price_return": month_1_return,
                    "One-month_return_percentile": "N/A",
                    "HQM_score": "N/A",
                }
                self.df = self.df.append(row, ignore_index=True)

        periods = ["One-year", "Six-month", "Three-month", "One-month"]

        # Calculate percentile ranks for each period
        for period in periods:
            price_col = f"{period}_price_return"
            percentile_col = f"{period}_return_percentile"
            self.df[percentile_col] = self.df[price_col].rank(pct=True)

        # Calculate HQM score as mean of percentiles
        percentile_cols = [f"{period}_return_percentile" for period in periods]
        self.df["HQM_score"] = self.df[percentile_cols].mean(axis=1)

    @staticmethod
    def chunks(arr: list, n: int):
        """Yield successive n-sized chunks from list."""
        if not arr or not isinstance(arr, list) or not isinstance(n, int) or n <= 0:
            raise ValueError("Invalid input: arr must be a list and n a positive integer.")
        for i in range(0, len(arr), n):
            yield arr[i : i + n]

    def naive_remove_low_momentum_stocks(self) -> pd.DataFrame:
        """Remove low momentum stocks by sorting by one-year return and selecting top 50."""
        self.naive_load_stock_info_from_api()
        sorted_df = self.df.sort_values(by="One-year_price_return", ascending=False)
        return sorted_df.head(50).reset_index(drop=True)

    def more_practical_remove_low_momentum_stocks(self) -> pd.DataFrame:
        """Remove low momentum stocks based on HQM score, returning top 50."""
        self.more_practical_load_stock_info_from_api()
        sorted_df = self.df.sort_values(by="HQM_score", ascending=False)
        return sorted_df.head(50).reset_index(drop=True)

    def portfolio_input(self) -> float:
        """Prompt user to enter portfolio size and validate the input."""
        while True:
            value = input("Enter the size of your portfolio: \n").strip()
            try:
                portfolio_size = float(value)
                if portfolio_size <= 0:
                    print("Portfolio size must be a positive number.")
                else:
                    return portfolio_size
            except ValueError:
                print("Please enter a valid number.")

    def calculate_number_of_shares_to_buy(self) -> pd.DataFrame:
        """Calculate number of shares to buy for an equal-weight portfolio."""
        portfolio_size = self.portfolio_input()
        top_stocks_df = self.more_practical_remove_low_momentum_stocks()
        position_size = portfolio_size / len(top_stocks_df)

        top_stocks_df["Number_of_shares_to_buy"] = top_stocks_df["Price"].apply(
            lambda price: math.floor(position_size / price) if price and not pd.isna(price) else 0
        )
        self.df = top_stocks_df
        return self.df

    def load_stocks_list(self, path_to_file: str) -> None:
        """Load list of stock tickers from CSV file (assumed in first column)."""
        tickers = pd.read_csv(path_to_file)
        self.stocks = tickers.iloc[:, 0].dropna().astype(str).tolist()

    def load_csv(self, path_to_file: str) -> pd.DataFrame:
        """Load data from CSV file into the DataFrame."""
        self.df = pd.read_csv(path_to_file)
        return self.df

    def load_excel(self, path_to_file: str) -> pd.DataFrame:
        """Load data from Excel file into the DataFrame."""
        self.df = pd.read_excel(path_to_file)
        return self.df

    def to_excel(self, path_to_file: str) -> None:
        """Export DataFrame to an Excel file with formatted columns."""
        with pd.ExcelWriter(path_to_file, engine="xlsxwriter") as writer:
            self.df.to_excel(writer, index=False, sheet_name="Recommended Trades")
            workbook = writer.book
            worksheet = writer.sheets["Recommended Trades"]

            background_colour = "#0a0a23"
            font_colour = "#ffffff"

            string_format = workbook.add_format({
                "font_color": font_colour,
                "bg_color": background_colour,
                "border": 1,
            })
            currency_format = workbook.add_format({
                "num_format": "KES 0.00",
                "font_color": font_colour,
                "bg_color": background_colour,
                "border": 1,
            })
            integer_format = workbook.add_format({
                "num_format": "0",
                "font_color": font_colour,
                "bg_color": background_colour,
                "border": 1,
            })
            percent_format = workbook.add_format({
                "num_format": "0.0%",
                "font_color": font_colour,
                "bg_color": background_colour,
                "border": 1,
            })

            col_formats = {
                0: string_format,   # Ticker
                1: currency_format, # Price
                2: integer_format,  # Number_of_shares_to_buy
                3: percent_format,  # One-year_price_return
                4: percent_format,  # One-year_return_percentile
                5: percent_format,  # Six-month_price_return
                6: percent_format,  # Six-month_return_percentile
                7: percent_format,  # Three-month_price_return
                8: percent_format,  # Three-month_return_percentile
                9: percent_format,  # One-month_price_return
                10: percent_format, # One-month_return_percentile
                11: percent_format, # HQM_score
            }

            for idx, fmt in col_formats.items():
                worksheet.set_column(idx, idx, 25, fmt)
