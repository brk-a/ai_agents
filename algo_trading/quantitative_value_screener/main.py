"""
Quantitative value screener
Build an investing strategy that selects
50 stocks with the highest value from
a value investing perspective
then calculate recommended trades for an
equal-weight portfolio of said stocks
"""

# 3:39:38

import math
from statistics import mean

import numpy as np
import pandas as pd
import requests
from scipy import stats

from secrets import IEX_CLOUD_API_KEY, BASE_URL


class QuantitativeValueScreener:
    """ Quantitative value screener """
    def __init__(self):
        self.df = None
        self.stocks = None
        self.columns = None

    def __str__(self):
        pass

    def __repr__(self):
        pass

    @classmethod
    def from_file(cls, path: str, file_type: str = "csv") -> "QuantitativeValueScreener":
        """Create instance and load data from CSV or Excel file."""
        instance = cls()
        if file_type.lower() == "csv":
            instance.load_csv(path)
        elif file_type.lower() == "excel":
            instance.load_excel(path)
        else:
            raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")
        return instance

    def single_metric_load_data_from_api(self):
        """ load stock(s) data from API. this is what we will analyse in the screeners that follow """
        if self.df.empty:
            self.create_initial_dataframe()
        
        ticker_groups = list(chunks(self.stocks, 100))
        ticker_strings = []
        for i in range(0, len(ticker_groups)):
            ticker_strings.append(",".join(ticker_groups[i]))
        
        for ticker_group in ticker_groups:
            data = requests.get(f"{BASE_URL}/{ticker_group}/quote?token={IEX_CLOUD_API_KEY}").json()
            for ticker in ticker_group:
                try:
                    price = data[ticker]["quote"]["latestPrice"]
                    pe_ratio = data[ticker]["quote"]["peRatio"]
                except KeyError:
                    price = np.nan
                    pe_ratio = np.nan
                
                row = {
                    "Ticker": ticker,
                    "Price": price,
                    "PE_ratio": pe_ratio,
                    "Number_of_stocks_to_buy": "N/A"
                }
                self.df = self.df.append(row, ignore_index=True)
    
    def composite_metric_load_data_from_api(self):
        """ load stock(s) data from API. this is what we will analyse in the screeners that follow """
        if self.df.empty:
            self.create_initial_dataframe()
        
        ticker_groups = list(chunks(self.stocks, 100))
        ticker_strings = []
        for i in range(0, len(ticker_groups)):
            ticker_strings.append(",".join(ticker_groups[i]))
        
        for ticker_group in ticker_groups:
            url = f"{BASE_URL}/{ticker_group}/quote,advanced-stats?token={IEX_CLOUD_API_KEY}"
            data = requests.get(url).json
            for ticker in ticker_group:
                try:
                    price = data[ticker]["quote"]["latestPrice"]
                    pe_ratio = data[ticker]["quote"]["peRatio"]
                    pb_ratio = data[ticker]["advanced-stats"]["priceToBook"]
                    ps_ratio = data[ticker]["advanced-stats"]["priceToSales"]
                    enterprise_value = data[ticker]["advanced-stats"]["enterpriseValue"]
                    ebitda = data[ticker]["advanced-stats"]["EBITDA"]
                    gross_profit = data[ticker]["advanced-stats"]["grossProfit"]
                    ev_ebitda_ratio = enterprise_value / ebitda
                    ev_gp_ratio = enterprise_value / gross_profit
                except KeyError:
                    price = np.nan
                    pe_ratio = np.nan
                    pb_ratio = np.nan
                    ps_ratio = np.nan
                    ev_ebitda_ratio = np.nan
                    ev_gp_ratio = np.nan
                
                row = {
                    "Ticker": ticker,
                    "Price": price,
                    "PE_ratio": pe_ratio,
                    "PB_ratio": pb_ratio,
                    "PS_ratio": ps_ratio,
                    "EV_Ebitda_ratio": ev_ebitda_ratio,
                    "EV_GP_ratio": ev_gp_ratio,
                    "Number_of_stocks_to_buy": "N/A"
                }
                self.df = self.df.append(row, ignore_index=True)

    def single_metric_remove_glamour_stocks(self) -> pd.DataFrame:
        """ determine which stocks to buy based on one metric: PE ratio """
        if not self.df or len(self.df) == 0:
            raise ValueError("data is not available")

        # sort on PE ratio in ascending order
        self.df.sort_values("PE_ratio", ascending = True, inplace = True)

        # remove stocks with negative PE ratio
        self.df = self.df[self.df["PE_ratio"] > 0]

        # reset indices
        self.df.reset_index(inplace = True)
        self.df.drop("index", axis= 1, inplace=True)

        # get the top 50 highest PE ratio  entries
        self.df = self.df[:50]

        return self.df

    def composite_metric_remove_glamour_stocks(self) -> pd.DataFrame:
        """ determine which stocks to buy based on a number of metrics: PE ratio, PB ratio, PS ratio, EV/EBITDA and EV/gross profit """
        if not self.df or len self.df == 0:
            raise ValueError("data is not available")

        # sort on PE ratio in ascending order
        self.df.sort_values("PE_ratio", ascending = True, inplace = True)

        # remove stocks with negative PE ratio
        self.df = self.df[self.df["PE_ratio"] > 0]

        # reset indices
        self.df.reset_index(inplace = True)
        self.df.drop("index", axis= 1, inplace=True)

        # get the top 50 highest PE ratio  entries
        self.df = self.df[:50]

        return self.df

    @staticmethod
    def chunks(arr: list, n: int):
        """Yield successive n-sized chunks from list."""
        if not arr or not isinstance(arr, list) or not isinstance(n, int) or n <= 0:
            raise ValueError("Invalid input: arr must be a list and n a positive integer.")
        for i in range(0, len(arr), n):
            yield arr[i : i + n]
    
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
        top_stocks_df = self.composite_metric_remove_glamour_stocks()
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