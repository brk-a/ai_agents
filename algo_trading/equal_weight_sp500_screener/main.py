"""
Equal-weight S&P 500 Index Fund.

This class implements logic for an equal-weight S&P 500 index fund,
which is different from the market-cap weighted SPY ETF.
"""

import logging
import requests
import pandas as pd
from secrets import IEX_CLOUD_API_KEY  # Secure API key management


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EqualWeightSPY:
    """Equal-weight S&P 500 index fund."""

    def __init__(self):
        self.df = None
        self.data_json = None
        self.stocks = None

    def __str__(self):
        return f"<EqualWeightSPY stocks count={len(self.stocks) if self.stocks is not None else 0}>"

    @classmethod
    def from_file(cls, path: str, file_type: str = "csv") -> "EqualWeightSPY":
        """
        Alternative constructor to create instance from CSV or Excel file.

        Args:
            path: Path to the file.
            file_type: File type - 'csv' or 'excel'.
        """
        instance = cls()
        if file_type.lower() == "csv":
            instance.load_csv(path)
        elif file_type.lower() == "excel":
            instance.load_excel(path)
        else:
            raise ValueError("Unsupported file_type. Use 'csv' or 'excel'.")
        return instance

    def load_csv(self, path_to_file: str) -> None:
        """Load data from CSV file."""
        self.df = pd.read_csv(path_to_file)

    def load_excel(self, path_to_file: str) -> None:
        """Load data from Excel file."""
        self.df = pd.read_excel(path_to_file)

    def load_stock_info_from_api(self, symbol: str) -> dict:
        """Load stock data from the IEX API for a given symbol."""
        try:
            url = f"https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.HTTPError as e:
            logger.error(f"HTTP error for {symbol}: {e}")
        except requests.RequestException as e:
            logger.error(f"Request error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {e}")
        return {}

    def get_stock_price(self) -> float:
        """Get stock price from data_json."""
        if self.data_json and "latestPrice" in self.data_json:
            return self.data_json["latestPrice"]
        return 0.0

    def get_market_cap(self) -> float:
        """Get market cap from data_json."""
        if self.data_json and "marketCap" in self.data_json:
            return self.data_json["marketCap"]
        return 0.0

    def create_initial_dataframe(self) -> pd.DataFrame:
        """Create an empty DataFrame with columns for stock data."""
        columns = ["Ticker", "Stock_price", "Market_cap", "Number_of_shares_to_buy"]
        self.df = pd.DataFrame(columns=columns)
        return self.df

    def append_data_to_df(self, data: dict) -> None:
        """
        Append a row of stock data to the DataFrame.

        Args:
            data: Dict with keys matching DataFrame columns.
        """
        if self.df is None:
            self.create_initial_dataframe()
        series = pd.Series(data)
        self.df = pd.concat([self.df, pd.DataFrame([series])], ignore_index=True)

    def load_data_all_stocks_list(self) -> None:
        """Load data for all stocks listed in self.stocks DataFrame one by one."""
        if self.stocks is None or "Ticker" not in self.stocks:
            raise ValueError("Stocks list is not defined or does not include 'Ticker' column.")

        self.create_initial_dataframe()
        for stock in self.stocks["Ticker"]:
            stock_data = self.load_stock_info_from_api(stock)
            if stock_data:
                self.append_data_to_df(
                    {
                        "Ticker": stock,
                        "Stock_price": stock_data.get("latestPrice", 0.0),
                        "Market_cap": stock_data.get("marketCap", 0.0),
                        "Number_of_shares_to_buy": "N/A",
                    }
                )
            else:
                logger.warning(f"No data found for stock: {stock}")

    def batch_load_data_all_stocks_list(self) -> None:
        """Load all stocks data in batch from API using batch endpoint."""
        if self.stocks is None or "Ticker" not in self.stocks:
            raise ValueError("Stocks list is not defined or does not include 'Ticker' column.")

        symbols = ",".join(self.stocks["Ticker"].tolist())
        url = (
            f"https://sandbox.iexapis.com/stable/stock/market/batch?"
            f"symbols={symbols}&types=quote&token={IEX_CLOUD_API_KEY}"
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            batch_data = response.json()
        except requests.RequestException as e:
            logger.error(f"Batch API call failed: {e}")
            return

        self.create_initial_dataframe()
        for symbol, data in batch_data.items():
            quote = data.get("quote", {})
            self.append_data_to_df(
                {
                    "Ticker": symbol,
                    "Stock_price": quote.get("latestPrice", 0.0),
                    "Market_cap": quote.get("marketCap", 0.0),
                    "Number_of_shares_to_buy": "N/A",
                }
            )

    def calculate_shares_to_buy(self, total_portfolio_value: float) -> None:
        """
        Calculate the number of shares to buy for each stock to maintain equal weighting.

        Args:
            total_portfolio_value: The total value of the portfolio available for investment.
        """
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty. Load stock data before calculating shares.")

        equal_weight_value = total_portfolio_value / len(self.df)
        self.df["Number_of_shares_to_buy"] = (
            equal_weight_value / self.df["Stock_price"]
        ).apply(int)

    @staticmethod
    def chunks(lst: list, n: int):
        """Yield successive n-sized chunks from a list."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

#57:57