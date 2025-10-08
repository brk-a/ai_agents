"""
Quantitative value screener
Build an investing strategy that selects
50 stocks with the highest value from
a value investing perspective
then calculate recommended trades for an
equal-weight portfolio of said stocks
"""

import math
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from secrets import IEX_CLOUD_API_KEY, BASE_URL


class QuantitativeValueScreener:
    """Quantitative value screener to build a value-investing portfolio."""

    def __init__(self, stocks: Optional[List[str]] = None):
        self.df: Optional[pd.DataFrame] = None
        self.stocks = stocks or []

    def __str__(self) -> str:
        return f"QuantitativeValueScreener with {len(self.stocks)} stocks."

    def __repr__(self) -> str:
        rows = len(self.df) if self.df is not None else 0
        return f"<QuantitativeValueScreener stocks={len(self.stocks)} rows={rows}>"

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

    @staticmethod
    def chunks(lst: List[str], n: int) -> List[List[str]]:
        """Split list into successive n-sized chunks."""
        if not lst or n <= 0:
            raise ValueError("Invalid input: lst must be non-empty list, n a positive integer.")
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    def initialise_dataframe(self) -> None:
        """Initialise empty DataFrame for composite screener data."""
        columns = [
            "Ticker", "Price", "PE_ratio", "PE_percentile",
            "PB_ratio", "PB_percentile", "PS_ratio", "PS_percentile",
            "EV_EBITDA_ratio", "EV_EBITDA_percentile", "EV_GP_ratio",
            "EV_GP_percentile", "Robust_value_score",
            "Number_of_shares_to_buy"
        ]
        self.df = pd.DataFrame(columns=columns)

    def load_data_from_api(self, composite: bool = False) -> None:
        """Load stock data from API into DataFrame.

        Args:
            composite (bool): If True, loads extended metrics.
        """
        if not self.stocks:
            raise ValueError("Stock list is empty; load or set stocks first.")

        if self.df is None or self.df.empty:
            self.initialise_dataframe()

        rows = []
        ticker_chunks = self.chunks(self.stocks, 100)

        for chunk in ticker_chunks:
            tickers_str = ",".join(chunk)
            url = f"{BASE_URL}/{tickers_str}/quote"
            if composite:
                url += ",advanced-stats"
            url += f"?token={IEX_CLOUD_API_KEY}"

            response = requests.get(url)
            if response.status_code != 200:
                raise ConnectionError(f"API request failed with status {response.status_code}")

            data = response.json()

            for ticker in chunk:
                try:
                    quote = data[ticker]["quote"]
                    price = quote.get("latestPrice", np.nan)
                    pe_ratio = quote.get("peRatio", np.nan)

                    if composite:
                        stats_adv = data[ticker].get("advanced-stats", {})
                        pb_ratio = stats_adv.get("priceToBook", np.nan)
                        ps_ratio = stats_adv.get("priceToSales", np.nan)
                        ev = stats_adv.get("enterpriseValue", np.nan)
                        ebitda = stats_adv.get("EBITDA", np.nan)
                        gross_profit = stats_adv.get("grossProfit", np.nan)

                        ev_ebitda_ratio = (
                            float(ev) / float(ebitda)
                            if ebitda not in [None, 0, np.nan] else np.nan
                        )
                        ev_gp_ratio = (
                            float(ev) / float(gross_profit)
                            if gross_profit not in [None, 0, np.nan] else np.nan
                        )

                        row = {
                            "Ticker": ticker,
                            "Price": price,
                            "PE_ratio": pe_ratio,
                            "PE_percentile": np.nan,
                            "PB_ratio": pb_ratio,
                            "PB_percentile": np.nan,
                            "PS_ratio": ps_ratio,
                            "PS_percentile": np.nan,
                            "EV_EBITDA_ratio": ev_ebitda_ratio,
                            "EV_EBITDA_percentile": np.nan,
                            "EV_GP_ratio": ev_gp_ratio,
                            "EV_GP_percentile": np.nan,
                            "Robust_value_score": np.nan,
                            "Number_of_shares_to_buy": np.nan,
                        }
                    else:
                        row = {
                            "Ticker": ticker,
                            "Price": price,
                            "PE_ratio": pe_ratio,
                            "Number_of_shares_to_buy": np.nan
                        }
                except KeyError:
                    row = {k: np.nan for k in self.df.columns}
                    row["Ticker"] = ticker

                rows.append(row)

        self.df = pd.DataFrame(rows)

    def remove_glamour_stocks_single_metric(self) -> pd.DataFrame:
        """Filter stocks by PE ratio ascending, positive only, top 50."""
        if self.df is None or self.df.empty:
            raise ValueError("Data is not loaded.")

        df_filtered = self.df.dropna(subset=["PE_ratio"])
        df_filtered = df_filtered[df_filtered["PE_ratio"] > 0]
        df_filtered = df_filtered.sort_values("PE_ratio").head(50).reset_index(drop=True)

        return df_filtered

    def remove_glamour_stocks_composite(self) -> pd.DataFrame:
        """Filter stocks using composite metrics with NaN handling and ranking.

        Returns:
            DataFrame with top 50 stocks by composite value score.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Data is not loaded.")

        percent_na = self.df.isnull().any(axis=1).mean()
        if percent_na <= 0.05:
            df_clean = self.df.dropna()
        else:
            df_clean = self.df.copy()
            cols_to_impute = [
                "PE_ratio", "PB_ratio", "PS_ratio", "EV_EBITDA_ratio", "EV_GP_ratio"
            ]
            for col in cols_to_impute:
                if col in df_clean.columns:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)

        df_clean = df_clean[df_clean["PE_ratio"] > 0]

        # Calculate percentile ranks
        for metric in [
            "PE_ratio", "PB_ratio", "PS_ratio", "EV_EBITDA_ratio", "EV_GP_ratio"
        ]:
            if metric in df_clean.columns:
                percentile_col = metric.replace("ratio", "percentile")
                df_clean[percentile_col] = df_clean[metric].rank(pct=True)

        # Average percentile ranks as composite value score; lower is better
        percentile_cols = [col for col in df_clean.columns if col.endswith("percentile")]
        df_clean["Robust_value_score"] = df_clean[percentile_cols].mean(axis=1)

        df_final = df_clean.sort_values("Robust_value_score").head(50).reset_index(drop=True)

        return df_final

    def portfolio_input(self) -> float:
        """Prompt user to enter portfolio size with input validation.

        Returns:
            Float portfolio size entered by user.
        """
        while True:
            value = input("Enter the size of your portfolio: ").strip()
            try:
                portfolio_size = float(value)
                if portfolio_size <= 0:
                    print("Portfolio size must be a positive number.")
                else:
                    return portfolio_size
            except ValueError:
                print("Please enter a valid number.")

    def calculate_number_of_shares_to_buy(self, equal_weight: bool = True) -> pd.DataFrame:
        """Calculate number of shares to buy for equal-weight portfolio.

        Args:
            equal_weight (bool): If True, use filtered composite stocks.

        Returns:
            DataFrame including number of shares to buy.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Data is not loaded.")

        if equal_weight:
            top_stocks_df = self.remove_glamour_stocks_composite()
        else:
            top_stocks_df = self.df.copy()

        portfolio_size = self.portfolio_input()
        position_size = portfolio_size / len(top_stocks_df)

        top_stocks_df["Number_of_shares_to_buy"] = top_stocks_df["Price"].apply(
            lambda price: math.floor(position_size / price)
            if price and not pd.isna(price) else 0
        )
        self.df = top_stocks_df
        return self.df

    def load_stocks_list(self, path_to_file: str) -> None:
        """Load list of stock tickers from CSV file's first column."""
        tickers = pd.read_csv(path_to_file, usecols=[0])
        self.stocks = tickers.iloc[:, 0].dropna().astype(str).tolist()

    def load_csv(self, path_to_file: str) -> pd.DataFrame:
        """Load DataFrame from CSV file."""
        self.df = pd.read_csv(path_to_file)
        return self.df

    def load_excel(self, path_to_file: str) -> pd.DataFrame:
        """Load DataFrame from Excel file."""
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

            # TODO: implement columns to reflect the composite screener output
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
            }

            for idx, fmt in col_formats.items():
                worksheet.set_column(idx, idx, 25, fmt)


if __name__ == "__main__":
    # Example usage
    screener = QuantitativeValueScreener()
    screener.load_stocks_list("stocks.csv")  # Loads stock ticker list
    screener.load_data_from_api(composite=True)  # Fetch composite data
    filtered_df = screener.remove_glamour_stocks_composite()
    screener.df = filtered_df
    screener.calculate_number_of_shares_to_buy()
    screener.to_excel("recommended_trades.xlsx")
