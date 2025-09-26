# algo trading fundamentals

## WTF is algo trading?
* to use algorithms  to investment decisions
    - computers do most of the work financial analysts and investment researchers used to do
* many types of algo trading; difference lies in the speed of execution
    - high-frequency (HF) algos
    - big brain
    - ~~big balls, perhaps?~~
    - etc
* any trading and/or investment strategy made by humans can be augmented by algo trading through
    - optimisation &rarr; make parts or all of the strategy faster, less error-prone etc
    - etc
* python is the most common/popular programming language for algo trading
    - such a shame because C or Go or Rust are right there
    - python is slow, therefore, it is mostly used as a *"glue language"* to trigger code that runs in other languages
    - the other way to solve python's slowness is to call the underlying C modules; CPython, regular python, is written in C
## who TF is in the algo trading landscape?
* well... almost anyone with a computer and a trading strategy
    - renaissance tech: $165B AUM (famous for their medallion fund)
    - AQR capital management: $61B AUM (the IBM of the algo trading world)
    - citadel securities: $32B AUM (famous for high-frequency trades)
    - you & I
    - etc
## the algo trading process
* these are the steps one takes when running a quantitiative investing strategy
* the steps *viz*:
    - collect data
    - develop a hypothesis for a strategy
    - backtest said strategy
    - implement said strategy in prod
> ***CAVEAT***<br/> - this is an intro; there is so much more to learn out there <br/> - the code here is not prod-ready <br/> - we will consume an API (we will build our own later) <br/> - we will not execute trades (goes w/o saying); we will generate order sheets instead <br/> - we will save recommended trades in an excel file
## WTF is an API?
* application programming interface
* software that allows your software to interact with someone else's software
    - we will use IEX Cloud API to gather stock market data and to make investment decisions
    - example of an API call

        ```python
            # snip
            symbol = "AAPL"
            api_url = f"https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}"
            data = requests.get(api_url).json()
            print(data)
        ```

* use HTTP verbs to interact make calls to an API
    - said verbs are: `GET`, `POST`, `PUT`, `UPDATE`, `DELETE`, `OPTIONS` etc
    - `GET` allows you to, well, get a record from a DB through the API
    - `POST` allows you to create a record in a DB through the API
    - `PUT` and `UPDATE` allow you to edit/update/modify a record
        - `UPDATE` &rarr; modify info w/o replacing the original entry
        - `PUT` &rarr; modify info by replacing the original entry
    - `DELETE` allows you to, well, delete a record from a DB through the API
    - etc
## WTF is the S&P 500?
* world's most popular stock index
    - standard and poor's 500: represents the performance of the 500 most valuable companies listed in the US
    - many investment funds are benchmarked to the S&P 500; they seek to, at the veryleast, replicate the performance of this index
* it is market-cap-weighted
    - the proportion/weight of a stock in the index is proportional to the company's market capitalisation
    - larger companies have a more outsized influence on the index (another way of saying this is: market cap is directly proportional to weight)
## WTF is momentum investing?
* investing in assets that have increased in price the most
* example
     - say you have to choose between two stocks whose YoY performance *viz*: stock A has grown 35% and stock B has grown 20%
    - naive momentum investing strategy suggests we invest in stock A because it grew faster/more
    - there are other nuances to momentum investing; beware the trap of purity
## WTF is value investing?
* investing in stocks that trade below their perceived intrinsic value
    - buying a shilling for 75 cents with the hope of selling it for a shilling
* popularised by Ben Graham, Warren Buffett ans Seth Klarman
* algo value trading strategies rely on the concept of **multiples**
* **WTF are multiples?**
     - glad you asked...
     - a way value investors use to decide how valuable a company is
     - calculated by dividing a company's stock price by some measure of company's worth, say, earnings or assets
     - example: say you sold all the assets at book value, paid off liabilities and paid the difference, if positive, to each shareholder; how much, per share, would each shareholder get? is that amount larger or smaller than the per-share price based on current market capitalisation/value? if larger, buy, else, sell/do not buy
     - common multiples used: price-to-earnings (PE), price-to-book-value (PB), price-to-free-cash-flow etc
     - each individual multiple has its up and downsides; one way to minimise the impact of any specific multiple is to use a **composite**
* **WTF is a composite?**
    - a combination of multiples