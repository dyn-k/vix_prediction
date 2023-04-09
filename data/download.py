import os

import pandas as pd
import yfinance as yf

TICKER = {
    "vix" : "^VIX",
    "snp500" : "^GSPC",
}

class RawData:
    
    @staticmethod
    def get(name="vix"):
        data_path = f"./data/raw/{name}.csv"
        if not os.path.exists(data_path):
            data = yf.download(TICKER[name], start="2013-01-01", end="2022-12-31")
            data.index = data.index.strftime('%Y-%m-%d')
            data.to_csv(data_path, index=True, encoding="utf-8")
        else:
            data = pd.read_csv(data_path, index_col="Date", parse_dates=["Date"], encoding="utf-8")
        return data

if __name__ == "__main__":
    data = RawData.get('vixfuture')
    print(data)