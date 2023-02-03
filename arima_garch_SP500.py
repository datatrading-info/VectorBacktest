#!/usr/bin/python
# -*- coding: utf-8 -*-
# arima_garch_SP500.py

# codice python relativo all'articolo presente su datatrading.info
# https://datatrading.info/strategia-di-trading-sullindice-sp500-con-i-modelli-arima-e-garch/

import yfinance as yf
import pandas as pd
import numpy as np

symbol = '^GSPC'
start = '2000-01-01'
end = '2018-07-01'
SP500 = yf.download(symbol, start=start, end=end)

log_ret = np.log(SP500['Adj Close']) - np.log(SP500['Adj Close'].shift(1))
print("")