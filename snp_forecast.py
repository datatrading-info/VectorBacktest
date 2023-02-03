# snp_forecast.py

# codice python relativo all'articolo presente su datatrading.info
# https://datatrading.info/strategia-di-forecasting-sul-sp500-backtesting-con-python-e-pandas/

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

import pandas_datareader as pdr
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from backtest.backtest import Strategy, Portfolio
from model.forecast import create_lagged_series

# snp_forecast.py

class SNPForecastingStrategy(Strategy):
    """
    Richiede:
    symbol - simbolo di un'azione per il quale applicare la strategia.
    bars - DataFrame delle barre del simbolo precedente.
    """

    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars
        self.create_periods()
        self.fit_model()

    def create_periods(self):
        """Crea i periodi di training/test."""
        self.start_train = datetime.datetime(2001,1,10)
        self.start_test = datetime.datetime(2005,1,1)
        self.end_period = datetime.datetime(2005,12,31)

    def fit_model(self):
        """
        Applica il Quadratic Discriminant Analyser al indice
        del mercato azionario US (^GPSC in Yahoo).
        """
        # Crea la serie ritardata dell'indice S&P500 del mercato azionario US
        snpret = create_lagged_series(self.symbol, self.start_train, self.end_period, lags=5)

        # Usa i rendimenti dei 2 giorni precedenti come
        # valori di predizione, con la direzione come risposta
        X = snpret[["Lag1","Lag2"]]
        y = snpret["Direction"]

        # Crea i dataset di training e di test
        X_train = X[X.index < self.start_test]
        y_train = y[y.index < self.start_test]

        # Crea i fattori di predizioni per usare
        # la direzione predetta
        self.predictors = X[X.index >= self.start_test]

        # Crea il modello di Quadratic Discriminant Analysis
        # e la strategia previsionale
        self.model = QDA()
        self.model.fit(X_train, y_train)

    def generate_signals(self):
        """
        Restituisce il DataFrame dei simboli che contiene i segnali
        per andare long, short o flat (1, -1 or 0).
        """
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0

        # Predizione del periodo successivo con il modello QDA
        signals['signal'] = self.model.predict(self.predictors)

        # Rimuove i primi 5 segnali per eliminare gli elementi
        # NaN nel DataFrame dei segnali
        signals['signal'][0:5] = 0.0
        signals['positions'] = signals['signal'].diff()

        return signals


class MarketIntradayPortfolio(Portfolio):
    """
    Acquista o vende 500 azioni di un'asset al prezzo di apertura di
    ogni barra, a seconda della direzione della previsione, chiude
    il trade alla chiusura della barra.

    Richiede:
    symbol - Un simbolo azionario che costituisce la base del portafoglio.
    bars - Un DataFrame di barre per un set di simboli.
    signals - Un DataFrame panda di segnali (1, 0, -1) per ogni simbolo.
    initial_capital - L'importo in contanti all'inizio del portafoglio.
    """

    def __init__(self, symbol, bars, signals, initial_capital=100000.0):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()

    def generate_positions(self):
        """
        Genera il DataFrame delle posizioni, basate sui segnali
        forniti dal DataFrame 'signals'.
        """
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)

        # Long o short di 500 azioni dello SPY basate sui
        # segnali direzionali giornalieri
        positions[self.symbol] = 500 * self.signals['signal']
        return positions

    def backtest_portfolio(self):
        """
        Backtest del portafoglio e restituisce un DataFrame contenente
        la curva equity e i precentuali dei rendimenti."""

        # Imposta l'oggetto Portfolio per avere lo stesso periodo
        # del DataFrame delle posizioni
        portfolio = pd.DataFrame(index=self.positions.index)
        pos_diff = self.positions.diff()

        # Calcola il profitto infragiornaliero della differenza tra
        # i prezzi di apertura e chiusura e quindi determina il
        # profitto giornaliero andando long se è previsto un giorno
        # positivo e short se è previsto un giorno negativo
        portfolio['price_diff'] = self.bars['Close'] - self.bars['Open']
        portfolio['price_diff'][0:5] = 0.0
        portfolio['profit'] = self.positions[self.symbol] * portfolio['price_diff']

        # Genera la curva equity e la percentuale dei rendimenti
        portfolio['total'] = self.initial_capital + portfolio['profit'].cumsum()
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio


if __name__ == "__main__":
    start_test = datetime.datetime(2005, 1, 1)
    end_period = datetime.datetime(2005, 12, 31)

    # Download le barre dello ETF SPY che rispecchia l'indice S&P500
    bars = pdr.DataReader("SPY", "yahoo", start_test, end_period)

    # Crea la strategia di previsione dell'S&P500
    snpf = SNPForecastingStrategy("^GSPC", bars)
    signals = snpf.generate_signals()

    # Crea il portafoglio basate sul predittore
    portfolio = MarketIntradayPortfolio("SPY", bars, signals,
                                        initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()

    # Stampa il grafico dei risultati
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    # Stampa il prezzo dell'ETF dello SPY
    ax1 = fig.add_subplot(211, ylabel='SPY ETF price in $')
    bars['Close'].plot(ax=ax1, color='r', lw=2.)

    # Stampa la curva di equity
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    returns['total'].plot(ax=ax2, lw=2.)

    fig.show()
    print("")