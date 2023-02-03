# ma_cross.py

# codice python relativo all'articolo presente su datatrading.info

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas_datareader.data import DataReader
from backtest.backtest import Strategy, Portfolio


class MovingAverageCrossStrategy(Strategy):
    """
    Richiede:
    symbol - Un simbolo di un titolo azionario su cui formare una strategia.
    bars - Un DataFrame di barre per il simbolo.
    short_window - Periodo di ricerca per media mobile breve.
    long_window - Periodo di ricerca per media mobile lunga.
    """

    def __init__(self, symbol, bars, short_window=100, long_window=400):
        self.symbol = symbol
        self.bars = bars

        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        """
        Restituisce il DataFrame dei simboli che contiene i segnali
        per andare long, short o flat (1, -1 o 0).
        """
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0

        # Crea l'insieme di medie mobili semplici di breve e di
        # lungo periodo
        signals['short_mavg'] = self.bars['Close'].rolling(self.short_window).mean()
        signals['long_mavg'] = self.bars['Close'].rolling(self.long_window).mean()

        # Crea un "segnale" (investito o non investito) quando la media mobile corta incrocia la media
        # mobile lunga, ma solo per il periodo maggiore della finestra della media mobile più breve
        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:]
            > signals['long_mavg'][self.short_window:], 1.0, 0.0)

        # Si calcola la differenza dei segnali per generare gli effettivi ordini di trading
        signals['positions'] = signals['signal'].diff()

        return signals


class MarketOnClosePortfolio(Portfolio):
    """
    Incapsula la nozione di un portafoglio di posizioni basato
    su una serie di segnali forniti da una strategia.

    Richiede:
    symbol - Un simbolo di un titolo azionario che costituisce la base del portafoglio.
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
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = 100 * self.signals['signal']  # Questa strategia compra 100 azioni
        return positions

    def backtest_portfolio(self):
        portfolio = pd.DataFrame(index=self.signals.index).fillna(0.0)
        pos_diff = self.positions[self.symbol].diff()

        portfolio['holdings'] = (self.positions[self.symbol] * self.bars['Close'])
        portfolio['cash'] = self.initial_capital - (pos_diff * self.bars['Close']).cumsum()

        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio


if __name__ == "__main__":
    # Download delle barre giornaliere di AAPL da Yahoo Finance per il periodo
    # Dal 1 ° gennaio 1990 al 1 ° gennaio 2002 - Questo è un esempio tratto da ZipLine
    symbol = 'AAPL'
    bars = DataReader(symbol, "yahoo", datetime.datetime(1990, 1, 1), datetime.datetime(2002, 1, 1))

    # Crea un'istanza della classe MovingAverageCrossStrategy con un periodo della media
    # mobile breve pari a 100 giorni e un periodo per la media lunga pari a 400 giorni
    mac = MovingAverageCrossStrategy(symbol, bars, short_window=100, long_window=400)
    signals = mac.generate_signals()

    # Crea un portofoglio per AAPL, con $100,000 di capitale iniziale
    portfolio = MarketOnClosePortfolio(symbol, bars, signals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()

    # Visualizza 2 grafici per i trade e la curva di equity
    fig = plt.figure()
    fig.patch.set_facecolor('white')  # Imposta il colore di fondo a bianco
    ax1 = fig.add_subplot(211, ylabel='Price in $')

    # Visualizza il grafico dei prezzi di chiusura di AAPL con la media mobile
    bars['Close'].plot(ax=ax1, color='r', lw=2.)
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

    # Visualizza i trade "buy" su AAPL
    ax1.plot(signals.loc[signals.positions == 1.0].index,
             signals.short_mavg[signals.positions == 1.0],
             '^', markersize=10, color='m')

    # Visualizza i trade "sell" su AAPL
    ax1.plot(signals.loc[signals.positions == -1.0].index,
             signals.short_mavg[signals.positions == -1.0],
             'v', markersize=10, color='k')

    # Visualizza la curva di equity in dollari
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    returns['total'].plot(ax=ax2, lw=2.)

    # Visualizza i trade "buy" e "sell" su la curva di equity
    ax2.plot(returns.loc[signals.positions == 1.0].index,
             returns.total[signals.positions == 1.0],
             '^', markersize=10, color='m')
    ax2.plot(returns.loc[signals.positions == -1.0].index,
             returns.total[signals.positions == -1.0],
             'v', markersize=10, color='k')

    # Stampa il grafico
    fig.show()

    print("")