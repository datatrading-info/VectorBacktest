# random_forecast.py

import numpy as np
import pandas as pd
import quandl  # Necessary for obtaining financial data easily

from backtest import Strategy, Portfolio


class RandomForecastingStrategy(Strategy):
    """
    Classe derivata da Strategy per produrre un insieme di segnali che
    sono long / short generati casualmente. Chiaramente una strategia non corretta, ma perfettamente accettabile per dimostrare il
    infrastruttura di backtest!
    """

    def __init__(self, symbol, bars):
        """Necessita del ticker del simbolo e il dataframe delle barre"""
        self.symbol = symbol
        self.bars = bars

    def generate_signals(self):
        """Creazione del DataFrame pandas dei segnali random."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = np.sign(np.random.randn(len(signals)))

        # I primi cinque elementi sono impostati a zero in modo da minimizzare
        # la generazione di errori NaN nella previsione.
        signals['signal'][0:5] = 0.0
        return signals

    # random_forecast.py

class MarketOnOpenPortfolio(Portfolio):
    """
    Eredita la classe Portfolio per creare un sistema che acquista 100 unità di
    uno specifico simbolo per un segnale long / short, utilizzando il prezzo
    open di una barra.

    Inoltre, non ci sono costi di transazione e il denaro può essere immediatamente
    preso in prestito per vendita allo scoperto (nessuna registrazione di
    margini o requisiti di interesse).

    Richiede:
    symbol - Un simbolo di una azione che costituisce la base del portafoglio.
    bars - Un DataFrame di barre per un simbolo.
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
        Crea un DataFrame "positions" che semplicemente è long o short
        per 100 pezzi di uno specfico simbolo basato sui segnali di
        previsione di {1, 0, -1} dal DataFrame dei segnali.
        """
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = 100 * self.signals['signal']
        return positions


    def backtest_portfolio(self):
        """
        Costruisce un portafoglio a partire dal Dataframe delle posizioni
        assumendo la capacità di negoziare all'esatto prezzo open di ogni
        barra (un'ipotesi irrealistica!).

        Calcola il totale della liquidità e delle partecipazioni (prezzo
        di mercato di ogni posizione per barra), al fine di generare una
        curva di equity ('totale') e una serie di rendimenti basati sulle
        barre ('ritorni').

        Restituisce l'oggetto portfolio da utilizzare altrove.
        """

        # Costruzione di un DataFrame 'portfolio' che usa lo stesso indice
        # delle "posizioni" e con una serie di "ordini di trading"
        # nell'oggetto 'pos_diff', assumendo prezzi open.

        portfolio = self.positions * self.bars['Open']
        pos_diff = self.positions.diff()

        # Crea le serie "holding" e "cash" scorrendo il dataframe
        # delle operazioni e aggiungendo / sottraendo la relativa quantità di
        # ogni colonna
        portfolio['holdings'] = (self.positions * self.bars['Open']).sum(axis=1)
        portfolio['cash'] = self.initial_capital - (pos_diff * self.bars['Open']).sum(axis=1).cumsum()

        # Finalizza i rendimenti totali e basati su barre in base al "contante"
        # e dati sulle "partecipazioni" per il portafoglio
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio


if __name__ == "__main__":
    # Ottenere le barre giornaliere di SPY (ETF che generalmente
    # segue l'S&P500) da Quandl (richiede 'pip install Quandl'
    # sulla riga di comando)
    symbol = 'SPY'
    bars = quandl.get("GOOG/NYSE_%s" % symbol, collapse="daily")

    # Crea un insieme di segnali randome per SPY
    rfs = RandomForecastingStrategy(symbol, bars)
    signals = rfs.generate_signals()

    # Crea un portfolio di SPY
    portfolio = MarketOnOpenPortfolio(symbol, bars, signals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()

    print(returns.tail(10))