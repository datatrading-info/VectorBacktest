# mr_spy_iwm.py

import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

sns.set_style("darkgrid")


def create_pairs_dataframe(datadir, symbols):
    """
    Crea un DataFrame pandas che contiene i prezzi di chiusura di una
    coppia di simboli a partire da file CSV che contiene un datatime e
    i dati OHLCV.

    Parameters
    ----------
    datadir : `str`
        Directory dove sono archiviati file CSV che contengono i dati OHLCV.
    symbols : `tup`
        Tuple contenente i simboli ticker come `str`.

    Returns
    -------
    pairs : `pd.DataFrame`
        Un DataFrame contanente i prezzi di chiusura per SPY e IWM. L'indice è un
        oggetto Datetime.
    """
    # Apre i file CSV individualmente e legge il contenuto in un DataFrames pandas
    # usando la prima colonna come un indice e col_names per gli headers

    print("Importing CSV data...")
    col_names = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'na']
    sym1 = pd.read_csv(
        os.path.join(datadir, '%s.csv' % symbols[0]),
        header=0,
        index_col=0,
        names=col_names
    )
    sym2 = pd.read_csv(
        os.path.join(datadir, '%s.csv' % symbols[1]),
        header=0,
        index_col=0,
        names=col_names
    )

    # Crea un DataFrame pandas con i prezzi di chiusura per ogni simbolo
    # correttamente allineate e elimenanto gli elementi mancanti
    print("Constructing dual matrix for %s and %s..." % symbols)
    pairs = pd.DataFrame(index=sym1.index)
    pairs['%s_close' % symbols[0].lower()] = sym1['close']
    pairs['%s_close' % symbols[1].lower()] = sym2['close']
    pairs.index = pd.to_datetime(pairs.index)
    pairs = pairs.dropna()
    return pairs


def calculate_spread_zscore(pairs, symbols, lookback=100):
    """
    Crea un hedge ratio tra i due simboli calcolando una regressione
    lineare mobile con uno specifico periodo di lookback. Questa è
    usata per creare uno z-score dello 'spread' tra i due simboli
    basato da una combinazione lineare dei due.

    Parameters
    ----------
    pairs : `pd.DataFrame`
        Un DataFrame contanente i prezzi di chiusura per SPY e IWM. L'indice è un
        oggetto Datetime.
    symbols : `tup`
        Tuple contenente i simboli ticker come `str`.
    lookback : `int`, optional (default: 100)
        Periodo di Lookback per la regressione lineare mobile.

    Returns
    -------
    pairs : 'pd.DataFrame'
        Aggiornamento del DataFrame con lo spred e lo z score tra i
        due simboli basati sulla regressione lineare mobile.
    """

    # Uso del metodo Rolling Ordinary Least Squares di statsmodels per allenare
    # una regressione lineare mobile tra le due serie temporali dei prezzi di chiusura
    print("Fitting the rolling Linear Regression...")

    model = RollingOLS(
        endog=pairs['%s_close' % symbols[0].lower()],
        exog=sm.add_constant(pairs['%s_close' % symbols[1].lower()]),
        window=lookback
    )
    rres = model.fit()
    params = rres.params.copy()

    # Costruzione del hedge ratio ed eliminazione del primo elemento della
    # finestra di lookbackand vuoto/NaN
    pairs['hedge_ratio'] = params['iwm_close']
    pairs.dropna(inplace=True)

    # Crea uno spread e quindi uno z-score dello spread
    print("Creating the spread/zscore columns...")
    pairs['spread'] = (pairs['spy_close'] - pairs['hedge_ratio'] * pairs['iwm_close'])
    pairs['zscore'] = (pairs['spread'] - np.mean(pairs['spread'])) / np.std(pairs['spread']
                                                                                   )
    return pairs


def create_long_short_market_signals(pairs, symbols, z_entry_threshold=2.0, z_exit_threshold=1.0):
    """
    Crea i segnali di entrata/uscita in base al superamento di z_entry_threshold
    per entrare in una posizione e al di sotto di z_exit_threshold per
    uscire da una posizione.

    Parameters
    ----------
    pairs : `pd.DataFrame`
        DataFrame aggiornato contenente il prezzo di chiusura, lo spread
        e il punteggio z tra i due simboli.
    symbols : `tup`
        Tupla contenente simboli ticker come `str`.
    z_entry_threshold : `float`, optional (default:2.0)
        Soglia di punteggio Z per l'ingresso nel mercato.
    z_exit_threshold : `float`, optional (default:1.0)
        Soglia di punteggio Z per l'uscita dal mercato.

    Returns
    -------
    pairs : `pd.DataFrame`
        DataFrame aggiornato contenente segnali long, short e di uscita
    """

    # Calcola quando essere long, short e quando uscire
    pairs['longs'] = (pairs['zscore'] <= -z_entry_threshold)*1.0
    pairs['shorts'] = (pairs['zscore'] >= z_entry_threshold)*1.0
    pairs['exits'] = (np.abs(pairs['zscore']) <= z_exit_threshold)*1.0

    # Questi segnali sono necessari perché dobbiamo propagare
    # una posizione in avanti, ovvero dobbiamo rimanere long se
    # la soglia zscore è inferiore a z_entry_threshold di ancora
    # maggiore di z_exit_threshold, e viceversa per short.
    pairs['long_market'] = 0.0
    pairs['short_market'] = 0.0

    # Queste variabili tracciano se essere long o short
    # durante l'iterazione tra le barre
    long_market = 0
    short_market = 0

    # Poiché utilizza iterrows per eseguire il loop su un dataframe,
    # sarà significativamente meno efficiente di un'operazione vettorializzata,
    # cioè più lenta!
    print("Calculating when to be in the market (long and short)...")
    for i, b in enumerate(pairs.iterrows()):
        # Calcola i long
        if b[1]['longs'] == 1.0:
            long_market = 1
        # Calcola gli short
        if b[1]['shorts'] == 1.0:
            short_market = 1
        # Calcola le uscite
        if b[1]['exits'] == 1.0:
            long_market = 0
            short_market = 0

        # Assegna direttamente un 1 o 0 alle colonne long_market/short_market,
        # in modo tale che la strategia sappia quando effettivamente entrare!
        pairs.iloc[i]['long_market'] = long_market
        pairs.iloc[i]['short_market'] = short_market
    return pairs


def create_portfolio_returns(pairs, symbols):
    """
    Crea un DataFrame pandas di portafoglio che tiene traccia del
    capitale dell'account e alla fine genera una curva di equity.
    Questo può essere utilizzato per generare drawdown e rapporti
    rischio/rendimento.

    Parameters
    ----------
    pairs : `pd.DataFrame`
        DataFrame aggiornato contenente il prezzo di chiusura, lo spread
        e il punteggio z tra i due simboli e i segnali long, short e uscita.
    symbols : `tup`
        Tupla contenente simboli ticker come `str`.

    Returns
    -------
    portfolio : 'pd.DataFrame'
        Un DataFrame con l'indice datetime del DataFrame dei pairs, le posizioni,
        il valore di mercato totale e rendimenti.
    """

    # Variabili di convenzione per i simboli
    sym1 = symbols[0].lower()
    sym2 = symbols[1].lower()

    # Crea l'oggetto portfolio con le informazioni sulle posizioni
    # Notare la sottrazione per tenere traccia degli short!
    print("Constructing a portfolio...")
    portfolio = pd.DataFrame(index=pairs.index)
    portfolio['positions'] = pairs['long_market'] - pairs['short_market']
    portfolio[sym1] = -1.0 * pairs['%s_close' % sym1] * portfolio['positions']
    portfolio[sym2] = pairs['%s_close' % sym2] * portfolio['positions']
    portfolio['total'] = portfolio[sym1] + portfolio[sym2]

    # Crea un flusso di rendimenti percentuali ed elimina
    # tutte le celle NaN e -inf/+inf
    print("Constructing the equity curve...")
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['returns'].fillna(0.0, inplace=True)
    portfolio['returns'].replace([np.inf, -np.inf], 0.0, inplace=True)
    portfolio['returns'].replace(-1.0, 0.0, inplace=True)

    # Calcola la curva di equity
    portfolio['returns'] = (portfolio['returns'] + 1.0).cumprod()
    return portfolio

if __name__ == "__main__":
    datadir = '/your/path/to/data/'  # Da modificare
    symbols = ('SPY', 'IWM')

    lookbacks = range(50, 210, 10)
    returns = []

    # Regola il periodo di ricerca da 50 a 200 con
    # incrementi di 10 per produrre sensibilità
    for lb in lookbacks:
        print("Calculating lookback=%s..." % lb)
        pairs = create_pairs_dataframe(datadir, symbols)
        pairs = calculate_spread_zscore(pairs, symbols, lookback=lb)
        pairs = create_long_short_market_signals(
            pairs, symbols, z_entry_threshold=2.0, z_exit_threshold=1.0
        )
        portfolio = create_portfolio_returns(pairs, symbols)
        returns.append(portfolio.iloc[-1]['returns'])

    print("Plot the lookback-performance scatterchart...")
    plt.plot(lookbacks, returns, '-o')
    plt.show()

    # Questo è ancora nella funzione main
    pairs = create_pairs_dataframe(datadir, symbols)
    pairs = calculate_spread_zscore(pairs, symbols, lookback=100)
    pairs = create_long_short_market_signals(
        pairs, symbols, z_entry_threshold=2.0, z_exit_threshold=1.0
    )
    portfolio = create_portfolio_returns(pairs, symbols)

    print("Plotting the performance charts...")
    fig = plt.figure()

    ax1 = fig.add_subplot(211,  ylabel='%s growth (%%)' % symbols[0])
    (pairs['%s_close' % symbols[0].lower()].pct_change()+1.0).cumprod().plot(ax=ax1, color='r', lw=2.)

    ax2 = fig.add_subplot(212, ylabel='Portfolio value growth (%%)')
    portfolio['returns'].plot(ax=ax2, lw=2.)

    plt.show()