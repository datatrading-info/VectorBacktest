# https://datatrading.info/ambiente-di-backtesting-even-driven-con-python-parte-vii/

# performance.py

import numpy as np
import pandas as pd

# performance.py

def create_sharpe_ratio(returns, periods=252):
    """
    Crea il Sharpe ratio per la strategia, basato su a benchmark
    pari a zero (ovvero nessuna informazione sui tassi privi di rischio).

    Parametri:
    returns - Una serie panda che rappresenta i rendimenti percentuali nel periodo.
    periods - Giornaliero (252), orario (252 * 6,5), minuto (252 * 6,5 * 60) ecc.
    """
    return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)


def create_drawdowns(equity_curve):
    """
    Calcola il massimo drawdown tra il picco e il minimo della curva PnL
    così come la durata del drawdown. Richiede che il pnl_returns
    sia una serie di pandas.

    Parametri:
    pnl - Una serie pandas che rappresenta i rendimenti percentuali del periodo.

    Restituisce:
    Drawdown, duration - Massimo drawdown picco-minimo e relativa durata.
    """

    # Calcola la curva cumulativa dei rendimenti
    # e imposta un "High Water Mark"
    # Quindi crea le serie dei drawdown e relative durate
    hwm = [0]
    eq_idx = equity_curve.index
    drawdown = pd.Series(index = eq_idx)
    duration = pd.Series(index = eq_idx)

    # Loop over the index range
    for t in range(1, len(eq_idx)):
        cur_hwm = max(hwm[t-1], equity_curve[t])
        hwm.append(cur_hwm)
        drawdown[t]= hwm[t] - equity_curve[t]
        duration[t]= 0 if drawdown[t] == 0 else duration[t-1] + 1
    return drawdown.max(), duration.max()