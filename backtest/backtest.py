# backtest.py

# codice python relativo all'articolo presente su datatrading.info
# https://datatrading.info/sviluppo-di-un-backtesting-vettoriale-con-python-e-pandas/

from abc import ABCMeta, abstractmethod

class Strategy(object):
    """
    Strategy è una classe base astratta che fornisce un'interfaccia per
    tutte le strategie di trading successive (ereditate).

    L'obiettivo di un oggetto Strategy (derivato) è produrre un elenco di segnali,
    che ha la forma di un DataFrame pandas indicizzato di serie temporale.

    In questo caso è gestito solo un singolo simbolo / strumento.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_signals(self):
        """
        È necessaria un'implementazione per restituire il DataFrame dei simboli
        contenente i segnali per andare long, short o flat (1, -1 o 0)
        """
        raise NotImplementedError("Should implement generate_signals()!")


class Portfolio(object):
    """
    Una classe base astratta che rappresenta un portfolio di
    posizioni (inclusi strumenti e contanti), determinate
    sulla base di una serie di segnali forniti da una Strategy
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_positions(self):
        """
        Fornisce la logica per determinare come le posizioni del
        portafoglio sono allocate sulla base dei segnali
        previsionali e dei contanti disponibili
        """
        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def backtest_portfolio(self):
        """
        Fornisce la logica per generare gli ordini di trading
        e la successiva curva del patrimonio netto (ovvero la
        crescita del patrimonio netto totale), come somma
        di partecipazioni e contanti, e il periodo delle barre
        associato a questa curva in base al DataFrame delle "posizioni".

        Produce un oggetto portfolio che può essere esaminato da
        altre classi / funzioni.
        """
        raise NotImplementedError("Should implement backtest_portfolio()!")