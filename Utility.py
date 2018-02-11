"""
2/10/2018
mmelv
MACrossover
Utility
"""

import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader_gdax as pdr
from matplotlib import style
# TODO: Fix this deprecation
from matplotlib.finance import candlestick_ohlc


class Utility:
    """This is a class that is used to compartmentalize the utility methods of the program.
    Methods of this class get data from servers, plot graphs, and/or initialize DataFrames."""

    def __init__(self):
        pass

    @staticmethod
    def get_data_to_csv(crypto_currencies, grain, start, end):
        """Gets the data from GDAX and saves it to a csv file.
        :param crypto_currencies: A list of 'Tickers' from GDAX
        :type crypto_currencies: list
        :param grain: Time in seconds
        :type grain: int
        :param start: Start date
        :type start: object
        :param end: End date
        :type end: object
        """

        # Create DataFrames
        for crypto in crypto_currencies:
            df = pdr.get_data_gdax(crypto, granularity=grain, start=start, end=end)
            df.to_csv('data/{}.csv'.format(crypto))

    @staticmethod
    def get_data_frame(crypto):
        """Creates a pandas.DataFrame object for the given cryptocurrency.
        :param crypto: A 'Ticker' from GDAX
        :type crypto: list
        :return: Data for the cryptocurrency
        """

        df = pd.read_csv('data/{}.csv'.format(crypto), parse_dates=True, index_col=0)
        return df

    @staticmethod
    def plot_ohlc(crypto):
        """Creates OHLC and Volume Graphs of the cryptocurrency given.
        :param crypto: A 'Ticker' from GDAX
        :type crypto: list
        """

        style.use('ggplot')
        # Read in data from the csv files.
        do_util = Utility()
        df = do_util.get_data_frame(crypto)

        # TODO: Make it to where the user can select the OHLC sample size
        # Resamples the data. Gets values for OHLC every 3rd day.
        df_ohlc = df['Close'].resample('3D').ohlc()
        df_volume = df['Volume'].resample('3D').sum()

        # This is necessary to reformat the data to plot with matplotlib
        df_ohlc.reset_index(inplace=True)
        df_ohlc['index'] = df_ohlc['index'].map(mdates.date2num)

        # Set up graphs
        plt.rcParams['figure.figsize'] = [15, 11]  # Adjust size
        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
        ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=5, colspan=1, sharex=ax1)
        ax1.xaxis.set_visible(False)  # Don't need the date on the graph twice

        # Graph the OHLC and the Volume against dates
        candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
        ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

        # Make the graph prettier
        ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, borderaxespad=0.5)
        plt.title(crypto, y=7)
        plt.xlabel('Date')
        plt.ylabel('Price (USD)', y=4)

        plt.show()

    @staticmethod
    def generate_signal(crypto, short_moving_average, long_moving_average):
        """Creates  and returns a pandas.DataFrame object 'signals', that
        will produce a buy/sell signal
        :param crypto: A 'Ticker' from GDAX
        :type crypto: list
        :param short_moving_average: Number of days to calculate the Short-Moving-Average
        :type short_moving_average: int
        :param long_moving_average: Number of days to calculate the Short-Moving-Average
        :type long_moving_average: int
        :return: Signals DataFrame
        """

        short_window = short_moving_average

        # Create a dataframe for the buy/sell signals
        do_util = Utility()
        df = do_util.get_data_frame(crypto)
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0.0
        signals['Close'] = df['Close']

        # Create short moving average
        signals['short_mavg'] = df['Close'].rolling(window=short_moving_average,
                                                    min_periods=1).mean()
        # Create long moving average
        signals['long_mavg'] = df['Close'].rolling(window=long_moving_average,
                                                   min_periods=1).mean()
        # Create signals
        signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]
                                                    > signals['long_mavg'][short_window:],
                                                    1.0, 0.0)
        # Generate trade orders
        signals['positions'] = signals['signal'].diff()

        return signals

    # TODO: Fix the x-axis to show dates
    @staticmethod
    def plot_signals(signals, crypto):
        """Graphs the long and short moving averages.
        When the short moving average crosses above the long moving average from
        the bottom, a buy signal will be placed on the graph. When the short
        moving average crosses below the long moving average, a sell signal
        will be placed on the graph.
        :param signals: Signals DataFrame
        :type signals: pandas.DataFrame
        :param crypto: A 'Ticker' from GDAX
        :type crypto: list
        """

        # Make a graph
        style.use('ggplot')

        # Reset Index and convert to mdates
        signals.reset_index(inplace=True)
        signals['index'] = signals['index'].map(mdates.date2num)

        # Add subplot and axes labels
        plt.rcParams['figure.figsize'] = [15, 11]  # Adjust size
        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

        # Plot the closing price
        signals['Close'].plot(ax=ax1, color='r', lw=2.)

        # Plot the moving averages
        signals['short_mavg'].plot(ax=ax1, color='g', lw=2.)
        signals['long_mavg'].plot(ax=ax1, color='b', lw=2.)

        # Plot buy signals
        ax1.plot(signals.loc[signals.positions == 1.0].index,
                 signals.short_mavg[signals.positions == 1.0],
                 '^', markersize=10, color='m')

        # Plot sell signals
        ax1.plot(signals.loc[signals.positions == -1.0].index,
                 signals.short_mavg[signals.positions == -1.0],
                 'v', markersize=10, color='k')

        # TODO: Figure out why the labels and title are not showing up on the graph
        # Make the graph prettier
        plt.title(crypto, y=5)
        plt.xlabel('Date')
        plt.ylabel('Price (USD)', y=4)

        plt.show()

    @staticmethod
    def portfolio(initial_cash, buy_sell_amount, crypto, signals):
        """Creates and returns a pandas.DataFrame object 'portfolio'.
        This will tell us if we made any money or not.
        :param initial_cash: How much cash to start off with
        :type initial_cash: float
        :param buy_sell_amount: How many coins to buy/sell per signal
        :type buy_sell_amount: float
        :param crypto: A 'Ticker' from GDAX
        :type crypto: list
        :param signals: Signals DataFrame
        :type signals: pandas.DataFrame
        :return: portfolio
        :rtype: pandas.DataFrame
        """

        initial_cash = float(initial_cash)

        # Create a DataFrame
        positions = pd.DataFrame(index=signals.index).fillna(0.0)

        # Buy cryptos if 'signal' is buy, sell if signal is sell
        positions[crypto] = buy_sell_amount * signals['signal']

        # Initialize the portfolio with value owned
        portfolio = positions.multiply(signals['Close'], axis=0)

        # Store the difference
        position_diff = positions.diff()

        # Add 'holdings' , 'cash', 'total', and 'returns' to portfolio
        portfolio['holdings'] = (positions.multiply(signals['Close'],
                                                    axis=0)).sum(axis=1)
        portfolio['cash'] = initial_cash - \
                            (position_diff.multiply(signals['Close'],
                                                    axis=0)).sum(axis=1).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        print(portfolio.head())
        print(portfolio.tail())
        profit_loss = portfolio['total'].iloc[-1] - initial_cash
        if profit_loss < 0:
            print 'This strategy lost $' + str(profit_loss)
        elif profit_loss > 0:
            print 'This strategy profited $' + str(profit_loss)
        else:
            print 'This strategy neither made nor lost money.'

        return portfolio
