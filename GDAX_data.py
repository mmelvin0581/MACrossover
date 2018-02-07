"""
1/30/2018
mjmelvin@bu.edu
Final Project

This will grab cryptocurrency data fro GDAX, a cryptocurrency exchange
that is used by the popular Coinbase app. With this data, we will plot a time series
OHLC graph, with Volume. Next this implements a simple moving average crossover trading
strategy based on historical data from GDAX. Graphs of the closing price of the
cryptocurrency along with a short and long moving average will be produced. That graph
will have 'buy/sell' signals on it where the moving averages cross, hints the name moving
average crossover. Using this strategy, we will generate a mock portfolio and make
mock trades based on the 'buy/sell' signals produced, and see if we made or lost money.
"""

# TODO: Modularize more? Maybe use some classes?
import datetime as dt
import pandas_datareader_gdax as pdr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
# TODO: Fix this deprecation
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np


def get_data_to_csv(crypto_currencies, grain, start, end):
    """Gets the data from GDAX and saves it to a csv file."""

    # Create DataFrames
    for crypto in crypto_currencies:
        df = pdr.get_data_gdax(crypto, granularity=grain, start=start, end=end)
        df.to_csv('data/{}.csv'.format(crypto))

def get_data_frame(crypto):
    """Creates a pandas.DataFrame object for the given cryptocurrency."""
    df = pd.read_csv('data/{}.csv'.format(crypto), parse_dates=True, index_col=0)
    return df


def plot_ohlc(crypto):
    """Creates OHLC and Volume Graphs of the cryptocurrency given."""

    style.use('ggplot')
    # Read in data from the csv files.
    df = get_data_frame(crypto)

    # TODO: Maybe make it to where the user can select a sample size
    # Resamples the data. Gets values for OHLC every 3rd day.
    df_ohlc = df['Close'].resample('3D').ohlc()
    df_volume = df['Volume'].resample('3D').sum()

    # This is necessary to reformat the data to plot with matplotlib
    df_ohlc.reset_index(inplace=True)
    df_ohlc['index'] = df_ohlc['index'].map(mdates.date2num)

    # Set up graphs
    plt.rcParams['figure.figsize'] = [15, 11]    # Adjust size
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=5, colspan=1, sharex=ax1)
    ax1.xaxis.set_visible(False)    # Don't need the date on the graph twice

    # Graph the OHLC and the Volume against dates
    candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

    # Make the graph prettier
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, borderaxespad=0.5)
    plt.title(crypto, y=7)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)', y=4)

    plt.show()


def generate_signal(crypto, short_moving_average, long_moving_average):
    """Creates  and returns a pandas.DataFrame object 'signals', that
    will produce a buy/sell signal"""

    short_window = short_moving_average

    # Create a dataframe for the buy/sell signals
    df = get_data_frame(crypto)
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
def plot_signals(signals, crypto):
    """Graphs the closing price and the long and short moving averages.
    When the short moving average crosses above the long moving average from
    the bottom, a buy signal will be placed on the graph. When the short
    moving average crosses below the long moving average, a sell signal
    will be placed on the graph."""

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

    # TODO: The labels and title are not showing up on the graph
    # Make the graph prettier
    plt.title(crypto, y=5)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)', y=4)

    plt.show()


def portfolio(initial_cash, buy_sell_amount, crypto, signals):
    """Creates and returns a pandas.DataFrame object 'portfolio',
    This will basically tell us if we made any money or not."""

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


def visualize_portfolio(portfolio):
    # Create a figure
    fig = plt.figure()

    ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

    # Plot the equity curve in dollars
    portfolio['total'].plot(ax=ax1, lw=2.)

    ax1.plot(portfolio.loc[signals.positions == 1.0].index,
             portfolio.total[signals.positions == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(portfolio.loc[signals.positions == -1.0].index,
             portfolio.total[signals.positions == -1.0],
             'v', markersize=10, color='k')

    # Show the plot
    plt.show()


if __name__ == '__main__':

    # The Crypto Currencies available on Coinbase/GDAX
    cryptoList = ['BTC-USD', 'ETH-USD', 'LTC-USD']
    granularity = 24*3600  # time in seconds of date index

    # get_data_to_csv(cryptoList, granularity,
    #                  start=dt.datetime(2017, 1, 1), end=dt.datetime.today())

    # Prompt for cryptocurrency to buy
    while True:
        selection = input('Enter 1 for Bitcoin, 2 for Ethereum, or 3 for Litecoin: ')
        if selection not in (1, 2, 3):
            print 'You must select either 1, 2, or 3.'
            continue
        else:
            break

    # Get the strategy
    while True:
        try:
            # Prompt for short moving average
            short = input('Enter a time frame (days) for the Short Moving Average: ')

            # Prompt for long moving average
            long_ = input('Enter a time frame (days) for the Long Moving Average: ')

            # Prompt for initial cash in portfolio
            cash = input('How much cash to start with in your portfolio (USD)? ')

            # Prompt for buy sell amount
            buy_sell = input('How many coins would you buy/sell at a time? ')

            # This is where the ValueError may occur. The 'window' must be an integer.
            signals = generate_signal(cryptoList[selection - 1], short, long_)

        except ValueError:
            print 'The moving averages must be integer values. Try again please.'
            continue
        break

    plot_signals(signals, cryptoList[selection - 1])
    portfolio = portfolio(cash, buy_sell, cryptoList[selection - 1], signals)
    visualize_portfolio(portfolio)

    plot_ohlc(cryptoList[selection - 1])

    # TODO: Figure put how to export to a nice excel sheet
    portfolio.to_csv('portfolio/{}/{}_{}_{}_{}.csv'.format(cryptoList[selection-1],
                                                    short, long_, buy_sell, cash))
