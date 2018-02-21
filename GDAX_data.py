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
A graph of the portfolios value in USD along with the buy/sell signals is produced.
"""

import matplotlib.pyplot as plt
import Utility
import datetime as dt


# TODO: Fix the x-axis to show the dates
def visualize_portfolio(port):
    """Plots the portfolio value in USD along with the buy/sell signals.
    :param port: portfolio DataFrame
    :type port: pandas.DataFrame
    """

    # Create a figure
    fig = plt.figure()

    ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

    # Plot the equity curve in dollars
    port['total'].plot(ax=ax1, lw=2.)

    ax1.plot(port.loc[signals.positions == 1.0].index,
             port.total[signals.positions == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(port.loc[signals.positions == -1.0].index,
             port.total[signals.positions == -1.0],
             'v', markersize=10, color='k')

    # Show the plot
    plt.show()


if __name__ == '__main__':

    # The Crypto Currencies available on Coinbase/GDAX
    cryptoList = ['BTC-USD', 'ETH-USD', 'LTC-USD']
    granularity = 24 * 3600  # time in seconds of date index

    # Initialize Utility class
    do_utilities = Utility.Utility()
    # TODO: Call this function prior to the presentation to update the data
    do_utilities.get_data_to_csv(cryptoList, granularity,
                     start=dt.datetime(2017, 1, 1), end=dt.datetime.today())

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
            signals = do_utilities.generate_signal(cryptoList[selection - 1], short, long_)

        except ValueError:
            print 'The moving averages must be integer values. Try again please.'
            continue
        break

    do_utilities.plot_ohlc(cryptoList[selection - 1])
    do_utilities.plot_signals(signals, cryptoList[selection - 1])
    portfolio = do_utilities.portfolio(cash, buy_sell, cryptoList[selection - 1], signals)
    visualize_portfolio(portfolio)
