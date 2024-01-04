import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import numpy as np
from credentials import login, password, server
import multiprocessing

# Color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"


# Initialize MetaTrader 5 connection
def initialize():
    if not mt5.initialize():
        print("Failed to initialize MetaTrader 5. Please check your connection.")
        return False

    # Login to your MetaTrader 5 account
    authorized = mt5.login(login=login, password=password, server=server)
    if not authorized:
        print("Failed to login to MetaTrader 5. Please check your credentials.")
        mt5.shutdown()
        return False

    return True


# Get symbol information
def get_symbol_info(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol_info for {symbol}")
        return None
    return symbol_info


# Get exposure of a symbol
def get_exposure(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        pos_df = pd.DataFrame(positions, columns=positions[0]._asdict().keys())
        exposure = pos_df['volume'].sum()
        return exposure
    return 0


def calculate_atr(symbol, timeframe):
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, 14)  # Adjust the period as needed
    close_prices = np.array(bars['close'])
    high_prices = np.array(bars['high'])
    low_prices = np.array(bars['low'])

    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], np.abs(high_prices[1:] - close_prices[:-1]),
                             np.abs(low_prices[1:] - close_prices[:-1]))
    atr = np.mean(true_ranges)  # Calculate the mean of true ranges

    return atr



def calculate_adx(symbol, timeframe):
    # Request ADX data
    adx_data = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)

    # Check if ADX data is received successfully
    if adx_data is not None:
        adx_values = [entry[6] for entry in adx_data]
        return np.mean(adx_values)  # Calculate the mean of ADX values
    else:
        print(f"No ADX data available for symbol: {symbol} and timeframe: {timeframe}")
        return None


def calculate_directional_movement_index(high, low):
    # Calculation logic for Directional Movement Index (DX)
    # Replace with the actual calculation code
    dx = np.random.rand(len(high))
    return dx


def calculate_ichimoku_cloud(symbol, high, low):
    timeframe= mt5.TIMEFRAME_M1
    conversion_line = (max(high[:9]) + min(low[:9])) / 2
    base_line = (max(high[10:26]) + min(low[10:26])) / 2
    leading_span_a = (conversion_line + base_line) / 2
    leading_span_b = (max(high[27:52]) + min(low[27:52])) / 2
    lagging_span = high[-26]

    kijun_sen = (max(high[10:26]) + min(low[10:26])) / 2
    kijun_sen_previous = (max(high[9:25]) + min(low[9:25])) / 2
    conversion_line_previous = (max(high[8:24]) + min(low[8:24])) / 2

    atr = calculate_atr(symbol, timeframe)

    adx = calculate_adx(symbol, timeframe)

    if kijun_sen > conversion_line and kijun_sen_previous < conversion_line_previous:
        return 'Sell'
    elif kijun_sen < conversion_line and kijun_sen_previous > conversion_line_previous:
        return 'Buy'
    elif ((leading_span_a < leading_span_b and kijun_sen > conversion_line and kijun_sen < leading_span_a) or
          (leading_span_a > leading_span_b and kijun_sen < conversion_line and kijun_sen > leading_span_a)) :
        return 'No Trend'
    else:
        return 'No Signal'




def open_order(symbol, volume, order_type, magic_number, comment):
    print("Reached OpenOrder Function for:", order_type)
    time.sleep(0.1)  # Delay for tick data update
    tick = mt5.symbol_info_tick(symbol)
    order_dict = {'buy': mt5.ORDER_TYPE_BUY, 'sell': mt5.ORDER_TYPE_SELL}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_dict[order_type],
        "price": price_dict[order_type],
        "magic": magic_number,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": f"{magic_number}-{comment}",
    }
    result = mt5.order_send(request)
    if result is None:
        print("Order placement failed. Error: order_result is None")
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Order placement failed. Error:", result.comment)
    else:
        print("Order opened successfully.")
    return result

# Close an order
def close_order(ticket, magic_number):
    positions = mt5.positions_get()

    for pos in positions:
        if pos.ticket == ticket and pos.magic == magic_number:
            print(pos.ticket, pos.magic)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": 1 if pos.type == 0 else 0,  # Close the opposite type of position
                "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 20,
                "magic": magic_number,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            try:
                order_result = mt5.order_send(request)
                if order_result is None:
                    raise Exception(f"Order placement failed. Symbol: {pos.symbol}, Ticket: {pos.ticket}. Error: order_result is None")
                if order_result.retcode != mt5.TRADE_RETCODE_DONE:
                    raise Exception(f"Order placement failed. Symbol: {pos.symbol}, Ticket: {pos.ticket}. Error: {order_result.comment}")
                print("Order closed successfully.")
                return order_result
            except Exception as e:
                print(f"Error occurred while closing order: {str(e)}")
                return None
    return None




def calculate_ichimoku_span_b(symbol, conversion_period, base_period, span_b_period):
    highs = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, conversion_period + span_b_period - 1)['high']
    lows = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, conversion_period + span_b_period - 1)['low']
    conversion_line = (max(highs[:conversion_period]) + min(lows[:conversion_period])) / 2
    base_line = (max(highs[:base_period]) + min(lows[:base_period])) / 2
    span_b = (max(highs[:span_b_period]) + min(lows[:span_b_period])) / 2
    return span_b

def calculate_ichimoku_span_a(symbol):
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, 26)
    bars = np.array(bars, dtype=np.dtype([('time', '<M8[s]'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<i8'), ('spread', '<i4'), ('real_volume', '<i8')]))

    high_prices = bars['high']
    low_prices = bars['low']
    span_a = sum(high_prices[:9]) / 9 + sum(low_prices[:9]) / 9

    return span_a

def modify_sltp(symbol, magic_number):
    positions = mt5.positions_get(symbol=symbol, magic=magic_number)
    timeframe = mt5.TIMEFRAME_M1
    symbol_info = mt5.symbol_info(symbol)
    conversion_period = 9
    base_period = 26
    span_b_period = 52

    if not positions or symbol_info is None:
        print(f"No positions found for symbol: {symbol}")
        return

    ichimoku_span_b = calculate_ichimoku_span_b(symbol, conversion_period, base_period, span_b_period)
    ichimoku_span_a = calculate_ichimoku_span_a(symbol)
    atr = calculate_atr(symbol, timeframe)

    trailing_stop_distance = round(atr * 3, symbol_info.digits)

    for position in positions:
        if position.type == mt5.ORDER_TYPE_BUY and position.price_open < ichimoku_span_a:
            order_type = mt5.ORDER_TYPE_BUY
            stop_loss = round(ichimoku_span_a, symbol_info.digits)
            take_profit = round(position.price_open + (1000 * symbol_info.point), symbol_info.digits)
            trailing_stop = round(position.price_open - trailing_stop_distance * symbol_info.point, symbol_info.digits)
            if position.price_current > position.price_open:
                trailing_stop = round(position.price_current - trailing_stop_distance * symbol_info.point, symbol_info.digits)
        elif position.type == mt5.ORDER_TYPE_SELL and position.price_open > ichimoku_span_a:
            order_type = mt5.ORDER_TYPE_SELL
            stop_loss = round(ichimoku_span_b, symbol_info.digits)
            take_profit = round(position.price_open - (1000 * symbol_info.point), symbol_info.digits)
            trailing_stop = round(position.price_open + trailing_stop_distance * symbol_info.point, symbol_info.digits)
            if position.price_current < position.price_open:
                trailing_stop = round(position.price_current + trailing_stop_distance * symbol_info.point, symbol_info.digits)
        else:
            continue

        sltp_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "sl": stop_loss,
            "tp": take_profit,
            "price": mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask,
            "magic": magic_number,
            "comment": "Modify SL/TP",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if trailing_stop_distance > 0:
            sltp_request["ts"] = trailing_stop

        result = mt5.order_send(sltp_request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to modify SL/TP for ticket {position.ticket}. Error code: {result.retcode}. Error description: {result.comment}")

def modify_sltp_n(symbol, magic_number, trailing_stop_distance):
    positions = mt5.positions_get(symbol=symbol, magic=magic_number)
    symbol_info = mt5.symbol_info(symbol)
    if not positions or symbol_info is None:
        print(f"No positions found for symbol: {symbol}")
        return

    for position in positions:
        stop_loss = position.price_open - trailing_stop_distance
        take_profit = position.price_open + (trailing_stop_distance * 3)

        sltp_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": position.ticket,
            "sl": stop_loss,
            "tp": take_profit,
            "magic": magic_number,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(sltp_request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to modify SL/TP for ticket {position.ticket}. Error code: {result.retcode}. Error description: {result.comment}")

def trade_symbols(symbols, volume, timeframe):
    signal_history = {symbol: [] for symbol in symbols}
    minimum_distance = 500

    while True:
        for symbol in symbols:
            symbol_info = get_symbol_info(symbol)
            if symbol_info is None:
                return

            pip_value = symbol_info.point
            digits = symbol_info.digits
            magic_number = MAGIC_NUMBERS.get(symbol)

            exposure = get_exposure(symbol)

            bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, 53)
            if bars is not None and len(bars) > 0:
                high = np.array(bars['high'])
                low = np.array(bars['low'])
                ichimoku_signal = calculate_ichimoku_cloud(symbol, high, low)
            else:
                ichimoku_signal = 'No Signal'

            if ichimoku_signal != 'No Signal' or ichimoku_signal != 'No Trend':
                positions = mt5.positions_get(symbol=symbol, magic=magic_number)
                positions_total = len(positions)
                sell_positions_total = len([pos for pos in positions if pos.type == mt5.ORDER_TYPE_SELL])
                buy_positions_total = len([pos for pos in positions if pos.type == mt5.ORDER_TYPE_BUY])
                sell_positions = [pos for pos in positions if pos.type == mt5.ORDER_TYPE_SELL]
                buy_positions = [pos for pos in positions if pos.type == mt5.ORDER_TYPE_BUY]

                lowest_buy_price = min(pos.price_open for pos in buy_positions) if buy_positions else 0
                highest_sell_price = max(pos.price_open for pos in sell_positions) if sell_positions else float('inf')

                average_sell_price = 0
                if sell_positions_total > 0:
                    average_sell_price = np.average([pos.price_open for pos in sell_positions],
                                                    weights=[pos.volume for pos in sell_positions])

                average_buy_price = 0
                if buy_positions_total > 0:
                    average_buy_price = np.average([pos.price_open for pos in buy_positions],
                                                   weights=[pos.volume for pos in buy_positions])

                pip_size = symbol_info.point
                profit_target = 10

                if sell_positions_total == 1:
                    for sell_pos in sell_positions:
                        position_profit = sell_pos.profit
                        print(sell_pos.ticket, ':::', sell_pos.price_open, ':::', average_sell_price, ':::',
                              position_profit)
                        if position_profit > profit_target * 2:
                            close_order(sell_pos.ticket, magic_number)

                if buy_positions_total == 1:
                    for buy_pos in buy_positions:
                        position_profit = buy_pos.profit
                        print(buy_pos.ticket, ':::', buy_pos.price_open, ':::', average_buy_price, ':::',
                              position_profit)
                        if position_profit > profit_target * 2:
                            close_order(buy_pos.ticket, magic_number)

                if sell_positions_total > 1:
                    for sell_pos in sell_positions:
                        #profit_in_pips = (sell_pos.price_open - symbol_info.bid) / pip_size
                        position_profit = sell_pos.profit
                        ##print(sell_pos.ticket, ':::', sell_pos.price_open, ':::', average_sell_price, ':::', position_profit)
                        if position_profit > profit_target  and symbol_info.bid < average_sell_price:
                            print(sell_pos.ticket, ':::', sell_pos.price_open, ':::', average_sell_price, ':::', position_profit)
                            close_order(sell_pos.ticket, magic_number)
                        if position_profit > profit_target*2:
                            close_order(sell_pos.ticket, magic_number)
                        if ichimoku_signal == 'buy' and position_profit > 0:
                            close_order(sell_pos.ticket, magic_number)

                if buy_positions_total > 1:
                    for buy_pos in buy_positions:
                        #profit_in_pips = (symbol_info.ask - buy_pos.price_open) / pip_size
                        position_profit = buy_pos.profit
                        print(buy_pos.ticket, ':::', buy_pos.price_open, ':::', average_buy_price, ':::',position_profit)
                        if position_profit > profit_target and symbol_info.ask > average_buy_price:
                            print(buy_pos.ticket, ':::', buy_pos.price_open, ':::', average_buy_price, ':::',position_profit)
                            close_order(buy_pos.ticket, magic_number)
                        if position_profit > profit_target*2:
                            close_order(buy_pos.ticket, magic_number)
                        if ichimoku_signal == 'Sell' and position_profit > 0:
                            close_order(buy_pos.ticket, magic_number)



                if ichimoku_signal == 'Buy':
                    if buy_positions_total == 0 or (symbol_info.ask < lowest_buy_price and
                                                    lowest_buy_price - symbol_info.ask >= minimum_distance * pip_size):
                        open_order(symbol, volume, 'buy', magic_number, 'Ichimoku')
                        #lowest_buy_price = symbol_info.ask
                elif ichimoku_signal == 'Sell':
                    if sell_positions_total == 0 or (symbol_info.bid > highest_sell_price and
                                                     symbol_info.bid - highest_sell_price >= minimum_distance * pip_size):
                        open_order(symbol, volume, 'sell', magic_number, 'Ichimoku')
                        #highest_sell_price = symbol_info.bid

                atr = calculate_atr(symbol, timeframe)
                trailing_stop_distance = round(atr * 3, symbol_info.digits)

                signal_history[symbol].append(ichimoku_signal)

                time.sleep(1)

                print(f"time: \033[1m{datetime.now()}\033[0m Symbol: \033[1m{symbol}\033[0m")
                print(f"exposure: {exposure}")
                print(f"ichimoku_signal: \033[1m{ichimoku_signal}\033[0m")
                print(f"positions_total: {positions_total}")
                print(f"lowest_buy_price: {lowest_buy_price}")
                print(f"highest_sell_price: {highest_sell_price}")
                print(
                    f"\033[91msell total: {sell_positions_total} sell breakeven: {round(average_sell_price, digits)}\033[0m")
                print(
                    f"\033[94mbuy total: {buy_positions_total} buy breakeven: {round(average_buy_price, digits)}\033[0m")
                print(f"ATR: {atr}")
                print(f"Trailing Stop Distance: {trailing_stop_distance}")
                print('-------')

                #modify_sltp(symbol, magic_number)

            time.sleep(2)

def backtest_strategy(symbol):
    try:
        timeframe = mt5.TIMEFRAME_M1  # Use M1 timeframe for backtesting
        minimum_distance = 50
        # Calculate the start and end date for the desired historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=350)  # Adjust the number of days as needed
        # Initialize MetaTrader 5 connection
        mt5.initialize()

        # Login to your MetaTrader 5 account
        authorized = mt5.login(login, password, server)
        if not authorized:
            print("Failed to login to MetaTrader 5. Please check your credentials.")
            mt5.shutdown()
            return

        # Select the symbol
        mt5.symbol_select(symbol)

        # Download historical data
        bars = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

        # Process the downloaded data
        if len(bars) == 0:
            print(f"No historical data available for symbol: {symbol}")
            return None

        print(f"Downloaded {len(bars)} bars of historical data for symbol: {symbol}")
        close_prices = np.array(bars['close'])
        high_prices = np.array(bars['high'])
        low_prices = np.array(bars['low'])


        if len(high_prices) > 0 and len(low_prices) > 0:
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            total_profit = 0.0

            for i in range(len(bars) - 1):
                print(len(high_prices), len(low_prices), symbol)
                print(f"Symbol: {symbol}")
                print(f"High prices: {high_prices[:i+1]}")
                print(f"Low prices: {low_prices[:i+1]}")
                #Calculate the Ichimoku signal based on historical price data
                ichimoku_signal = calculate_ichimoku_cloud(symbol, high_prices[:i+1], low_prices[:i+1])

                print(f"Symbol: {symbol}", ichimoku_signal )

                # Backtesting - Track trade performance
                if ichimoku_signal != 'No Signal' or ichimoku_signal != 'No Trend':
                    # ... Trading logic ...

                    # Simulate trade execution and calculate profit/loss
                    if ichimoku_signal == 'Buy' or ichimoku_signal == 'Sell':
                        total_trades += 1

                        # Simulate trade execution and calculate profit/loss
                        if ichimoku_signal == 'Buy':
                            # Simulate buy trade execution
                            # Calculate profit/loss based on the next bar's close price
                            trade_price = bars[i+1]['close']
                            profit = (close_prices[i+1] - trade_price)
                            if profit > 0:
                                winning_trades += 1
                            else:
                                losing_trades += 1
                            total_profit += profit

                        elif ichimoku_signal == 'Sell':
                            # Simulate sell trade execution
                            # Calculate profit/loss based on the next bar's close price
                            trade_price = bars[i+1]['close']
                            profit = (trade_price - close_prices[i+1])
                            if profit > 0:
                                winning_trades += 1
                            else:
                                losing_trades += 1
                            total_profit += profit

            #Return backtest results
            return symbol, total_trades, winning_trades, losing_trades, total_profit

    except Exception as e:
        print(f"Error occurred while backtesting symbol {symbol}: {str(e)}")
        return None


def run_backtests(symbols):
    for symbol in symbols:
        backtest_result = backtest_strategy(symbol)
        if backtest_result is not None:
            print(f"Backtest Results for {symbol}:")
            print(f"Total Trades: {backtest_result[1]}")
            print(f"Winning Trades: {backtest_result[2]}")
            print(f"Losing Trades: {backtest_result[3]}")
            print(f"Total Profit: {backtest_result[4]}")
            print()


def connect_to_mt5():
    if initialize(login=login, password=password, server=server):
        return True
    else:
        return False

def disconnect_from_mt5():
    mt5.shutdown()
    print("Disconnected from MetaTrader 5")

def plot_signal_history(signal_history):
    plt.figure(figsize=(12, 6))
    for symbol, signals in signal_history.items():
        plt.plot(signals, label=symbol)
    plt.xlabel("Signal History")
    plt.ylabel("Signal")
    plt.title("Ichimoku Signal History")
    plt.legend()
    plt.grid(True)
    plt.show()

def remove_failed_symbols(symbols):
    symbols_copy = symbols.copy()
    for symbol_info in symbols_copy:
        try:
            mt5.symbol_info(symbol_info.name)
        except Exception:
            symbols.remove(symbol_info)
    return symbols

if __name__ == '__main__':
    SYMBOLS = ["EURUSD_SB", "GBPUSD_SB", "USDJPY_SB", "GER40_SB", "NAS100_SB", "US30_SB", "EURGBP_SB",
               "AUDUSD_SB", "USDCAD_SB", "UK100_SB"]
    MAGIC_NUMBERS = {
        "EURUSD_SB": 123456,
        "GBPUSD_SB": 789012,
        "USDJPY_SB": 345678,
        "GER40_SB": 901234,
        "NAS100_SB": 567890,
        "US30_SB": 123789,
        "EURGBP_SB": 123790,
        "AUDUSD_SB": 123791,
        "UK100_SB":  123792
    }
    VOLUME = 0.3
    TIMEFRAME = mt5.TIMEFRAME_M1

    initialize()

    if not mt5.initialize():
        print("Failed to initialize MetaTrader 5")
        exit(1)

    run_backtests(SYMBOLS)  # Perform the backtests


    trade_symbols(SYMBOLS, VOLUME, TIMEFRAME)

    mt5.shutdown()




