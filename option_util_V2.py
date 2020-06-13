import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import math
import scipy.stats as st
cdf = st.norm.cdf
import option_util_V2 as ou
sqrt = math.sqrt
from importlib import reload
import re
import itertools
from multiprocessing import Pool
import multiprocessing
import timeit
import time

# File containing Python functions used for calculating Options information


def get_std(hist_data, ma_days, exp_date):              
    
    # Days til 1 week before expiry 
    start = hist_data.tail(1).index[0].strftime('%Y-%m-%d')
    end = exp_date
    expiry_days = np.busday_count( start, end )  # business days only
 
   
    # Bare bones copy of the dataset
    stock_data = hist_data[['Close','Volume']].copy()

    #Calc % Change of the stock price over X days
    stock_data['ret'] = stock_data['Close'].pct_change() # also tried using 21 days. but decided may be better to 'annualize' daily

    # Moving average % change
    # stock_data['ret_ma'] = stock_data['ret'].rolling(window=ma_days).mean() # 20 = 1 month

    # Calculate Variance - Pct Chg
    stock_data['ret_var'] = stock_data.apply(lambda x: x['ret']**2, axis=1)  # decided not to subtract from MA ret, using 0 baseline

    # Average Variance - Pct Chg
    stock_data['ret_std'] = stock_data['ret_var'].rolling(window=ma_days).mean()

    # Final STD Calculation - sqrt of avg variance
    stock_data['ret_std'] = stock_data['ret_std'].apply(lambda x: sqrt(x))
    #        stock_data['ret_std'] = stock_data['ret_std'].apply(lambda x: math.sqrt(x))        

    # PRICE - 30 day moving average - (or whatever time range chosen)
    stock_data['price_ma'] = stock_data['Close'].rolling(window=ma_days).mean()

    # PRICE - Variance - closing difference from mean squared
    stock_data['price_var'] = stock_data.apply(lambda x: (x['price_ma'] - x['Close'])**2, axis=1)

    #  PRICE - Average Variance
    stock_data['price_std'] = stock_data['price_var'].rolling(window=ma_days).mean()

    # PRICE - STD
    stock_data['price_std'] = stock_data['price_std'].apply(lambda x: sqrt(x))   # sqrt function imported from inputs


    # PRICE - Calculating Pct Std
    price_std, final_ma_denominator = stock_data['price_std'].tail(1)[0], stock_data['price_ma'].tail(ma_days).mean() 
    price_std_pct = price_std / final_ma_denominator

    
    # Final Return STD
    final_ret_std = stock_data['ret_std'].tail(1)[0]

    
    # Calculating STD of return OVER the time window
    mthly_ret_std = final_ret_std*sqrt(expiry_days)    # convert daily ret std to monthly

    return stock_data, final_ret_std, mthly_ret_std, price_std_pct, price_std
    
    

def opt_price_data(opt, live, strikes):
    if str(live) != '1':
        bid = 'bid'
        ask = 'ask'
    else:
        bid = 'lastPrice'
        ask = 'lastPrice'

    if type(opt) == dict:
        calls = opt['calls']
        puts = opt['puts']
    else:
        calls = opt.calls
        puts = opt.puts

    # mid point calculations
    if str(live) == '2': 
        try: 
            buy_call_price = round(sum([calls[ calls['strike'] == strikes[0] ] [f'{ask}'].iloc[0], \
                         calls[ calls['strike'] == strikes[0] ] [f'{bid}'].iloc[0] ] ) / 2, 2) # calculate mid

        except:
            buy_call_price = 0.0
        try:
            buy_put_price = round(sum([puts[ puts['strike'] == strikes[1] ] [f'{ask}'].iloc[0], \
                         puts[ puts['strike'] == strikes[1] ] [f'{bid}'].iloc[0] ] ) / 2, 2) 
        except:
            buy_put_price = 0.0

        try:
            sell_call_price = round(sum([calls[ calls['strike'] == strikes[2] ] [f'{bid}'].iloc[0], \
                          calls[ calls['strike'] == strikes[2] ] [f'{ask}'].iloc[0] ]) / 2, 2) 
        except:
            sell_call_price = 0.0
        try:
            sell_put_price = round(sum( [puts[ puts['strike'] == strikes[3] ] [f'{bid}'].iloc[0], \
                               puts[ puts['strike'] == strikes[3] ] [f'{ask}'].iloc[0] ]) / 2, 2) 
        except:
            sell_put_price = 0.0
    # OR bid/ask or lastPrice calcs
    else:
        try:
            buy_call_price = calls[ calls['strike'] == strikes[0] ] [f'{ask}'].iloc[0] # Ask in prod. lastPrice in test
        except:
            buy_call_price = 0.0
        try:
            buy_put_price = puts[ puts['strike'] == strikes[1] ] [f'{ask}'].iloc[0]
        except:
            buy_put_price = 0.0
        try:
            sell_call_price = calls[ calls['strike'] == strikes[2] ] [f'{bid}'].iloc[0] 
        except:
            sell_call_price = 0.0
        try:
            sell_put_price = puts[ puts['strike'] == strikes[3] ] [f'{bid}'].iloc[0]
        except:
            sell_put_price = 0.0
    
    return [ buy_call_price, buy_put_price, sell_call_price, sell_put_price ]

    
    
def options_outcomes(perc_range, increments, current_price, future_price, std, \
                     opt_strikes, opt_prices):
    
    #for key, value in packages.items():
    #    if key == 'pd':
    #        pd = value
    #    elif key == 'np':
    #        np = value
    #    elif key == 'cdf':
    #        cdf = value
    
    # Creating Hypothetical Price Ranges
    price_perc = pd.Series(np.arange(1-perc_range, 1+perc_range+increments, increments))
    prices = pd.Series( [ round(i * current_price,2) for i in price_perc])
    price_perc = price_perc.apply(lambda x: round(x - 1,2))
    
    df = pd.DataFrame( {'prices': prices, 'price_perc': price_perc})
    
    std_in_price = future_price*std  # here you could swap in the predicted MA price in 30 days
    
    # Calculating Z-score & probabilities based on STD calcualted earlier
    df['z-score'] = df['prices'].apply(lambda x: abs(current_price-x)/std_in_price)
    df['prob'] = df['z-score'].apply(lambda x: round(1- cdf(x),4)*2)  # convert z-score to probability, x2 two sided
    # df['prob'] = df['z-score'].apply(lambda x: round(1- st.norm.cdf(x),4)*2)  # convert z-score to probability, x2 two sided
    df['prob_pct'] = df['prob']/df['prob'].sum()
    
    
    
    # Calculating Options Values for Each Potential Price -- use separate df to sum across rows -- CALCULATES INTRINSIC VALUE
    
    buy_call_strike, buy_put_strike, sell_call_strike, sell_put_strike = \
                opt_strikes[0], opt_strikes[1], opt_strikes[2], opt_strikes[3]
    
    buy_call_price, buy_put_price, sell_call_price, sell_put_price = \
                opt_prices[0], opt_prices[1], opt_prices[2], opt_prices[3]

    df2 = pd.DataFrame()
    df2[f'buy_call_{buy_call_strike}'] = [ 0 if buy_call_strike == \
           None else max((x-buy_call_strike)*100-buy_call_price*100,- \
           buy_call_price*100) for x in df.prices ]
    df2[f'buy_put_{buy_put_strike}'] = [ 0 if buy_put_strike == None else \
           max((buy_put_strike-x)*100-buy_put_price*100,-buy_put_price*100) \
                                        for x in df.prices ]
    df2[f'sell_call_{sell_call_strike}'] = [ 0 if sell_call_strike == None else \
          (sell_call_price*100 - max(0, x - sell_call_strike)*100) for x in df.prices  ]
    df2[f'sell_put_{sell_put_strike}'] = [ 0 if sell_put_strike == None else \
          min(sell_put_price*100 - (sell_put_strike - x)*100, sell_put_price*100) \
                                          for x in df.prices ]
    
    df2['profit_loss'] = df2.sum(axis=1)
    

    df = pd.concat( [df, df2], axis=1)
    
    # Expected profit

    # weighted profit
    df['exp_profit_loss'] = df['profit_loss'] * df['prob_pct']

    
    return df




     
def risk_analysis(df, opt_strikes, opt_prices, var_std, exp_date, ticker):
    
    # Grab strikes & price data
    buy_call, buy_put, sell_call, sell_put = opt_strikes[0], \
                        opt_strikes[1], opt_strikes[2], opt_strikes[3]
    
    buy_call_price, buy_put_price, sell_call_price, sell_put_price = round(opt_prices[0],2)\
                           , round(opt_prices[1],2), round(opt_prices[2],2), \
                            round(opt_prices[3],2)
    
    
    # Weighted Profit based on probability percentages
    exp_profit = df['exp_profit_loss'].mean()
    
    # Risk Spread 
    mock = df[['profit_loss','price_perc']].copy()
    mock['price_perc'] = mock.apply(lambda x: x['price_perc'] if x['profit_loss'] < 0 else 100, axis = 1)
    mock['lag'] = mock['price_perc'].shift(1)
    mock['risk_spread'] = mock.apply(lambda x: abs(x['price_perc'] - x['lag']) if (x['lag'] + x['price_perc']) < 90 \
        else 0, axis=1)
    risk_spread = round(sum(mock['risk_spread']),2)
    
    # Cost of trade
    cost = -(-buy_call_price - buy_put_price + sell_call_price + sell_put_price)*100
    
    # Bear Bull -  probability of profit with positive movement 30% - 10% = 20% more proability upside
    bull_bear = df[ ((df['price_perc'] > 0)) & (df['profit_loss'] > 0) ]['prob_pct'].sum() \
           -  df[ (df['price_perc'] < 0) & (df['profit_loss'] > 0) ]['prob_pct'].sum()      
    
    # VAR -- Avg of the profits at the X standard deviation
    
    def closest(lst, K): 
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 
    
    # Give me the item in a list closest to the number 1 (1 standard dev)
    VAR = df[ round(df['z-score'],4) == round(closest(df['z-score'],var_std),4) ] \
        ['profit_loss'].mean()
    
    
    # Max Loss & Max Profit
    max_loss = df['profit_loss'].min()
    max_profit = df['profit_loss'].max()

    # Current Value
    current_value = df[df['price_perc'] == 0]['profit_loss'].values[0]
    head_profit = df['profit_loss'].head(1).values[0]
    tail_profit = df['profit_loss'].tail(1).values[0]

    
    # Risk Ratio

    #try:
#    risk_ratio_95th = round(min(df[ (df['prob'] > .05) & (df['profit_loss'] > 0)]\
#                               ['profit_loss'].max() / \
#                    abs(df[ df['profit_loss'] < 0]['profit_loss'].min()), 3),2)

    risk_ratio_95th = round(min( \
                      min( head_profit, tail_profit)  / \
                    abs(df[ df['profit_loss'] < 0]['profit_loss'].min()), 1.3),2)

    min_leg_profit = min( head_profit, tail_profit )
    
    #risk_ratio_95th = round(min(df[ df['profit_loss'] > 0]['profit_loss'].max() / \
    #                    abs(df[ df['profit_loss'] < 0]['profit_loss'].min()), 3),2)

    # replacing errors    
    risk_ratio_95th = np.where(math.isnan(risk_ratio_95th),float(0),risk_ratio_95th)
    #except: # ZeroDivisionError:
    #    risk_ratio_95th = 0
    
    # Outcome Odds
    odds_profit = df[ df['profit_loss'] > 0]['prob_pct'].sum()
    odds_loss = df[ df['profit_loss'] < 0]['prob_pct'].sum()
    
    # Median Profit / loss
    median_profit = df[ df['profit_loss'] > 0].sort_values('profit_loss')['profit_loss'].median()
    median_loss = df[ df['profit_loss'] < 0].sort_values('profit_loss')['profit_loss'].median()



    return pd.DataFrame( { 'stock': ticker, 'exp_date': exp_date, 'buy_call': [str(buy_call)+' | '+str(buy_call_price)], \
                          'buy_put': [str(buy_put)+' | '+str(buy_put_price)], \
                          'sell_call': [str(sell_call)+' | '+str(sell_call_price)], \
                          'sell_put': [str(sell_put)+' | '+str(sell_put_price)], \
                          'cost': round(cost,2), \
                          'exp_profit': round(exp_profit,2), 'risk_spread': round(risk_spread,2), \
                           'odds_profit': round(odds_profit,2), 'odds_loss': round(odds_loss,2), \
                          'max_loss': round(max_loss,2), 'max_profit': round(max_profit,2), 'risk_ratio_95th': risk_ratio_95th, \
                          'median_profit': round(median_profit,2), 'median_loss': round(median_loss,2), \
                          'bull_bear': round(bull_bear,2), f'var_std_{var_std}': round(VAR,2), \
                          'buy_call_price': buy_call_price, 'buy_put_price': buy_put_price, \
                          'sell_call_price': sell_call_price, 'sell_put_price': sell_put_price, \
                          'current_value': round(current_value,2), 'head_profit': round(head_profit,2), \
                          'tail_profit': round(tail_profit,0), 'min_leg_profit': min_leg_profit
                         }
                       )
    

    
def ric_permutations3(inputs):
    
    stock = inputs['stock'] 
    exp_datelist = inputs['exp_datelist']
    volume = inputs['volume']
    openInterest = inputs['openInterest']
    hist_data = inputs['hist_data']
    past_time = inputs['past_time']
    max_permutations = inputs['max_permutations']
    strategy_list = inputs['strategy_list']
    ticker = inputs['ticker']
    option_chain = inputs['option_chain']    
    current_price = inputs['current_price']

    stock_data_dict = {}
    final_ret_std_dict = {}
    mthly_ret_std_dict = {}
    price_std_pct_dict = {}
    price_std_dict = {}
        
    real_final_combos = pd.DataFrame()
    perm_list = []
    exp_df_final = pd.DataFrame()
    
    for strategy in strategy_list:
        
        combos = pd.DataFrame()
        exp_list = []
        nums = [0]
        
        for i in range(len(exp_datelist)):
            # exp date cleansing
            exp = str(exp_datelist[i])
            exp = exp.strip("[]")
            exp = exp.strip("''")

            try:
            


                # Get Option Data
                if type(option_chain) == dict:
                    opt = option_chain[exp]  # exp_date
                    calls = opt['calls'] 
                    puts = opt['puts']
                else:
                    opt = stock.option_chain(exp)  #
                    calls = opt.calls
                    puts = opt.puts


                # Permutations    
                calls['date_mod'] = calls['lastTradeDate'].apply(lambda x: \
                                                                         x.strftime('%Y-%m-%d'))
                puts['date_mod'] = puts['lastTradeDate'].apply(lambda x: \
                                                                       x.strftime('%Y-%m-%d'))
                
                # Replace any missing values with 0s
                calls = calls.fillna(0)
                puts = puts.fillna(0)

                
                # Get options data & Filter to min volume & interest, and latest trade date only
                calls = calls[ (calls['volume'] >= volume) & (calls['openInterest'] >= openInterest) & \
                                 (calls['date_mod'] == calls['date_mod'].max()) \
                                 ]    # min volume & interest
                puts = puts[ (puts['volume'] >= volume) & (puts['openInterest'] >= openInterest) & \
                                 (puts['date_mod'] == puts['date_mod'].max()) \
                               ]


                # permutations   
                if strategy == 'ric':


                    exp_combos = pd.DataFrame(itertools.product([ticker], [exp], calls.strike.values, puts.strike.values, \
                                                                calls.strike.values, puts.strike.values),\
                                     columns=('stock','exp_date','buy_call','buy_put','sell_call','sell_put'))


                    # filter down to specific Options Strategy only
                    exp_combos = exp_combos[ (exp_combos['buy_put'] <= exp_combos['buy_call']*1.05  ) & \
                                    (exp_combos['sell_call'] > exp_combos['buy_call']) & \
                                    (exp_combos['sell_put'] < exp_combos['buy_put']) & \
                                    (exp_combos['buy_put'] <= current_price*1.1  ) & \
                                    (exp_combos['buy_call'] >= current_price*0.9  ) & \
                                    (exp_combos['sell_put'] < exp_combos['sell_call']) \
                                   ].reset_index(drop=True)


                    # Bringing in Volume Data
                    # BUY CALL
                    buy_call_df = pd.merge(exp_combos, calls[['strike','volume', 'openInterest']], \
                             left_on = 'buy_call', right_on = 'strike', how = 'left')
                    exp_combos[['buy_call_vol', 'buy_call_op_int']] = buy_call_df[['volume', 'openInterest']]

                    # BUY PUT
                    buy_put_df = pd.merge(exp_combos, puts[['strike', 'volume', 'openInterest']], \
                                         left_on = 'buy_put', right_on = 'strike', how = 'left')
                    exp_combos[['buy_put_vol', 'buy_put_op_int']] = buy_put_df[['volume', 'openInterest']]

                    # SELL CALL
                    sell_call_df = pd.merge(exp_combos, calls[['strike', 'volume', 'openInterest']], \
                                         left_on = 'sell_call', right_on = 'strike', how = 'left')
                    exp_combos[['sell_call_vol', 'sell_call_op_int']] = sell_call_df[['volume', 'openInterest']]

                    # SELL PUT
                    sell_put_df = pd.merge(exp_combos, puts[['strike', 'volume', 'openInterest']], \
                                         left_on = 'sell_put', right_on = 'strike', how = 'left')
                    exp_combos[['sell_put_vol', 'sell_put_op_int']] = sell_put_df[['volume', 'openInterest']]

                    # MIN VOL
                    exp_combos['min_vol'] = exp_combos.apply(lambda x: min(x['buy_call_vol'],x['buy_put_vol'], \
                                                                           x['sell_call_vol'],x['sell_put_vol']), axis=1)

                    # Log Strategy
                    exp_combos['strategy'] = strategy


                elif strategy == 'bear_put':
                    exp_combos = pd.DataFrame(itertools.product([ticker], [exp], [None], puts.strike.values, \
                                                                [None], puts.strike.values),\
                                     columns=('stock','exp_date','buy_call','buy_put','sell_call','sell_put'))

                    # filter down to specific Options Strategy only
                    exp_combos = exp_combos[ (exp_combos['sell_put'] < exp_combos['buy_put']  ) #& \
                                    #(exp_combos['sell_call'] > exp_combos['buy_call']) & \
                                    #(exp_combos['sell_put'] < exp_combos['sell_call']) \
                                   ].reset_index(drop=True)

                    # BUY CALL VOL
                    exp_combos['buy_call_vol'], exp_combos['buy_call_op_int'] = 0, 0

                    # BUY PUT VOL
                    buy_put_df = pd.merge(exp_combos, puts[['strike', 'volume', 'openInterest']], \
                                         left_on = 'buy_put', right_on = 'strike', how = 'left')
                    exp_combos[['buy_put_vol', 'buy_put_op_int']] = buy_put_df[['volume', 'openInterest']]

                    # SELL CALL VOL
                    exp_combos['sell_call_vol'], exp_combos['sell_call_op_int'] = 0, 0


                    # SELL PUT VOL
                    sell_put_df = pd.merge(exp_combos, puts[['strike', 'volume', 'openInterest']], \
                                         left_on = 'sell_put', right_on = 'strike', how = 'left')
                    exp_combos[['sell_put_vol', 'sell_put_op_int']] = sell_put_df[['volume', 'openInterest']]

                    # MIN VOL
                    exp_combos['min_vol'] = exp_combos.apply(lambda x: min(x['buy_put_vol'], \
                                                                           x['sell_put_vol']), axis=1)

                    # Log Strategy
                    exp_combos['strategy'] = strategy

                elif strategy == 'bull_call':
                    exp_combos = pd.DataFrame(itertools.product([ticker], [exp], calls.strike.values, [None], \
                                                calls.strike.values, [None]),\
                                            columns=('stock','exp_date','buy_call','buy_put','sell_call','sell_put'))

                    # filter down to specific Options Strategy only
                    exp_combos = exp_combos[ #(exp_combos['sell_put'] < exp_combos['buy_put']  ) & \
                                    (exp_combos['sell_call'] > exp_combos['buy_call']) \
                                    #(exp_combos['sell_put'] < exp_combos['sell_call']) \
                                   ].reset_index(drop=True)

                    # Bringing in Volume Data
                    # BUY CALL
                    buy_call_df = pd.merge(exp_combos, calls[['strike','volume', 'openInterest']], \
                             left_on = 'buy_call', right_on = 'strike', how = 'left')
                    exp_combos[['buy_call_vol', 'buy_call_op_int']] = buy_call_df[['volume', 'openInterest']]

                    # BUY PUT
                    exp_combos['buy_put_vol'], exp_combos['buy_put_op_int'] = 0, 0

                    # SELL CALL
                    sell_call_df = pd.merge(exp_combos, calls[['strike', 'volume', 'openInterest']], \
                                         left_on = 'sell_call', right_on = 'strike', how = 'left')
                    exp_combos[['sell_call_vol', 'sell_call_op_int']] = sell_call_df[['volume', 'openInterest']]

                    # SELL PUT
                    exp_combos['sell_put_vol'], exp_combos['sell_put_op_int'] = 0, 0

                    # MIN VOL 
                    exp_combos['min_vol'] = exp_combos.apply(lambda x: min(x['buy_call_vol'], \
                                                                           x['sell_call_vol']), axis=1)

                    # Log Strategy
                    exp_combos['strategy'] = strategy

                elif strategy == 'strangle':
                    exp_combos = pd.DataFrame(itertools.product([ticker], [exp], calls.strike.values, puts.strike.values, \
                                                [None], [None]),\
                                            columns=('stock','exp_date','buy_call','buy_put','sell_call','sell_put'))

                    # filter down to specific Options Strategy only
                    exp_combos = exp_combos[ #(exp_combos['sell_put'] < exp_combos['buy_put']  ) & \
                                    #(exp_combos['sell_call'] > exp_combos['buy_call']) & \
                                    (exp_combos['buy_put'] < exp_combos['buy_call']*1.05) \
                                   ].reset_index(drop=True)

                    # Bringing in Volume Data
                    # BUY CALL
                    buy_call_df = pd.merge(exp_combos, calls[['strike','volume', 'openInterest']], \
                             left_on = 'buy_call', right_on = 'strike', how = 'left')
                    exp_combos[['buy_call_vol', 'buy_call_op_int']] = buy_call_df[['volume', 'openInterest']]

                    # BUY PUT
                    buy_put_df = pd.merge(exp_combos, puts[['strike', 'volume', 'openInterest']], \
                                         left_on = 'buy_put', right_on = 'strike', how = 'left')
                    exp_combos[['buy_put_vol', 'buy_put_op_int']] = buy_put_df[['volume', 'openInterest']]

                    # SELL CALL
                    exp_combos['sell_call_vol'], exp_combos['sell_call_op_int'] = 0, 0

                    # SELL PUT
                    exp_combos['sell_put_vol'], exp_combos['sell_put_op_int'] = 0, 0

                    # MIN VOL
                    exp_combos['min_vol'] = exp_combos.apply(lambda x: min(x['buy_call_vol'],x['buy_put_vol']), axis=1)

                    # Log Strategy
                    exp_combos['strategy'] = strategy

                # Count the size
                nums.append(len(exp_combos))

                # Sort by highest min volume
                exp_combos = exp_combos.sort_values('min_vol', ascending = False)

                # list of exp's that worked, i.e. no missing data
                exp_list.append(exp)

                # Append to DF
                combos = combos.append(exp_combos, ignore_index=True)

                # Get STD Data
                stock_data, final_ret_std, mthly_ret_std, price_std_pct, \
                        price_std = ou.get_std(hist_data, past_time, exp)

                stock_data_dict.update({ f'{exp}': stock_data })
                final_ret_std_dict.update({ f'{exp}': final_ret_std })
                mthly_ret_std_dict.update({ f'{exp}': mthly_ret_std })
                price_std_pct_dict.update({ f'{exp}': price_std_pct })
                price_std_dict.update({ f'{exp}': price_std })


            except:
                pass

        if max_permutations > 0:  # weighted approach to limiting permutations
            perm_per_exp = pd.Series([ min(round(i/sum(nums)*max_permutations),round((i))) if sum(nums) > 0 \
                               else 0 for i in nums ])
            final_combos = pd.DataFrame()
            for i, exp in enumerate(exp_list):
                final_combos = final_combos.append(combos[ combos['exp_date'] == exp ].iloc[ :perm_per_exp[i+1] ] )
            combos = final_combos

        else:
            perm_per_exp = pd.Series([ round(i) for i in nums ])

        real_final_combos = real_final_combos.append(combos)
        
        # Calculating number of permutations per expiry
        exp_df = pd.Series(dict(zip(exp_list, list(perm_per_exp[1:]))))
        exp_df_final = exp_df_final.append(exp_df, ignore_index=True)
    
    # Summing up the results for Permutations per Expiry     
    exp_dict = exp_df_final.sum().to_dict()
    print(exp_dict)  # leave this in. won't interfere with multi_processing.

        
    std_data = {
        'stock_data': stock_data_dict,
        'final_ret_std': final_ret_std_dict,
        'mthly_ret_std': mthly_ret_std_dict,
        'price_std_pct': price_std_pct_dict,
        'price_std': price_std_dict
    }
    
    
    return real_final_combos, std_data, exp_dict


def stock_optimizer1(all_data):
    
    combos, std_data, inputs = all_data[0], all_data[1], all_data[2]
    
    # Unpacking the input data 
    live = inputs['live']        
    perc_range = inputs['perc_range']     
    increments = inputs['increments']
    std_for_var = inputs['std_for_var']
    stock = inputs['stock']
    current_price = inputs['current_price']
    ticker = inputs['ticker']
    option_chain = inputs['option_chain']

        
    # Other inputs needed
    exp = combos[1]  # expiration date 
    mthly_ret_std = std_data['mthly_ret_std'][exp]   # the standard deviation for the time period in question
    strategy = combos[15]  # ric, bear put, etc 

    # Get Option Data
    try:   # if the opt price doesn't exist, or something goes wrong with the yahoo finance function, get the fuck out!
        
        if type(option_chain) == dict:   # yahoo finance is a method, not a dict
            opt = option_chain[exp]  # exp_date
        else:
            opt = stock.option_chain(exp)  #
            
        opt_strikes = []
        for x in list(combos[2:6]):
            if x is None:
                strike = None
            else:
                strike = float(x)
            opt_strikes.append(strike)

        opt_prices = ou.opt_price_data(opt, live, opt_strikes) # i is opt_strikes
    
    except:
        opt_strikes = [0.0, 0.0, 0.0, 0.0]
        opt_prices = [0.0, 0.0, 0.0, 0.0] 
    
    # Calculate hypothetical outcomes 
    df = ou.options_outcomes(perc_range, increments, \
                         current_price, current_price # future pred price here
                         , mthly_ret_std, opt_strikes, opt_prices) 

    # Risk Analysis
    option_combo = ou.risk_analysis(df, opt_strikes, opt_prices, std_for_var, exp, ticker)
    

    # Adding Volumes info 
    option_combo['buy_call_vol'] = str(combos[6])+' | '+str(combos[7])
    option_combo['buy_put_vol'] = str(combos[8])+' | '+str(combos[9])
    option_combo['sell_call_vol'] = str(combos[10])+' | '+str(combos[11])
    option_combo['sell_put_vol'] = str(combos[12])+' | '+str(combos[13])

    # Adding Strategy
    option_combo.insert(2, 'strategy', strategy)
    option_combo.insert(3, 'price', current_price)

    final_values = np.array(option_combo.values.tolist())
    columns = np.array(option_combo.columns.tolist())
    
    return final_values, columns
    


def ric_score(df):

    # Expected Profit - max
    ep_wgt = .2

    # Risk Ratio- max
    rr_wgt = .3

    # risk spread - min
    rs_wgt = .25

    # odds profit -max   # better version of risk spread
    op_wgt = .25

    # median profit -max
    mp_wgt = 0

    # Convert to numeric
    #df['buy_call_price'] = df['buy_call_price'].apply(pd.to_numeric)
    #df['buy_put_price'] = df['buy_put_price'].apply(pd.to_numeric)
    #df['sell_call_price'] = df['sell_call_price'].apply(pd.to_numeric)
    #df['sell_put_price'] = df['sell_put_price'].apply(pd.to_numeric)
    df['cost'] = df['cost'].apply(pd.to_numeric)
    df['exp_profit'] = df['exp_profit'].apply(pd.to_numeric)
    df['max_loss'] = df['max_loss'].apply(pd.to_numeric)
    df['max_profit'] = df['max_profit'].apply(pd.to_numeric)
    df['risk_ratio_95th'] = df['risk_ratio_95th'].apply(pd.to_numeric)
    df['risk_spread'] = df['risk_spread'].apply(pd.to_numeric)
    df['odds_profit'] = df['odds_profit'].apply(pd.to_numeric)
    df['bull_bear'] = pd.to_numeric(df['bull_bear'], errors='coerce')


    df['median_profit'] = pd.to_numeric(df['median_profit'], errors='coerce')
    df['head_profit'] = pd.to_numeric(df['head_profit'], errors='coerce')
    df['tail_profit'] = pd.to_numeric(df['tail_profit'], errors='coerce')
    df['min_leg_profit'] = pd.to_numeric(df['min_leg_profit'], errors='coerce')
    df[['median_profit', 'head_profit', 'tail_profit', 'min_leg_profit']] = df[['median_profit', 'head_profit', 'tail_profit', \
                        'min_leg_profit']].fillna(0)

        
    
    # Clean up unwanted rows with no chance of profit
    df = df[ df['odds_profit'].apply(pd.to_numeric)  > 0 ]

    # if max_loss falls off the chart then the whole thing is useless
    df = df[ abs(df['cost']) == abs(df['max_loss']) ]

    # Get rid of rows where the odds profit is too high. it means the loss is likely going to be a margin thing. 
    df = df[ df['odds_profit'].apply(pd.to_numeric)  < 1 ]

        
    # Deleting rows with missing price data  --> remove rows where option != None & price == 0
    #df = df[ (df['buy_call_price'] > 0) | df['buy_call'].str.startswith('None') == True ]
    #df = df[ (df['buy_put_price'] > 0) | df['buy_put'].str.startswith('None') == True ]
    #df = df[ (df['sell_call_price'] > 0) | df['sell_call'].str.startswith('None') == True ]
    #df = df[ (df['sell_put_price'] > 0) | df['sell_put'].str.startswith('None') == True ]

    # drop the price columns
    #try:
    #    df = df.drop(['buy_call_price','buy_put_price','sell_call_price','sell_put_price'], axis =1)
    #except:
    #    pass

    # SCORING
    # Expected Profit - uses a weighted average profit using stock price probabilities
    ep_score = (( df['exp_profit'] - df['exp_profit'].min() ) / \
                (df['exp_profit'].max() - df['exp_profit'].min())*ep_wgt)

    # Risk Spread - the size of the 'zones' where the spread loses money, in terms of percentage movement
    # e.g. if -3% to +3% of current price results in a loss then rs = .06
    rs_score = ((1 - ( df['risk_spread'] - df['risk_spread'].min() ) / \
                (df['risk_spread'].max() - df['risk_spread'].min()))*rs_wgt)

    # Risk Ratio - the ratio of maximum profit to maximum loss. Higher is better. Capped at 3. 
    rr_score = (( df['risk_ratio_95th'] - df['risk_ratio_95th'].min() ) / \
                (df['risk_ratio_95th'].max() - df['risk_ratio_95th'].min())*rr_wgt)

    # Sum of stock price probabilities that result in profit
    op_score = (( df['odds_profit'] - df['odds_profit'].min() ) / \
                (df['odds_profit'].max() - df['odds_profit'].min())*op_wgt)

    # Median Profit - uses a weighted average profit using stock price probabilities
    mp_score = (( df['median_profit'] - df['median_profit'].min() ) / \
                (df['median_profit'].max() - df['median_profit'].min())*mp_wgt)

    df['score'] = np.where(ep_score.isnull(), ep_wgt, ep_score) + np.where(rr_score.isnull(), rr_wgt, rr_score) + \
                np.where(rs_score.isnull(), rs_wgt, rs_score) + np.where(op_score.isnull(), op_wgt, op_score) + \
                np.where(mp_score.isnull(), mp_wgt, mp_score)
    
    # Adding in call and price data separately
    df['buy_call_strike'] = df['buy_call'].apply(lambda x: x.split('|')[0])
    df['buy_put_strike'] = df['buy_put'].apply(lambda x: x.split('|')[0])
    df['sell_call_strike'] = df['sell_call'].apply(lambda x: x.split('|')[0])
    df['sell_put_strike'] = df['sell_put'].apply(lambda x: x.split('|')[0])
    
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return df


    
    

# Holding Stock Option Data in a dict 

def call_option_data(stock):
    final_dict = {}
    exp_list = [ [ stock.options[i] ] for i in range(len(stock.options)) ]
    for i in range(len(exp_list)):
        # exp date cleansing
        exp = str(exp_list[i])
        exp = exp.strip("[]")
        exp = exp.strip("''")
        try:
            opt_dict = { 'calls': stock.option_chain(exp).calls, 'puts': stock.option_chain(exp).puts }
            final_dict.update({ exp: opt_dict })
        except:
            pass

    return final_dict


# Grabbing and storing stock data
def get_stock_data(stock_list):
    full_stock_dict = {}
    for i in stock_list:
        stock = yf.Ticker(i)
        options = stock.options
        option_chain = ou.call_option_data(stock)
        ticker = stock.ticker
        hist_data = stock.history(period="240mo", interval="1d")
        single_stock_dict = { 'ticker': ticker, 'options': options, 'option_chain': option_chain, \
                           'hist_data': hist_data} 
        full_stock_dict.update( { i: single_stock_dict })
        
    return full_stock_dict


def date_filter(options, min_wks_out, max_wks_out):
    date_list = []
    min_time = pd.to_datetime((datetime.now() + \
                relativedelta(weeks=min_wks_out)).strftime('%Y-%m-%d'))
    max_time = pd.to_datetime((datetime.now() + \
            relativedelta(weeks=max_wks_out)).strftime('%Y-%m-%d'))

    for i in range(len(options)):
        exp_time = pd.to_datetime(options[i])
        if  min_time <= exp_time and max_time >= exp_time:
            date_list.append([ options[i] ])
    return date_list