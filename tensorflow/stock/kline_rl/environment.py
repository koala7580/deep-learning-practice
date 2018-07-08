"""The environment for train reinforcement leanring algorithm of K line trader.
"""
import tushare
import numpy as np
import pandas as pd

class Environment(object):

    def __init__(self, code, start_date, end_date, days_per_episode=20):
        self._code = code
        self._start_date = start_date
        self._end_date = end_date
        self._days_per_episode = days_per_episode

        self._data = tushare.get_k_data(code, start_date, end_date)

        if len(self._data) < one_episode_steps * 2:
            raise ValueError('The data length is %d, which is less than %d days per episode times 2.'
                             % (len(self._data, days_per_episode)))

        self._data['volume'] = self._normalize_volume(self._data['volume'])

        self._start_index = 0
        self._current_index = 0
        self._current_state = None
        self._current_hold_stock = 0

        self._current_money = 1000000.0 # 一开始有 100 万

    def _normalize_volume(self, volume):
        return (volume - volume.min()) / (volume.max() - volume.min())

    def reset(self):
        self._start_index = np.random.randint(0, len(self._data) - self._days_per_episode)
        self._current_index = self._start_index

        d = self._data.iloc[self._start_index]
        self._current_state = np.asarray([
            d['open'], d['high'], d['low'], d['close'], d['volume'],
        ])

        return self._current_state

    def step(self, action):
        self._current_index += 1
        state = self._data.iloc[self._current_index]
        state = np.asarray([
            state['open'],
            state['high'],
            state['low'],
            state['close'],
            state['volume']
        ])
        current_price = state[-2]
        done = self._current_index - self._start_index > self._days_per_episode

        if action == 0: # wait, do nothing
            reward = self._reward(current_price)
        elif action == 1: # buy, as much as possible
            self._buy()
            reward = self._reward(current_price)
        elif action == 2: # sell, all
            self._sell()
            reward = self._reward(current_price)
        else:
            raise ValueError('Unknown action %d' % action)

        self._current_state = state
        return self._current_state, reward, done, None

    def _reward(self, current_price):
        return current_price * self._current_hold_stock + self._current_money

    def _buy(self):
        # 买入股票所用金额：10元/股×10000股=100000元；
        # 过户费：0.6元÷1000股×10000股=6元(沪市股票计算，深市为0)；
        # 交易佣金：100000×1‰=100元(其中包含经手费：10万*0.0696‰=6.96元；证管费：10万*0.02‰=2元)；
        # 买入总成本：100000元+6元+100元=100106元(买入10000股，每股10元，所需总资金)。
        d = self._data.iloc[self._current_index]
        price = np.average([d['high'], d['low'], d['close']])
        n = self._current_money / price // 100 * 100
        if n > 0:
            fee1 = 0.6 / 1000 * n
            fee2 = n * price * 0.001
            self._current_money - fee1 - fee2 - n * price
            self._current_hold_stock += n

    def _sell(self):
        # 若以10.10元/股卖出计算：
        # 股票金额：10.10元/股×10000股=101000元；
　　     # 印花税：101000元×1‰=101元；
　　     # 过户费：0.6元÷1000股×10000股=6元；
　　     # 交易佣金：101000元×1‰=101元(其中包含经手费：101000元*0.0696‰=7.03元；证管费：101000元*0.02‰=2.02元)
        d = self._data.iloc[self._current_index]
        price = np.average([d['high'], d['low'], d['close']])
        n = self._current_hold_stock
        fee1 = n * price * 0.001
        fee2 = 0.6 / 1000 * n
        fee3 = n * price * 0.001
        self._current_money += n * price - fee1 - fee2 - fee3
        self._current_hold_stock = 0

    @property
    def reward_threshold(self):
        return 1000000 * 1.5
