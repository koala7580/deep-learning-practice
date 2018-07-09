"""The environment for train reinforcement leanring algorithm of K line trader.
"""
import tushare
import numpy as np
import pandas as pd

class Environment(object):

    def __init__(self, code, start_date, end_date, default_money=100000.0, days_per_episode=20):
        self._default_money = default_money
        self._days_per_episode = days_per_episode

        self._data = tushare.get_k_data(code, start_date, end_date)

        if len(self._data) < days_per_episode * 2:
            raise ValueError('The data length is %d, which is less than %d days per episode times 2.'
                             % (len(self._data, days_per_episode)))

        self._data['volume'] = self._normalize_volume(self._data['volume'])

        self._end_index = 0
        self._current_index = 0
        self._current_state = None
        self._current_hold_stock = 0
        self._current_money = default_money
        self._last_total = default_money

    def _normalize_volume(self, volume):
        return (volume - volume.min()) / (volume.max() - volume.min())

    def reset(self):
        self._current_index = np.random.randint(1, len(self._data) - self._days_per_episode - 10)
        self._end_index = self._current_index + self._days_per_episode
        self._current_hold_stock = 0
        self._current_money = self._default_money
        self._last_total = self._default_money

        d = self._data.iloc[self._current_index]
        self._current_state = np.asarray([
            d['open'], d['high'], d['low'], d['close'], d['volume'],
        ])

        return self._current_state

    def step(self, action):
        self._current_index += 1
        next_state = self._data.iloc[self._current_index]
        next_state = np.asarray([
            next_state['open'], next_state['high'], next_state['low'], next_state['close'],
            next_state['volume']
        ])
        current_price = next_state[-2]
        done = self._current_index > self._end_index

        extra_rewards = 0.0
        if action == 0: # buy, as much as possible
            n = self._buy()
            if n < 1:
                extra_rewards = -2.0
        elif action == 1: # sell, all
            n = self._sell()
            if n < 1:
                extra_rewards = -1.0
        elif action == 2: # wait, do nothing
            extra_rewards = -10.0
        else:
            raise ValueError('Unknown action %d' % action)

        reward = self._reward(current_price) + extra_rewards
        self._current_state = next_state
        return self._current_state, reward, done, None

    def _reward(self, current_price):
        total = current_price * self._current_hold_stock + self._current_money
        reward = total - self._last_total
        self._last_total = total
        return reward

    def _buy(self):
        """
        买入股票所用金额：10元/股×10000股=100000元；
        过户费：0.6元÷1000股×10000股=6元(沪市股票计算，深市为0)；
        交易佣金：100000×1‰=100元(其中包含经手费：10万*0.0696‰=6.96元；证管费：10万*0.02‰=2元)；
        买入总成本：100000元+6元+100元=100106元(买入10000股，每股10元，所需总资金)。
        """
        d = self._data.iloc[self._current_index]
        price = np.average([d['high'], d['low'], d['close']])
        n = self._current_money / price // 100 * 100 - 100
        if n > 0:
            fee1 = 0.6 / 1000 * n
            fee2 = n * price * 0.001
            self._current_money -= fee1 + fee2 + n * price
            self._current_hold_stock += n

        # days = self._days_per_episode - (self._end_index - self._current_index)
        # print("{} B {:.2f} x {}, {:.2f}".format(days, price, n, self._current_money))
        return n

    def _sell(self):
        """
        股票金额：10.10元/股×10000股=101000元；
　　     印花税：101000元×1‰=101元；
　　     过户费：0.6元÷1000股×10000股=6元；
　　     交易佣金：101000元×1‰=101元(其中包含经手费：101000元*0.0696‰=7.03元；证管费：101000元*0.02‰=2.02元)
　　     卖出后收入：101000元-101元-6元-101元=100792元；
        """
        d = self._data.iloc[self._current_index]
        price = np.average([d['high'], d['low'], d['close']])
        n = self._current_hold_stock
        if n > 0:
            fee1 = n * price * 0.001
            fee2 = 0.6 / 1000 * n
            fee3 = n * price * 0.001
            self._current_money += n * price - fee1 - fee2 - fee3
            self._current_hold_stock = 0
        # print("S {:.2f} x {}, {:.2f}".format(price, n, self._current_money))
        return n

    @property
    def reward_threshold(self):
        return self._default_money * 2.0
