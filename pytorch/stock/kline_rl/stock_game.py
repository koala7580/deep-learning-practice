from __future__ import division

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import tushare as ts


class StockGame(gym.Env):
    """Stock Game
    The goal of stock game is to earn more money.

    Each step player could choose:
    0 - wait, do nothing
    1 - buy, spend all money to by the stock
    2 - sell, sell all stock the player hold

    After each step the agent receives an observation of the next day k line:
    open
    high
    low
    close
    volume
    where volume is normalized.

    The rewards is calculated as:
    hold_stock * close_price + rest_money - previous_total

    If the player wait, will receive a little minus reward.
    If do sell but nothing hold, will receive -1.0
    If do buy but not enough money, will receive -1.0
    """
    def __init__(self, code, start_date, end_date, start_money):
        self._code = code
        self._start_date = start_date
        self._end_date = end_date
        self._start_money = start_money

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([-1.0, -10.0, -1.0, 0.0, -1.0]),
                                            high=np.array([1.0,  10.0,  1.0, 1.0,  1.0]), dtype=np.float32)

        self._data = ts.get_k_data(code, start_date, end_date)
        self._data['volume'] = self._normalize_volume(self._data['volume'])

        self._hold_stock = 0
        self._money = start_money
        self._last_total = start_money

        self.seed()
        self.reset()

    def _normalize_volume(self, volume):
        return (volume - volume.min()) / (volume.max() - volume.min())

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        _index1 = self._current_index
        _index2 = self._current_index + 1
        d1 = self._data.iloc[_index1]
        d2 = self._data.iloc[_index2]
        self.observation = self._normalize_kline(d1, d2)

        done = _index2 >= len(self._data) - 1

        if not done:
            extra_reward = 0
            if action == 0: # wait, do nothing
                extra_reward = -0.01
            elif action == 1: # buy
                count = self._buy(d2['close'])
                if count <= 0:
                    extra_reward = -1.0
            elif action == 2: # sell
                if self._hold_stock > 0:
                    self._sell(d2['high'])
                else:
                    extra_reward = -1.0
            else:
                raise ValueError('action value error: {}'.format(action))

            reward = self._reward(d2['close']) + extra_reward
            return self.observation, reward, done, { 'hold_stock': self._hold_stock, 'money': self._money }
        else: # sell  all the last day
            if self._hold_stock > 0:
                self._sell(d2['close'])
            reward = self._reward(d2['price'])
            return self.observation, reward, done, { 'hold_stock': self._hold_stock, 'money': self._money }

    def _reward(self, price):
        total = self._hold_stock * price + self._money
        self._last_total = total
        return total

    def _buy(self, price):
        """
        买入股票所用金额：10元/股×10000股=100000元；
        过户费：0.6元÷1000股×10000股=6元(沪市股票计算，深市为0)；
        交易佣金：100000×1‰=100元(其中包含经手费：10万*0.0696‰=6.96元；证管费：10万*0.02‰=2元)；
        买入总成本：100000元+6元+100元=100106元(买入10000股，每股10元，所需总资金)。
        """
        def rest_money(n):
            fee1 = min(n / 1000 * 0.6, 2)
            fee2 = min(n * price * 0.001, 2)
            return self._money - fee1 - fee2 - n * price

        count = self._money / price // 100 * 100 - 200
        while True:
            n = count + 100
            if rest_money(n) < 0:
                break
            count = n

        if count > 0:
            self._money = rest_money(count)
            self._hold_stock += count

        return count

    def _sell(self, price):
        """
        股票金额：10.10元/股×10000股=101000元；
　　     印花税：101000元×1‰=101元；
　　     过户费：0.6元÷1000股×10000股=6元；
　　     交易佣金：101000元×1‰=101元(其中包含经手费：101000元*0.0696‰=7.03元；证管费：101000元*0.02‰=2.02元)
　　     卖出后收入：101000元-101元-6元-101元=100792元；
        """
        n = self._hold_stock
        fee1 = n * price * 0.001
        fee2 = min(n / 1000 * 0.6, 2)
        fee3 = min(n * price * 0.001, 2)
        self._money += n * price - fee1 - fee2 - fee3
        self._hold_stock = 0

    def reset(self):
        self._current_index = self.np_random.randint(1, len(self._data) // 2)
        self._hold_stock = 0
        self._money = self._start_money
        self._last_total = self._start_money

        _index1 = self._current_index - 1
        _index2 = self._current_index

        d1 = self._data.iloc[_index1]
        d2 = self._data.iloc[_index2]

        self.observation = self._normalize_kline(d1, d2)

        return self.observation

    def _normalize_kline(self, d1, d2):
        l_l = (d2['low'] - d1['low']) / d1['low']
        h_l = d2['high'] - d2['low']
        delta_l = d2['low'] - d1['low']
        delta_h = d2['high'] - d1['high']

        v1 = delta_h / delta_l if np.fabs(delta_l) > 1e-4 else 0.0 # the else value should be consider again
        v2 = (d2['close'] - d2['open']) / h_l if h_l > 0 else 0.0
        v3 = (d2['high'] - d2['close']) / h_l if h_l > 0 else 0.0

        return np.array([
            l_l, v1, v2, v3, d2['volume']
        ])
