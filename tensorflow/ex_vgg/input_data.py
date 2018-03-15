# -*- coding:utf-8 -*-
"""Generate input data.
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import datetime

from matplotlib.pylab import date2num
from codelist import sh50

def date_to_num(dates):
    num_time = []
    for date in dates:
        date_time = date.to_pydatetime()
        num_date = date2num(date_time)
        num_time.append(num_date)
    return num_time

def draw(data, filename):
    mat = data.as_matrix()
    num_time = date_to_num(mat[:, 0])
    mat[:, 0] = num_time

    fig, ax = plt.subplots(figsize=(5,5))
    mpf.candlestick_ochl(ax, mat, width=0.6, colorup='goldenrod', colordown='lightgreen', alpha=1.0)

    # 设置日期刻度旋转的角度
    plt.xticks(rotation=30)
    # x轴的刻度为日期
    ax.xaxis_date ()

    fig.savefig(filename)
    plt.close(fig)

filename = 'train_data/stock_raw.hdf5'

date_list = []

for code in sh50():
    df = pd.read_hdf(filename, 'A' + code)
    mat = df.as_matrix()
    date_list.extend(mat[:, 0])

date_list = list(set(date_list))
date_list.sort()

for code in sh50():
    df = pd.read_hdf(filename, 'A' + code)
    df['date'] = pd.to_datetime(df['date'])
    for i in range(len(date_list) - 121):
        start_date = date_list[i]
        buy_date = date_list[i + 120]
        sell_date = date_list[i + 121]

        mask = (df['date'] >= start_date) & (df['date'] < buy_date)

        sub_df = df.loc[mask]

        buy_day_df = df.loc[df['date'] == buy_date]
        sell_day_df = df.loc[df['date'] == sell_date]

        if not buy_day_df.empty and not sell_day_df.empty:
            buy_day_close = buy_day_df['close'].iloc[0]
            sell_day_high = sell_day_df['high'].iloc[0]
            label = (buy_day_close * 1.01 < sell_day_high) and 1 or 0

            kline_filename = 'train_data/%s_%s_%s_%d.png' % (code, start_date, buy_date, label)
            draw(sub_df, kline_filename)

            print (kline_filename, sell_date)
