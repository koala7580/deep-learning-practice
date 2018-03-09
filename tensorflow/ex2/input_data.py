# -*- coding:utf-8 -*-
"""Download and draw K graph.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import datetime

from matplotlib.pylab import date2num

def date_to_num(dates):
    num_time = []
    for date in dates:
        date_time = datetime.datetime.strptime(date,'%Y-%m-%d')
        num_date = date2num(date_time)
        num_time.append(num_date)
    return num_time


def draw(data, filename):
    mat = data.as_matrix()
    num_time = date_to_num(mat[:, 0])
    mat[:, 0] = num_time

    fig, ax = plt.subplots(figsize=(15,5))
    # fig.subplots_adjust(bottom=0.5)
    mpf.candlestick_ochl(ax, mat, width=0.6, colorup='g', colordown='r', alpha=1.0)
    plt.grid(True)

    # 设置日期刻度旋转的角度
    plt.xticks(rotation=30)
    # x轴的刻度为日期
    ax.xaxis_date ()

    fig.savefig(filename)

data = download(['002739'], '2017-01-01')
data[0].to_csv('train_data/002739.csv')
draw(data[0], 'train_data/k-line.png')