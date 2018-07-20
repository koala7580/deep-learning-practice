# -*- coding: utf-8 -*-
"""生成 K 线图，并保存到 TFRecord 中。
感谢 https://ipreacher.github.io/2017/candlestick/ 对 matplotlib.finance
进行了详细的解读。
因为 Matplotlib 2.2 以后，finance 模块被转换到了另一个 module 里，而这个包现在不在
pip 仓库里，所以需要这样安装：

    pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
"""
import io
import os
import sys
import argparse

import tushare as ts
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_finance as mpf
import tensorflow as tf
import skimage
import skimage.io as skimage_io
import skimage.color as skimage_color
import skimage.transform as skimage_transform

plt.switch_backend('agg')

sh50 = ts.get_sz50s()['code'].values

def read_stock_data(file_path, code):
    return pd.read_hdf(file_path, 'SH' + code)


def collect_date_list(file_path):
    dl_set = set()

    for code in sh50:
        df = read_stock_data(file_path, code)
        dl_set |= set(df['date'])

    dl = list(dl_set)
    dl.sort()

    return dl


def draw(data):
    fig = plt.figure(figsize=(21, 7))
    ax = fig.add_subplot(1, 1, 1)

    mpf.candlestick2_ochl(ax, data['open'], data['close'], data['high'], data['low'],
                            width=0.6, colorup='#804020', colordown='#208040',
                            alpha=1.0)

    with io.BytesIO() as image_buffer:
        fig.savefig(image_buffer, format='png')
        plt.close(fig)

        image_data = skimage_io.imread(image_buffer)
        image_rgb = skimage_color.rgba2rgb(image_data)
        image_resized = skimage_transform.resize(image_rgb, (224, 224 * 3), order=3, preserve_range=True)
        image_ubyte = skimage.img_as_ubyte(image_resized)

        return image_ubyte.tobytes()


def make_example(image, label, buy_date, sell_date, code):
    _int64_feature = lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))
    _bytes_feature = lambda x: tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))

    return tf.train.Example(features=tf.train.Features(
        feature={
            'image': _bytes_feature(image),
            'label': _int64_feature(label),
            'buy_date': _bytes_feature(bytes(buy_date, 'utf-8')),
            'sell_date': _bytes_feature(bytes(sell_date, 'utf-8')),
            'code': _bytes_feature(bytes(code, 'utf-8'))
        }))


def generate_kline_and_label(start_date, buy_date, sell_date, df):
    mask = (df['date'] >= start_date) & (df['date'] < buy_date)
    sub_df = df.loc[mask]
    buy_day_df = df.loc[df['date'] == buy_date]
    sell_day_df = df.loc[df['date'] == sell_date]

    img_bytes = draw(sub_df)

    if not buy_day_df.empty and not sell_day_df.empty:
        b = buy_day_df.iloc[0]
        # buy_price = (b['high'] + b['low'] + b['close']) / 3.0
        buy_price = b['open']
        sell_price = sell_day_df.iloc[0]['high']

        # NOTE: The label condition could be change here.
        ratio = (sell_price - buy_price) / buy_price * 100
        label = 1 if buy_price * 1.015 < sell_price else 0

        return img_bytes, label, ratio
    else:
        raise ValueError('Some data is empty.')


def main(args):
    hdf_file_path = os.path.join(args.data_dir, args.hdf)
    train_tfrecords_file_path = os.path.join(args.data_dir, 'stock_train.tfrecords')
    eval_tfrecords_file_path = os.path.join(args.data_dir, 'stock_eval.tfrecords')

    assert os.path.exists(hdf_file_path)

    if os.path.exists(train_tfrecords_file_path):
        print('Remove file: %s' % train_tfrecords_file_path)
        os.unlink(train_tfrecords_file_path)

    if os.path.exists(eval_tfrecords_file_path):
        print('Remove file: %s' % eval_tfrecords_file_path)
        os.unlink(eval_tfrecords_file_path)

    date_list = collect_date_list(hdf_file_path)

    total_count = 0
    train_count = 0
    eval_count = 0
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(train_tfrecords_file_path, options=options) as train_record_writer:
        with tf.python_io.TFRecordWriter(eval_tfrecords_file_path, options=options) as eval_record_writer:

            for code_index, code in enumerate(sh50):
                df = read_stock_data(hdf_file_path, code)

                for i in range(len(date_list) - 121):
                    start_date = date_list[i]
                    buy_date = date_list[i + 120]
                    sell_date = date_list[i + 121]

                    try:
                        img_bytes, label, ratio = generate_kline_and_label(start_date, buy_date, sell_date, df)
                    except ValueError as e:
                        print(e)
                        continue

                    example = make_example(img_bytes, label, buy_date, sell_date, code)

                    if i < len(date_list) - 121 - args.split_at:
                        train_record_writer.write(example.SerializeToString())
                        print('{:2d} {} B {} S {} L {} R {:5.2f} T'.format(code_index, code, buy_date, sell_date, label, ratio))
                        train_count += 1
                    else:
                        eval_record_writer.write(example.SerializeToString())
                        print('{:2d} {} B {} S {} L {} R {:5.2f} E'.format(code_index, code, buy_date, sell_date, label, ratio))
                        eval_count += 1
                    total_count += 1
    print('Total records: %d, %d train, %d eval' % (total_count, train_count, eval_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/tmp/stock',
        help='Directory to load and store stock data.')

    parser.add_argument(
        '--hdf',
        type=str,
        default='kline_raw.hdf',
        help='Filename to load kline raw data.')

    parser.add_argument(
        '--split-at',
        type=int,
        default=10,
        help='Default split index for train and eval dataset.')

    args = parser.parse_args()
    main(args)

