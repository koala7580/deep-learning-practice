# -*- coding: utf-8 -*-
"""生成 K 线图，并保存到 TFRecord 中。
"""
import io
import os
import argparse
import datetime

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import skimage.io as skimage_io
import tensorflow as tf
from matplotlib.pylab import date2num
from skimage.color import rgba2rgb

from download_data import SH50


def read_stock_data(file_path, code):
	return pd.read_hdf(file_path, 'SH' + code)


def collect_date_list(file_path):
	dl = []

	for code in SH50:
		df = read_stock_data(file_path, code)
		mat = df.as_matrix()
		dl.extend(mat[:, 0])

	dl = list(set(dl))
	dl.sort()

	return dl


def date_to_num(date):
	return date2num(date.to_pydatetime())


def draw(data):
	mat = data.as_matrix()
	mat[:, 0] = [date_to_num(x) for x in mat[:, 0]]

	fig, ax = plt.subplots(figsize=(5,5))
	mpf.candlestick_ochl(ax, mat, width=0.6,
						 colorup='#FFAABB',
						 colordown='#CCFFDD',
						 alpha=1.0)

	# 设置日期刻度旋转的角度
	plt.xticks(rotation=30)

	# x轴的刻度为日期
	ax.xaxis_date()

	with io.BytesIO() as image_buffer:
		fig.savefig(image_buffer, format='png')
		plt.close(fig)

		image = skimage_io.imread(image_buffer)
		image = rgba2rgb(image)
		return image.tobytes()


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_example(image, label, buy_date, sell_date, code):
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

	if not buy_day_df.empty and not sell_day_df.empty:
		buy_day_close = buy_day_df['close'].iloc[0]
		sell_day_high = sell_day_df['high'].iloc[0]

		# NOTE: The label condition could be change here.
		label = (buy_day_close * 1.015 < sell_day_high) and 1 or 0

		img_bytes = draw(sub_df)
		return img_bytes, label
	else:
		if buy_day_df.empty:
			raise ValueError('Buy day has no data.')

		if sell_day_df.empty:
			raise ValueError('Sell day has no data')


def main(args):
	hdf_file_path = os.path.join(args.data_dir, args.hdf)
	tfrecords_file_path = os.path.join(args.data_dir, args.tfrecords)

	assert os.path.exists(hdf_file_path)

	if os.path.exists(tfrecords_file_path):
		print('Remove file: {}'.format(tfrecords_file_path))
		os.unlink(tfrecords_file_path)

	date_list = collect_date_list(hdf_file_path)

	with tf.python_io.TFRecordWriter(tfrecords_file_path) as record_writer:
		for code in SH50:
			df = read_stock_data(hdf_file_path, code)
			df['date'] = pd.to_datetime(df['date'])

			for i in range(len(date_list) - 121):
				start_date = date_list[i]
				buy_date = date_list[i + 120]
				sell_date = date_list[i + 121]

				try:
					img_bytes, label = generate_kline_and_label(start_date, buy_date, sell_date, df)
				except ValueError:
					continue

				example = make_example(img_bytes, label, buy_date, sell_date, code)
				record_writer.write(example.SerializeToString())
				print('{} B {} S {} L {}'.format(code, buy_date, sell_date, label))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data_dir',
		type=str,
		default='/tmp/stock',
		help='Directory to load and store stock data.')

	parser.add_argument(
		'--hdf',
		type=str,
		default='kline_raw.hdf',
		help='Filename to load kline raw data.')

	parser.add_argument(
		'--tfrecords',
		type=str,
		default='kline.tfrecords',
		help='Filename to store training data.')

	args = parser.parse_args()
	main(args)
