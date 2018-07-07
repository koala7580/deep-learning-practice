# -*- coding: utf-8 -*-
"""下载 K 线原始数据。
"""
import os
import argparse
from datetime import datetime

import tushare as ts


def main(args):
	if not os.path.exists(args.data_dir):
		print('Create folder: {}'.format(args.data_dir))
		os.mkdir(args.data_dir)

	full_path = os.path.join(args.data_dir, args.hdf)
	if os.path.exists(full_path):
		print('Remove file: {}'.format(full_path))
		os.unlink(full_path)

	TODAY = datetime.today().strftime('%Y-%m-%d')

	sh50 = ts.get_sz50s()['code'].values
	for code in sh50:
		df = ts.get_k_data(code, args.start_date, TODAY)
		df.to_hdf(full_path, 'SH' + code, complevel=6)
		print(code, 'is done')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data-dir',
		type=str,
		default='/tmp/stock',
		help='Directory to download and store stock data to.')

	parser.add_argument(
		'--start-date',
		type=str,
		default='2017-01-01',
		help='Start date to download.')

	parser.add_argument(
		'--hdf',
		type=str,
		default='kline_raw.hdf',
		help='Filename to download and store stock data to.')

	args = parser.parse_args()
	main(args)
