# -*- coding: utf-8 -*-
"""下载 K 线原始数据。
"""
import os
import argparse
from datetime import datetime

import tushare as ts

# 上证50 成分股
#
# 浦发银行 (600000)	民生银行 (600016)	宝钢股份 (600019)
# 中国石化 (600028)	南方航空 (600029)	中信证券 (600030)
# 招商银行 (600036)	保利地产 (600048)	中国联通 (600050)
# 上汽集团 (600104)	北方稀土 (600111)	万华化学 (600309)
# 华夏幸福 (600340)	康美药业 (600518)	贵州茅台 (600519)
# 山东黄金 (600547)	绿地控股 (600606)	海通证券 (600837)
# 伊利股份 (600887)	江苏银行 (600919)	东方证券 (600958)
# 招商证券 (600999)	大秦铁路 (601006)	中国神华 (601088)
# 兴业银行 (601166)	北京银行 (601169)	中国铁建 (601186)
# 国泰君安 (601211)	上海银行 (601229)	农业银行 (601288)
# 中国平安 (601318)	交通银行 (601328)	新华保险 (601336)
# 中国中铁 (601390)	工商银行 (601398)	中国太保 (601601)
# 中国人寿 (601628)	中国建筑 (601668)	中国电建 (601669)
# 华泰证券 (601688)	中国中车 (601766)	中国交建 (601800)
# 光大银行 (601818)	中国石油 (601857)	浙商证券 (601878)
# 中国银河 (601881)	中国核电 (601985)	中国银行 (601988)
# 中国重工 (601989)	洛阳钼业 (603993)
SH50 = ['600000', '600016', '600019', '600028', '600029',
		'600030', '600036', '600048', '600050', '600104',
		'600547', '600606', '600837', '600887', '600919',
		'600111', '600309', '600340', '600518', '600519',
		'600958', '600999', '601006', '601088', '601166',
		'601169', '601186', '601211', '601229', '601288',
		'601318', '601328', '601336', '601390', '601398',
		'601601', '601628', '601668', '601669', '601688',
		'601766', '601800', '601818', '601857', '601878',
		'601881', '601985', '601988', '601989', '603993']


def main(args):
	if not os.path.exists(args.data_dir):
		print('Create folder: {}'.format(args.data_dir))
		os.mkdir(args.data_dir)
	
	full_path = os.path.join(args.data_dir, args.hdf)
	if os.path.exists(full_path):
		print('Remove file: {}'.format(full_path))		
		os.unlink(full_path)

	TODAY = datetime.today().strftime('%Y-%m-%d')

	for code in SH50:
		df = ts.get_k_data(code, args.start_date, TODAY)
		df.to_hdf(full_path, 'SH' + code)
		print(code, 'is done')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data_dir',
		type=str,
		default='/tmp/stock',
		help='Directory to download and store stock data to.')

	parser.add_argument(
		'--start_date',
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
