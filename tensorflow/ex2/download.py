# -*- coding: utf-8 -*-
import tushare as ts
from datetime import datetime
from codelist import sh50

start_date = '2017-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
filename = 'train_data/stock_raw.hdf5'

for code in sh50():
	df = ts.get_k_data(code, start_date, end_date)
	df.to_hdf(filename, 'A' + code)
	print(code, 'is done')
