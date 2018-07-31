# -*- coding: utf-8 -*-
import h5py
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime

TMP_HDF_FILE = '/tmp/kline_raw.hdf5'
OUTPUT_HDF_FILE = '/tmp/kline_lstm_data.hdf5'
START_DATE = '2017-01-01'


def download_data(code_list):
    for code in code_list:
        df = ts.get_k_data(code, START_DATE)
        df.to_hdf(TMP_HDF_FILE, 'SH' + code)


def date_list(code_list):
    date_set = set()
    for code in code_list:
        df = pd.read_hdf(TMP_HDF_FILE, 'SH' + code)

        for d in df['date']:
            date_set.add(d)

    _list = list(date_set)
    _list.sort()
    return _list


def generate_data(code_list, trading_dates, hdf):
    hdf.create_dataset('code_list', data=[x.encode('utf-8') for x in code_list])
    hdf.create_dataset('date_list', data=[x.encode('utf-8') for x in trading_dates])

    capacity = 100
    seq_dset = hdf.create_dataset("seq", (capacity, 60, 5), maxshape=(None, 60, 5))
    target_dset = hdf.create_dataset("target", (capacity,), maxshape=(None,), dtype='i8')

    dset_index = 0
    for code_index, code in enumerate(code_list):
        df = pd.read_hdf(TMP_HDF_FILE, 'SH' + code)

        days = 60
        index_pair = [(i, i + days) for i in range(len(df) - days - 2)]
        for index_beg, index_end in index_pair:
            df_data = df.iloc[index_beg:index_end]
            series_buy = df.iloc[index_end + 1]
            series_sell = df.iloc[index_end + 2]

            price = series_buy['open']
            target = 1 if series_sell['high'] > price * 1.01 else 0

            keys = ['open', 'high', 'low', 'close', 'volume']
            seq = np.zeros((len(df_data), len(keys)))

            for i in range(len(df_data)):
                for j, key in enumerate(keys):
                    if key == 'volume':
                        seq[i, j] = df_data.iloc[i][key] * 1e-4
                    else:
                        seq[i, j] = df_data.iloc[i][key]

            seq_dset[dset_index, :, :] = seq
            target_dset[dset_index] = target
            dset_index += 1

            if dset_index >= capacity:
                capacity = capacity + 100
                seq_dset.resize((capacity, 60, 5))
                target_dset.resize((capacity,))

            print('%d/%d %s %s -> %s %s' % (
                code_index, len(code_list),
                code, series_buy['date'], series_sell['date'],
                'T' if target > 0 else 'F'
            ))


def main():
    start = datetime.now()
    sh50 = ts.get_sz50s()

    print('Download data')
    download_data(sh50['code'].values)

    print('Get trading dates')
    trading_dates = date_list(sh50['code'].values)

    print('Generate data')
    with h5py.File(OUTPUT_HDF_FILE, 'w') as hdf:
        generate_data(sh50['code'].values, trading_dates, hdf)

    time_elapsed = datetime.now() - start

    secs = time_elapsed.total_seconds()
    print('Elapsed: %2d:%2d:%2d' % (secs // 3600, secs % 3600 // 60, secs % 3600 % 60))


if __name__ == '__main__':
    main()
