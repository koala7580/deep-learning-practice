# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Downloads the stock data and stored into HDF5 file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tushare as ts
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str,
    default=os.environ.get('TF_DATA_DIR', '/tmp/kline_lstm_data'),
    help='Directory to download data and extract the tarball')

parser.add_argument(
    '--start_date', type=str, required=True,
    help='The date from where to start.')


def main(_):
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    start = datetime.now()
    sh50 = ts.get_sz50s()

    full_path = os.path.join(FLAGS.data_dir, 'kline_lstm_data.h5')

    sh50['code'].to_hdf(full_path, 'code_list')
    for code in sh50['code']:
        df = ts.get_k_data(code, FLAGS.start_date)
        df.to_hdf(full_path, 'SH' + code)

    time_elapsed = datetime.now() - start

    secs = time_elapsed.total_seconds()
    print('Elapsed: %2d:%2d:%2d' % (secs // 3600, secs % 3600 // 60, secs % 3600 % 60))


if __name__ == '__main__':
    FLAGS, not_parsed = parser.parse_known_args()
    main(not_parsed)
