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

"""Downloads and convert the kline data and convert to TFRecord.
Draw finance chart: https://ipreacher.github.io/2017/candlestick/
With MACD and RSI: https://zhuanlan.zhihu.com/p/29519040
TALib: http://ta-lib.org/function.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import io

import tushare as ts
import tensorflow as tf
import mpl_finance as mpf
import matplotlib.pyplot as plt
import multiprocessing as mp

from datetime import datetime
from six.moves import queue
from collections import namedtuple

plt.switch_backend('agg')

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str,
    default=os.environ.get('TF_DATA_DIR', '/tmp/kline_data'),
    help='Directory to download data and store the data')

parser.add_argument(
    '--start_date', type=str, required=True,
    help='The start date')

parser.add_argument(
    '--split_date', type=str, required=True,
    help='The date before which are used for training.')


TaskTuple = namedtuple('TaskTuple', ['data_type', 'code_index', 'code',
                                     'df_chart', 'df_buy', 'df_sell'])
ResultTuple = namedtuple('ResultTuple', ['data_type', 'code_index', 'code',
                                         'image', 'label', 'buy_date',
                                         'sell_date', 'ratio'])


def worker_producer(q_in, q_out):
    while True:
        try:
            item = q_in.get(timeout=2)
        except queue.Empty:
            break

        image = draw_kline(item.df_chart)

        # buy_price = (b['high'] + b['low'] + b['close']) / 3.0
        buy_price = item.df_buy['open']
        sell_price = item.df_sell['high']

        ratio = (sell_price - buy_price) / buy_price * 100
        label = 1 if buy_price * 1.015 < sell_price else 0

        q_out.put(ResultTuple(
            data_type=item.data_type,
            code_index=item.code_index,
            code=item.code,
            image=image,
            label=label,
            buy_date=item.df_buy['date'],
            sell_date=item.df_sell['date'],
            ratio=ratio
        ))

        q_in.task_done()


def draw_kline(df):
    fig = plt.figure(figsize=(21, 7))
    ax = fig.add_subplot(1, 1, 1)

    mpf.candlestick2_ochl(ax, df['open'], df['close'], df['high'], df['low'],
                          colorup='#804020', colordown='#208040',
                          alpha=1.0)

    with io.BytesIO() as image_buffer:
        fig.savefig(image_buffer, format='png')
        plt.close(fig)
        return image_buffer.getvalue()


def worker_resize(q_in, q_out):
    with tf.device('/gpu:0'):
        image_ph = tf.placeholder(tf.string, shape=())
        decoded = tf.image.decode_png(image_ph, 3)
        resized = tf.image.resize_bilinear(tf.expand_dims(decoded, 0), (224, 224*3))
        resized = tf.cast(tf.squeeze(resized, 0), tf.uint8)

    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=os.cpu_count(),
        intra_op_parallelism_threads=os.cpu_count(),
        allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    session = tf.Session(config=session_config)
    while True:
        try:
            item = q_in.get(timeout=2)
        except queue.Empty:
            break

        image = session.run(resized, feed_dict={image_ph: item.image})
        q_out.put(item._replace(image=image))
        q_in.task_done()


def worker_write(q_in, code_list):
    total_code = len(code_list)
    complete_code = set()

    with create_writer('train') as train_writer:
        with create_writer('validation') as eval_writer:
            while True:
                try:
                    item = q_in.get(timeout=2)
                except queue.Empty:
                    break

                example = make_example(item.image, item.label,
                                       item.buy_date, item.sell_date, item.code)
                if item.data_type == 'T':
                    train_writer.write(example.SerializeToString())
                else:
                    eval_writer.write(example.SerializeToString())

                complete_code.add(item.code)

                print('%d/%d %s %s -> %s %d %5.2f %s' %
                      (len(complete_code), total_code,
                       item.code, item.buy_date, item.sell_date,
                       item.label, item.ratio, item.data_type))

                q_in.task_done()


def make_example(image, label, buy_date, sell_date, code):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    return tf.train.Example(features=tf.train.Features(
        feature={
            'image': _bytes_feature(image.tobytes()),
            'label': _int64_feature(label),
            'buy_date': _bytes_feature(bytes(buy_date, 'utf-8')),
            'sell_date': _bytes_feature(bytes(sell_date, 'utf-8')),
            'code': _bytes_feature(bytes(code, 'utf-8'))
        }))


def convert_all_code(code_list):
    processes, q1, q2, q3 = create_processes(code_list)

    train_count = 0
    eval_count = 0
    for code_index, code in enumerate(code_list):
        df = ts.get_k_data(code, FLAGS.start_date)

        days = 120
        index_pair = [(i, i + days) for i in range(len(df) - days - 2)]
        for index_beg, index_end in index_pair:
            df_buy = df.iloc[index_end + 1]
            df_sell = df.iloc[index_end + 2]
            df_chart = df.iloc[index_beg:index_end]

            if df_buy['date'] < FLAGS.split_date:
                data_type = 'T'
                train_count += 1
            else:
                data_type = 'E'
                eval_count += 1

            q1.put(TaskTuple(
                data_type=data_type,
                code_index=code_index,
                code=code,
                df_chart=df_chart,
                df_buy=df_buy,
                df_sell=df_sell
            ))

    wait_for_done(processes, q1, q2, q3)

    total_count = train_count + eval_count
    print('Total records: %d, %d train, %d eval' % (total_count, train_count, eval_count))


def create_processes(code_list):
    q1 = mp.JoinableQueue()
    q2 = mp.JoinableQueue()
    q3 = mp.JoinableQueue()

    num_producer_processes = os.cpu_count()
    num_resize_processes = os.cpu_count()

    processes = []
    # Worker for generating image and label
    for _ in range(num_producer_processes):
        p = mp.Process(target=worker_producer, args=(q1, q2))
        p.start()
        processes.append(p)

    # Worker for resize image
    for _ in range(num_resize_processes):
        p = mp.Process(target=worker_resize, args=(q2, q3))
        p.start()
        processes.append(p)

    # Worker for writing to tfrecord
    p = mp.Process(target=worker_write, args=(q3, code_list))
    p.start()
    processes.append(p)

    return processes, q1, q2, q3


def wait_for_done(processes, q1, q2, q3):
    q1.join()
    q2.join()
    q3.join()

    for p in processes:
        p.join()


def create_writer(subset):
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    filename = os.path.join(FLAGS.data_dir, 'kline_{}.tfrecords'.format(subset))

    if os.path.exists(filename):
        os.unlink(filename)

    return tf.python_io.TFRecordWriter(filename, options=options)


def main(_):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    sh50 = ts.get_sz50s()

    start = datetime.now()
    convert_all_code(sh50['code'].values)
    time_elapsed = datetime.now() - start

    secs = time_elapsed.total_seconds()
    print('Elapsed: %2d:%2d:%2d' % (secs // 3600, secs % 3600 // 60, secs % 3600 % 60))


if __name__ == '__main__':
    FLAGS, not_parsed = parser.parse_known_args()
    # tf.app.run(argv=[sys.argv[0]] + not_parsed)
    main(not_parsed)
