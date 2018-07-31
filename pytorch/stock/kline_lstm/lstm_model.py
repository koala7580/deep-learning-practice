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
"""K line LSTM model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str,
    default=os.environ.get('TF_DATA_DIR', '/tmp/kline_lstm_data'),
    help='Directory to find the data file')


class Model(nn.Module):

    def __init__(self, hidden_dim):
        super(Model, self).__init__()

        input_dim = 5
        self.hidden_dim = hidden_dim
        self.lstm_n_layers = 1

        # Linear transform layers
        self.linear1 = nn.Linear(in_features=input_dim, out_features=input_dim)

        # LSTM cell
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=self.lstm_n_layers, batch_first=True)

        # OUtput linear map
        self.linear2 = nn.Linear(in_features=self.hidden_dim, out_features=1)

        self.sigmoid = nn.Sigmoid()

        self.hidden = self.init_hidden()

    def init_hidden(self, device=None):
        h0, c0 = (torch.zeros(self.lstm_n_layers, 1, self.hidden_dim),
                  torch.zeros(self.lstm_n_layers, 1, self.hidden_dim))
        if device:
            h0 = h0.to(device)
            c0 = c0.to(device)

        return h0, c0

    def forward(self, inputs):
        x = self.linear1(inputs)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        out = self.linear2(lstm_out[:, -1, :])
        return self.sigmoid(out)


def dataset(mode, split_date):
    full_path = os.path.join(FLAGS.data_dir, 'kline_lstm_data.h5')
    assert os.path.exists(full_path), 'Data file not exists'

    code_list = pd.read_hdf(full_path, 'code_list')

    for code in code_list:
        df = pd.read_hdf(full_path, 'SH' + code)

        days = 60
        index_pair = [(i, i + days) for i in range(len(df) - days - 2)]
        for index_beg, index_end in index_pair:
            df_data = df.iloc[index_beg:index_end]
            series_buy = df.iloc[index_end + 1]
            series_sell = df.iloc[index_end + 2]

            if mode == 'train' and series_sell['date'] >= split_date:
                break

            if mode == 'eval' and series_sell['date'] < split_date:
                continue

            price = series_buy['open']
            target = torch.zeros((1, 1))
            target[0, 0] = 1 if series_sell['high'] > price * 1.01 else 0

            keys = ['open', 'high', 'low', 'close', 'volume']
            seq = torch.zeros((len(df_data), len(keys)))

            for i in range(len(df_data)):
                for j, key in enumerate(keys):
                    if key == 'volume':
                        seq[i, j] = df_data.iloc[i][key] * 1e-4
                    else:
                        seq[i, j] = df_data.iloc[i][key]

            yield seq.view(1, len(df_data), len(keys)), target, {'buy': series_buy, 'sell': series_sell}


def main(_):
    assert os.path.exists(FLAGS.data_dir), 'Data dir not exists.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(hidden_dim=128)
    model.to(device)

    loss_function = nn.BCELoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print('Start training')
    num_epochs = 100
    split_date = '2018-07-25'
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        examples_in_epoch = 0
        for seq, target, info in dataset('train', split_date):
            seq = seq.to(device)
            target = target.to(device)
    
            model.zero_grad()

            model.hidden = model.init_hidden(device)

            out = model(seq)

            loss = loss_function(out, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            examples_in_epoch += 1
            
            print('Epoch %d/%d [%d]: %s loss = %.2f, out = %.2f' % (
                epoch, num_epochs, examples_in_epoch,
                info['buy']['code'], loss, out[0, 0]
            ), end='\r')

        print('\nEpoch %d/%d, %d examples, loss=%.2f' % (epoch, num_epochs, examples_in_epoch,
                                                       epoch_loss / examples_in_epoch))
        for seq, target, info in dataset('eval', split_date):
            seq = seq.to(device)
            target = target.to(device)
    
            model.zero_grad()

            model.hidden = model.init_hidden(device)

            with torch.no_grad():
                out = model(seq)

            print('{} {} -> {} {}; {:.2f}'.format(
                info['buy']['code'],
                info['buy']['date'], info['sell']['date'],
                'T' if target > 0 else 'F',
                out[0, 0] * 100.0
            ))


if __name__ == '__main__':
    FLAGS, not_parsed = parser.parse_known_args()
    main(not_parsed)
