import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tushare as ts
import numpy as np


class LSTMModel(nn.Module):

    def __init__(self, hidden_dim):
        super(LSTMModel, self).__init__()

        input_dim = 5
        self.hidden_dim = hidden_dim
        self.lstm_n_layers = 2

        # Linear transform layers
        self.linear1 = nn.Linear(in_features=input_dim, out_features=input_dim*2)
        self.linear2 = nn.Linear(in_features=input_dim*2, out_features=input_dim)

        self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=self.lstm_n_layers)

        self.linear3 = nn.Linear(in_features=self.hidden_dim, out_features=1)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.lstm_n_layers, 1, self.hidden_dim),
                torch.zeros(self.lstm_n_layers, 1, self.hidden_dim)) 

    def reset_hidden(self):
        self.hidden = self.init_hidden()

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)
        lstm_out, self.hidden = self.lstm(
            x.view(x.shape[0], 1, -1), self.hidden)
        
        out = self.linear3(lstm_out[-1, :, :])

        return F.sigmoid(out.view(1, 1))


def training_dataset():
    with h5py.File('/tmp/kline_lstm_data.hdf5', 'r') as hdf:
        seq = hdf['seq']
        target = hdf['target']

        for s, t in zip(seq, target):
            seq_tensor = torch.from_numpy(s)
            target_tensor = torch.zeros((1, 1))
            target_tensor[0, 0] = t * 1.0
            yield seq_tensor.view(60, 1, 5), target_tensor


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(hidden_dim=5)
    model.to(device)

    loss_function = nn.BCELoss()

    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print('Start training')
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        example_counter = 0
        for seq, target in training_dataset():
            seq = seq.to(device)
            target = target.to(device)
    
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.reset_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.

            # Step 3. Run our forward pass.
            out = model(seq)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(out, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            example_counter += 1

        print('Epoch %d/%d, loss=%.2f' % (epoch, num_epochs, epoch_loss / example_counter))


if __name__ == '__main__':
    main()
