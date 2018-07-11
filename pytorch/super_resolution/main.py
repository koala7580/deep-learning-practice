"""Simple Supper-Resolution Algorithm
"""
from __future__ import division
from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import skimage.io as skimage_io
import numpy as np
from sklearn.preprocessing import normalize


class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()

        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv1_1 = nn.Conv2d(3, 32, (1, 1), 1, 0) # out 224 x 224
        self.conv1_2 = nn.Conv2d(3, 32, (3, 3), 1, (1, 1)) # out 224 x 224
        self.conv1_3 = nn.Conv2d(3, 32, (5, 5), 1, (2, 2)) # out 224 x 224

        self.conv2 = nn.Conv2d(32 * 3, 3 * 4, (1, 1), 1, 0)

        self.conv3_1 = nn.Conv2d(3, 1, (1, 1), 2)
        self.conv3_2 = nn.Conv2d(3, 1, (3, 3), 2, (1, 1))
        self.conv3_3 = nn.Conv2d(3, 1, (5, 5), 2, (2, 2))

        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, image):
        x_1 = self.conv1_1(image)
        x_2 = self.conv1_2(image)
        x_3 = self.conv1_3(image)
        x = self.activation(torch.cat([x_1, x_2, x_3], dim=1))

        x = self.tanh(self.conv2(x))

        sr_image = self.pixel_shuffle(x)

        x_1 = self.conv3_1(sr_image)
        x_2 = self.conv3_2(sr_image)
        x_3 = self.conv3_3(sr_image)
        x = self.activation(torch.cat([x_1, x_2, x_3], dim=1))

        return x, sr_image


def save_image(image, suffix):
    image = image.view(3, 448, 448).detach().numpy()
    image = np.transpose(image, [1, 2, 0])
    skimage_io.imsave('result_{}.png'.format(suffix), image)


def main(opt):
    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if opt.cuda else "cpu")

    model = SRNet().to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    image = skimage_io.imread('lena.jpg')
    image = np.transpose(image, [2, 0, 1])

    epoch_loss = 0
    total_iterations = 1000 * 5
    for iteration in range(total_iterations):
        image_tensor = torch.from_numpy(image).type(torch.float32)
        image_tensor = image_tensor.view(1, -1, image.shape[1], image.shape[2])
        image_tensor.to(device)

        optimizer.zero_grad()
        out, sr_image = model(image_tensor)

        if iteration % 100 == 0:
            save_image(sr_image, iteration)

        loss = criterion(out, image_tensor)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(0, iteration, total_iterations, loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(0, epoch_loss / total_iterations))
    save_image(sr_image, 'final')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple Super Res Example')
    # parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    opt = parser.parse_args()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    main(opt)
