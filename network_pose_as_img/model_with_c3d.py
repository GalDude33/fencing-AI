import torch
import torch.nn as nn
import torch.nn.functional as F


class C3D(nn.Module):

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):

        print('\nstart:       ' + str(x.shape))#[1, 64, 16, 128, 256]
        h = self.relu(self.conv1(x))
        print('after conv1: '+str(h.shape))#[1, 64, 16, 64, 128]
        h = self.pool1(h)
        print('after pool1: '+str(h.shape))#[1, 128, 16, 64, 128]

        h = self.relu(self.conv2(h))
        print('after conv2: '+str(h.shape))#[1, 128, 8, 32, 64]
        h = self.pool2(h)
        print('after pool2: '+str(h.shape))#[1, 256, 8, 32, 64]

        h = self.relu(self.conv3a(h))
        print('after conv3a: '+str(h.shape))#[1, 256, 8, 32, 64]
        h = self.relu(self.conv3b(h))
        print('after conv3b: '+str(h.shape))#[1, 256, 4, 16, 32]
        h = self.pool3(h)
        print('after pool3: '+str(h.shape))#[1, 512, 4, 16, 32]

        h = self.relu(self.conv4a(h))
        print('after conv4a: '+str(h.shape))#[1, 512, 4, 16, 32]
        h = self.relu(self.conv4b(h))
        print('after conv4b: '+str(h.shape))#[1, 512, 2, 8, 16]
        h = self.pool4(h)
        print(h.shape)

        h = self.relu(self.conv5a(h))
        print(h.shape)
        h = self.relu(self.conv5b(h))
        print(h.shape)
        h = self.pool5(h)
        print(h.shape)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        print(h.shape)
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        print(h.shape)
        h = self.dropout(h)

        logits = self.fc8(h)
        print(h.shape)
        probs = self.log_softmax(logits)

        return probs


class FencingModel(nn.Module):

    def __init__(self):
        super(FencingModel, self).__init__()
        self.c3d = C3D()


    def forward(self, frames_pose_tensor):
        seq_len = frames_pose_tensor.shape[1]
        seqs_to_count = [i for i in range(seq_len) if ((i>=25 and i < 40 and i % 3 == 0) or (i >= 40 and i < 51))]

        input = frames_pose_tensor[:, seqs_to_count].unsqueeze(1)#(N, C_{in}, D_{in}, H_{in}, W_{in})
        res = self.c3d(input)
        return res
