import torch.nn as nn
import torch


class C3D(nn.Module):

    def __init__(self):
        super(C3D, self).__init__()

        x = 32
        self.conv1 = nn.Conv3d(6, x, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(x, 2*x, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(2*x, 4*x, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(4*x, 4*x, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(4*x, 8*x, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(8*x, 8*x, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(8*x, 8*x, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(8*x, 8*x, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.log_softmax(logits)

        return probs


class FencingModel(nn.Module):

    def __init__(self):
        super(FencingModel, self).__init__()
        self.c3d = C3D()


    def forward(self, frames_pose_tensor):
        seq_len = frames_pose_tensor.shape[1]
        seqs_to_count = [i for i in range(seq_len) if (i>=0 and i<=50 and i%2==0)]

        input = frames_pose_tensor[:, seqs_to_count[-16:]]#filter sequence
        batch_size, filtered_seq_len, people_num, img_channel_size, h, w = input.shape
        #(N, C_{in}, D_{in}, H_{in}, W_{in})
        input = input.view(batch_size, filtered_seq_len, people_num*img_channel_size, h, w)
        input = input.transpose(1, 2)
        res = self.c3d(input)
        return res
