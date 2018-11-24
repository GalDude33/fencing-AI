
import torch.nn as nn
import torch


class C3D(nn.Module):

    def __init__(self, input_channel_num):
        super(C3D, self).__init__()

        x = 32
        self.conv1 = nn.Conv3d(input_channel_num, x, kernel_size=(3, 3, 3), padding=(1, 1, 1))
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

        self.conv6a = nn.Conv3d(8*x, 8*x, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv6b = nn.Conv3d(8*x, 8*x, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool6 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 3)

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

        h = self.relu(self.conv6a(h))
        h = self.relu(self.conv6b(h))
        h = self.pool6(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.log_softmax(logits)

        return probs



class FencingModel(nn.Module):

    def __init__(self, use_optical_flow, use_pose_img=True):
        super(FencingModel, self).__init__()
        self.use_optical_flow = use_optical_flow
        self.use_pose_img = use_pose_img

        self.input_channel_num = 0
        if self.use_pose_img:
            self.input_channel_num += 6
        if self.use_optical_flow:
            self.input_channel_num += 4

        self.c3d = C3D(self.input_channel_num)

    def forward(self, frames_pose_tensor, frames_optical_flow_tensor=None):
        if self.use_optical_flow:
            flow_input = frames_optical_flow_tensor
            batch_size, seq_len, people_num, flow_channel_size, h, w = flow_input.shape
            flow_input = flow_input.view(batch_size, seq_len, people_num * flow_channel_size, h, w)
            flow_input = flow_input.transpose(1, 2)

        if self.use_pose_img:
            pose_input = frames_pose_tensor
            batch_size, seq_len, people_num, img_channel_size, h, w = pose_input.shape
            pose_input = pose_input.view(batch_size, seq_len, people_num*img_channel_size, h, w)
            pose_input = pose_input.transpose(1, 2)

        input = []
        if self.use_pose_img and self.use_optical_flow:
            input = torch.cat([pose_input, flow_input], dim=1)
        elif self.use_optical_flow:
            input = flow_input
        else:
            input = pose_input

        # (N, C_{in}, D_{in}, H_{in}, W_{in})
        res = self.c3d(input)
        return res



