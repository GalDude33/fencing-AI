import torch
import torch.nn as nn
import torch.nn.functional as F

from network_pose_as_img.convlstm import ConvLSTM


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        ngf = 32#64
        self.layers = torch.nn.Sequential()

        # encoder_1: [batch, 128, 256, in_channels] => [batch, 64, 128, ngf]
        self.layers.add_module(name='conv2d0', module=torch.nn.Conv2d(kernel_size=4, stride=2, in_channels=1, out_channels=ngf, padding=1))

        layer_specs = [
            ngf * 2,  # encoder_2: [batch, 64, 128, ngf] => [batch, 32, 64, ngf * 2]
            ngf * 4,  # encoder_3: [batch, 32, 64, ngf * 2] => [batch, 16, 32, ngf * 4]
            ngf * 8,  # encoder_4: [batch, 16, 32, ngf * 4] => [batch, 8, 16, ngf * 8]
            #ngf * 8,  # encoder_5: [batch, 8, 16, ngf * 8] => [batch, 4, 8, ngf * 8]
            #ngf * 8,  # encoder_6: [batch, 4, 8, ngf * 8] => [batch, 2, 4, ngf * 8]
            #ngf * 8,  # encoder_7: [batch, 2, 4, ngf * 8] => [batch, 1, 2, ngf * 8]
            #ngf * 8,  # encoder_8: [batch, 2, 4, ngf * 8] => [batch, 1, 2, ngf * 8]
        ]

        layer_ind = 1
        for layer_specs_ind in range(len(layer_specs)):
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            in_channels = layer_specs[layer_specs_ind-1] if layer_specs_ind>0 else ngf
            out_channels = layer_specs[layer_specs_ind]
            self.layers.add_module(name='lrelu'+str(layer_ind), module=torch.nn.LeakyReLU(0.2))
            self.layers.add_module(name='conv2d'+str(layer_ind),
                                   module=torch.nn.Conv2d(kernel_size=4, stride=2, in_channels=in_channels, out_channels=out_channels, padding=1))
            self.layers.add_module(name='bn2d'+str(layer_ind), module=nn.BatchNorm2d(out_channels))
            layer_ind+=1

    def forward(self, pose_as_img):
        res = self.layers(pose_as_img.unsqueeze(1))
        return res


class FencingModel(nn.Module):

    def __init__(self):
        super(FencingModel, self).__init__()
        output_size = 2#3

        self.encoder = Encoder()

        self.rnn = ConvLSTM(input_size=(8, 16),
                         input_dim=32*8,
                         hidden_dim=[64],
                         kernel_size=(3, 3),
                         num_layers=1,
                         batch_first=True,
                         bias=True,
                         return_all_layers=False)

        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, output_size)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()


    def forward(self, frames_pose_tensor):
        seq_len = frames_pose_tensor.shape[1]
        batch_size = frames_pose_tensor.shape[0]
        encoding = []
        seqs_to_count = [i for i in range(seq_len) if ((i>=25 and i < 40 and i % 3 == 0) or (i >= 40 and i < 51))]

        for seq_ind in seqs_to_count:
            encoding.append(self.encoder(frames_pose_tensor[:, seq_ind]))

        clip_encodings = torch.stack(encoding).transpose(0,1)
        outputs, state = self.rnn(clip_encodings)

        last_output = outputs[0][:, -1].view(batch_size, -1)
        res = self.relu(self.fc1(self.dropout(last_output)))
        res = self.relu(self.fc2(self.dropout(res)))
        res = self.fc3(self.dropout(res))
        return F.log_softmax(res, dim=1)
