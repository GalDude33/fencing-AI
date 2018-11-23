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
        self.layers.add_module(name='conv2d0', module=torch.nn.Conv2d(kernel_size=4, stride=2, in_channels=3, out_channels=ngf, padding=1))

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
        res = self.layers(pose_as_img)
        return res


class FencingModel(nn.Module):

    def __init__(self):
        super(FencingModel, self).__init__()
        output_size = 2#3

        self.encoder = Encoder()

        self.rnn_left_player = ConvLSTM(input_size=(8, 16),
                                     input_dim=32 * 8,#256
                                     hidden_dim=[64],
                                     kernel_size=(3, 3),
                                     num_layers=1,
                                     batch_first=True,
                                     bias=True,
                                     return_all_layers=False)

        self.rnn_right_player = ConvLSTM(input_size=(8, 16),
                                        input_dim=32 * 8,#256
                                        hidden_dim=[64],
                                        kernel_size=(3, 3),
                                        num_layers=1,
                                        batch_first=True,
                                        bias=True,
                                        return_all_layers=False)

        self.rnn_overall = ConvLSTM(input_size=(8, 16),
                                         input_dim=64*2,
                                         hidden_dim=[64],
                                         kernel_size=(3, 3),
                                         num_layers=1,
                                         batch_first=True,
                                         bias=True,
                                         return_all_layers=False)

        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, output_size)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()


    def forward(self, frames_pose_tensor):
        batch_size, seq_len, people_num, img_channel_size, h, w = frames_pose_tensor.shape
        seqs_to_count = [i for i in range(seq_len) if (i >= 0 and i <= 52 and i % 2 == 0)]
        seqs_to_count = seqs_to_count[-16:]
        encodings_player0 = []
        encodings_player1 = []

        for seq_ind in seqs_to_count:
            for p in range(people_num):
                curr_encoding = self.encoder(frames_pose_tensor[:, seq_ind, p])
                if p==0:
                    encodings_player0.append(curr_encoding)
                else:
                    encodings_player1.append(curr_encoding)

        encodings_player0 = torch.stack(encodings_player0).transpose(0, 1)
        encodings_player1 = torch.stack(encodings_player1).transpose(0, 1)
        outputs_player0, state_player0 = self.rnn_left_player(encodings_player0)
        outputs_player1, state_player1 = self.rnn_right_player(encodings_player1)

        players_1_stage_outputs_combined = torch.cat([outputs_player0[0], outputs_player1[0]], dim=2)
        outputs_overall, state_overall = self.rnn_overall(players_1_stage_outputs_combined)

        #maybe add decoder here
        last_output = outputs_overall[0][:, -1].view(batch_size, -1)
        res = self.relu(self.fc1(self.dropout(last_output)))
        res = self.relu(self.fc2(self.dropout(res)))
        res = self.fc3(self.dropout(res))
        return F.log_softmax(res, dim=1)
