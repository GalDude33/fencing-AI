import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        ngf = 64
        self.layers = torch.nn.Sequential()

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        self.layers.add_module(name='conv2d0', module=torch.nn.Conv2d(kernel_size=4, stride=2, in_channels=1, out_channels=ngf))

        layer_specs = [
            ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        layer_ind = 1
        for layer_specs_ind in range(len(layer_specs)):
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            in_channels = layer_specs[layer_specs_ind-1] if layer_specs_ind>0 else ngf
            out_channels = layer_specs[layer_specs_ind]
            self.layers.add_module(name='lrelu'+str(layer_ind), module=torch.nn.LeakyReLU(0.2))
            self.layers.add_module(name='conv2d'+str(layer_ind),
                                   module=torch.nn.Conv2d(kernel_size=4, stride=2, in_channels=in_channels, out_channels=out_channels))
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


    def forward(self, frames_pose_tensor):
        seq_len = frames_pose_tensor.shape[1]
        encoding = []

        for seq_ind in range(seq_len):
            encoding.append(self.encoder(frames_pose_tensor[:, seq_ind]))
