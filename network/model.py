import torch
import torch.nn as nn
import torch.nn.functional as F
from network.PoseEstimationUtils import NUM_LIMBS


class FencingModel(nn.Module):

    def __init__(self):
        super(FencingModel, self).__init__()
        output_size = 3
        #rnn properties
        self.rnn_type = 'lstm'#opt.rnn_type
        self.rnn_size = 128#opt.rnn_size
        self.num_layers = 1#opt.num_layers
        self.drop_prob_lm = 0#0.5#opt.drop_prob_lm
        self.input_size = 96#192#2 * NUM_LIMBS * 4

        self.rnn_left_player = getattr(
            nn,
            self.rnn_type.upper())(
            self.input_size,
            self.rnn_size,
            self.num_layers,
            dropout=self.drop_prob_lm)

        self.rnn_right_player = getattr(
            nn,
            self.rnn_type.upper())(
            self.input_size,
            self.rnn_size,
            self.num_layers,
            dropout=self.drop_prob_lm)

        self.rnn_sum = getattr(
            nn,
            self.rnn_type.upper())(
            128*2,
            256,
            1,
            dropout=self.drop_prob_lm)

        self.fc = nn.Linear(int(self.rnn_size)*2, output_size)


    def forward(self, frames_pose_tensor):
        _frames_pose_tensor = frames_pose_tensor
        _frames_pose_tensor = _frames_pose_tensor.transpose(0, 1)
        output_left, state_left = self.rnn_left_player(_frames_pose_tensor[:,:,0,:])
        output_right, state_right = self.rnn_right_player(_frames_pose_tensor[:,:,1,:])

        output_combined = torch.cat([output_left, output_right], dim=2)
        output, state =self.rnn_sum(output_combined)
        res = self.fc(output[-1])
        return F.log_softmax(res, dim=1)
