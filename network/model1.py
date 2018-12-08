import torch.nn as nn
import torch.nn.functional as F
from network.PoseEstimationUtils import NUM_POINTS
from network.temporal_model import TemporalModel


class FencingModel(nn.Module):

    def __init__(self):
        super(FencingModel, self).__init__()
        output_size = 3
        architecture = '3,3,3'
        filter_widths = [int(x) for x in architecture.split(',')]
        self.temporal_model = TemporalModel(num_joints_in=NUM_POINTS, in_features=2*2, num_joints_out=output_size, filter_widths=filter_widths,
                                            causal=False, dropout=0.5, channels=1024, dense=False)


    def forward(self, frames_pose_tensor):
        batch_size, seq_len, people_num, limbs_num, limb_feature_size = frames_pose_tensor.shape
        _frames_pose_tensor = frames_pose_tensor.permute(0,1,3,2,4)
        _frames_pose_tensor = frames_pose_tensor.view(batch_size, seq_len, limbs_num, people_num*limb_feature_size)

        res = self.temporal_model(_frames_pose_tensor)
        res = res.squeeze()
        return F.log_softmax(res, dim=1)
