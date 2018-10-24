from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from network.model import FencingModel
from network.dataloader import Dataset
from network.utils import AverageMeter, BinCounterMeter, accuracy


batch_size = 5
workers = 2
use_cuda = True
checkpoint = 'fencing_exp/bestmodel.pth'
labels_arr = np.array([0, 1, 2])
device = torch.device("cuda" if use_cuda else "cpu")

test_dataset = Dataset(is_train=0, txt_path='train_val_test_splitter/test.txt')
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=workers,
                                          pin_memory=True)

def evaluate(model, criterion):
    model.eval()
    total = 0
    acc_meter = AverageMeter()
    output_count_meter = BinCounterMeter(labels_arr)
    test_enum = tqdm(test_loader, desc='Test')

    with torch.no_grad():
        for pose_dsc, label in test_enum:
            pose_dsc, label = pose_dsc.to(device), label.to(device)

            # Forward
            output = model(pose_dsc)
            loss = criterion(output, label)

            total += loss.item()
            # calculate accuracy
            acc, pix = accuracy(output, label)
            acc_meter.update(acc, pix)

            _, output_as_ind = torch.max(output, dim=1)
            unique, counts = np.unique(output_as_ind.cpu().numpy(), return_counts=True)
            output_count_meter.update(unique, counts)

    loss_avg = total / len(test_loader)
    acc_avg = acc_meter.average() * 100
    avg_dist_arr = output_count_meter.get_distribution()
    print('====> Total test set loss: {:.4f}, acc: {:.4f}, dist: ({:.4f}, {:.4f}, {:.4f})'.format(loss_avg, acc_avg, *avg_dist_arr))
    return loss_avg, acc_avg


def main():
    startTime = datetime.now()
    model = FencingModel().to(device)
    model.load_state_dict(torch.load(checkpoint))
    criterion = nn.NLLLoss().to(device)

    _, _ = evaluate(model, criterion)

    print('startTime=' + str(startTime))
    print('endTime=' + str(datetime.now()))

if __name__ == '__main__':
    main()
