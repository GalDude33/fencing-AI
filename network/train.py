from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import json
import numpy as np
from network.model_with_dilated_conv import FencingModel
from network.dataloader import Dataset
from network.utils import AverageMeter, BinCounterMeter, adjust_learning_rate, accuracy, check_grad


batch_size = 64
workers = 16
use_cuda = True
learning_rate = 1e-3
checkpoint = ''
expName = 'fencing_exp_coordinates_b_64_facebook_net_lr_1e-3_dropout_0.5'
epochs = 100
adjust_lr_manually = 1
max_not_improving_epochs = 10
clip_grad = 0.5
ignore_grad = 10000.0
labels_arr = np.array([0, 1, 2])
device = torch.device("cuda" if use_cuda else "cpu")

descriptor_dir = '/media/rabkinda/DATA/fencing/FinalPoseEstimationResults/jsons*'

valid_dataset = Dataset(mode='val', txt_path='network/train_val_test_splitter/val.txt', descriptor_dir=descriptor_dir)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                         batch_size=batch_size,
                         num_workers=workers,
                         pin_memory=True)

test_dataset = Dataset(mode='test', txt_path='network/train_val_test_splitter/test.txt', descriptor_dir=descriptor_dir)
test_loader = torch.utils.data.DataLoader(test_dataset,
                         batch_size=batch_size,
                         num_workers=workers,
                         pin_memory=True)

train_dataset = Dataset(mode='train', txt_path='network/train_val_test_splitter/train.txt', descriptor_dir=descriptor_dir)
train_loader = torch.utils.data.DataLoader(train_dataset,
                         batch_size=batch_size,
                         num_workers=workers,
                         pin_memory=True,
                         shuffle=True)


def train(model, criterion, optimizer, epoch, writer):
    model.train()
    total = 0   # Reset every plot_every
    acc_meter = AverageMeter()
    output_count_meter = BinCounterMeter(labels_arr)
    train_enum = tqdm(train_loader, desc='Train epoch %d' % epoch)

    for pose_dsc, label, _ in train_enum:
        pose_dsc, label = pose_dsc.to(device), label.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward
        output = model(pose_dsc)
        loss = criterion(output, label)

        # Backward
        loss.backward()

        grad_check_negative = check_grad(model.parameters(), clip_grad, ignore_grad)
        if grad_check_negative:
            print('Not a finite gradient or too big, ignoring.')
            optimizer.zero_grad()
            continue

        optimizer.step()
        total += loss.item()
        # calculate accuracy
        acc, pix = accuracy(output, label)
        acc_meter.update(acc, pix)

        _, output_as_ind = torch.max(output, dim=1)
        unique, counts = np.unique(output_as_ind.cpu().numpy(), return_counts=True)
        output_count_meter.update(unique, counts)

    for name, param in model.named_parameters():
        writer.add_histogram(name, param.data.cpu().numpy(), epoch)

        if param.grad is not None:
            writer.add_histogram(name + '/gradient', param.grad.cpu().numpy(), epoch)

    loss_avg = total / len(train_loader)
    acc_avg = acc_meter.average() * 100
    writer.add_scalar('Loss_Avg/Train', loss_avg, epoch)
    writer.add_scalar('Precision_Avg/Train', acc_avg, epoch)
    avg_dist_arr = output_count_meter.get_distribution()
    print('====> Total train set loss: {:.4f}, acc: {:.4f}, dist: ({:.4f}, {:.4f}, {:.4f})'.format(loss_avg, acc_avg, *avg_dist_arr))
    return loss_avg, acc_avg


def evaluate(model, criterion, epoch, writer, loader):
    model.eval()
    total = 0
    acc_meter = AverageMeter()
    output_count_meter = BinCounterMeter(labels_arr)
    valid_enum = tqdm(loader, desc='Valid epoch %d' % epoch)

    with torch.no_grad():
        for pose_dsc, label, _ in valid_enum:
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

    loss_avg = total / len(valid_loader)
    acc_avg = acc_meter.average() * 100

    if writer is not None:
        writer.add_scalar('Loss_Avg/Val', loss_avg, epoch)
        writer.add_scalar('Precision_Avg/Val', acc_avg, epoch)
    avg_dist_arr = output_count_meter.get_distribution()
    print('====> Total valid set loss: {:.4f}, acc: {:.4f}, dist: ({:.4f}, {:.4f}, {:.4f})\n'.format(loss_avg, acc_avg, *avg_dist_arr))
    return loss_avg, acc_avg


def main():
    startTime = datetime.now()
    start_epoch = 1

    model = FencingModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if checkpoint != '':
        checkpoint_args_path = os.path.dirname(checkpoint) + '/args.pth'
        checkpoint_args = torch.load(checkpoint_args_path)

        start_epoch = checkpoint_args[1]

        lr = checkpoint_args[2]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.load_state_dict(torch.load(checkpoint))

    weights = None
    criterion = nn.NLLLoss(weight=weights).to(device)

    best_eval = float('inf')
    best_val_err_full_info = {}

    writer = SummaryWriter(expName)
    epochs_since_improvement=0

    # Begin!
    for epoch in range(start_epoch, start_epoch + epochs):

        if adjust_lr_manually:
            # Halve learning rate if there is no improvement for 3 consecutive epochs, and terminate training after 8
            if epochs_since_improvement == max_not_improving_epochs:
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
                adjust_learning_rate(optimizer, 0.6)

        train_avg_loss, train_avg_acc = train(model, criterion, optimizer, epoch, writer)
        val_avg_loss, val_avg_acc = evaluate(model, criterion, epoch, writer, valid_loader)

        if val_avg_loss < best_eval:
            torch.save(model.state_dict(), '%s/bestmodel.pth' % (expName))
            best_eval = val_avg_loss
            epochs_since_improvement = 0
            best_val_err_full_info = {'epoch': epoch, 'train_avg_loss': train_avg_loss, 'train_avg_acc': train_avg_acc,
                                           'val_avg_loss': val_avg_loss, 'val_avg_acc': val_avg_acc}
        else:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        torch.save(model.state_dict(), '%s/lastmodel.pth' % (expName))
        torch.save([epoch, optimizer.param_groups[0]['lr']], '%s/args.pth' % (expName))

    writer.close()
    print(json.dumps(best_val_err_full_info, indent=4, sort_keys=True))

    print('Running Test\n')
    model.load_state_dict(torch.load('%s/bestmodel.pth' % (expName)))
    test_avg_loss, test_avg_acc = evaluate(model, criterion, epoch, None, test_loader)
    print('====> Total test set loss: {:.4f}, acc: {:.4f}'.format(test_avg_loss, test_avg_acc))

    print('startTime=' + str(startTime))
    print('endTime=' + str(datetime.now()))

if __name__ == '__main__':
    main()
