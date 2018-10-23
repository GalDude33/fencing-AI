import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
import os
import json
import numpy as np

from network.model import FencingModel
from network.dataloader import Dataset


batch_size = 32
workers = 2
use_cuda = True
learning_rate = 1e-4
checkpoint = ''
expName = 'fencing_exp'
epochs = 50
adjust_lr_manually = 1
max_not_improving_epochs = 10
clip_grad = 0.5
ignore_grad = 10000.0

train_dataset = Dataset(is_train=1)
train_loader = torch.utils.data.DataLoader(train_dataset,
                         batch_size=batch_size,
                         num_workers=workers,
                         pin_memory=True,
                         shuffle=True)

valid_dataset = Dataset(is_train=0)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                         batch_size=batch_size,
                         num_workers=workers,
                         pin_memory=True)


def train(model, criterion, optimizer, epoch, train_losses, writer, iterNum):
    model.train()
    total = 0   # Reset every plot_every
    train_enum = tqdm(train_loader, desc='Train epoch %d' % epoch)

    for pose_dsc, label in train_enum:
        pose_dsc_var = Variable(pose_dsc).cuda()
        label_var = Variable(label).cuda()

        # Zero gradients
        optimizer.zero_grad()

        # Forward
        output = model(pose_dsc_var)
        loss = criterion(output, label_var)

        # Backward
        loss.backward()

        grad_check_negative = check_grad(model.parameters(), clip_grad, ignore_grad)
        if grad_check_negative:
            print('Not a finite gradient or too big, ignoring.')
            optimizer.zero_grad()
            continue

        optimizer.step()
        total += loss.data[0]
        iterNum += 1

    for name, param in model.named_parameters():
        writer.add_histogram(name, param.data.cpu().numpy(), epoch)

        if param.grad is not None:
            writer.add_histogram(name + '/gradient', param.grad.data.cpu().numpy(), epoch)

    loss_avg = total / len(train_loader)
    writer.add_scalar('Loss_Avg/Train', loss_avg, epoch)
    train_losses.append(loss_avg)
    print('====> Total train set loss: {:.4f}'.format(loss_avg))
    return iterNum, loss_avg


def evaluate(model, criterion, epoch, eval_losses, writer, iterNum):
    model.eval()
    total = 0
    valid_enum = tqdm(valid_loader, desc='Valid epoch %d' % epoch)

    for pose_dsc, label in valid_enum:
        pose_dsc_var = Variable(pose_dsc).cuda()
        label_var = Variable(label).cuda()

        # Forward
        output = model(pose_dsc_var)
        loss = criterion(output, label_var)

        total += loss.data[0]
        iterNum += 1

    loss_avg = total / len(valid_loader)

    writer.add_scalar('Loss_Avg/Val', loss_avg, epoch)
    eval_losses.append(loss_avg)
    print('====> Total validation set loss: {:.4f}'.format(loss_avg))
    return iterNum, loss_avg


def check_grad(params, clip_th, ignore_th):
    befgad = torch.nn.utils.clip_grad_norm(params, clip_th)
    return (not np.isfinite(befgad) or (befgad > ignore_th))


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %.8f\n" % (optimizer.param_groups[0]['lr'],))


def main():
    startTime = datetime.now()
    start_epoch = 1
    trainIter = 1
    valIter = 1

    model = FencingModel()
    if use_cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if checkpoint != '':
        checkpoint_args_path = os.path.dirname(checkpoint) + '/args.pth'
        checkpoint_args = torch.load(checkpoint_args_path)

        start_epoch = checkpoint_args[3]
        trainIter = checkpoint_args[4]
        valIter = checkpoint_args[5]

        lr = checkpoint_args[6]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.load_state_dict(torch.load(checkpoint))

    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()

    # Keep track of losses
    train_losses = []
    eval_losses = []
    best_eval = float('inf')

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

        trainIter, train_avg_loss = train(model, criterion, optimizer, epoch, train_losses, writer, trainIter)
        valIter, val_avg_loss = evaluate(model, criterion, epoch, eval_losses, writer, valIter)

        if val_avg_loss < best_eval:
            torch.save(model.state_dict(), '%s/bestmodel.pth' % (expName))
            best_eval = val_avg_loss
            epochs_since_improvement = 0
            best_val_err_full_info = {'epoch': epoch, 'train_avg_loss': train_avg_loss, 'val_avg_loss': val_avg_loss}
        else:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        torch.save(model.state_dict(), '%s/lastmodel.pth' % (expName))
        torch.save([train_losses, eval_losses, epoch, trainIter, valIter, optimizer.param_groups[0]['lr']], '%s/args.pth' % (expName))

    writer.close()
    print(json.dumps(best_val_err_full_info, indent=4, sort_keys=True))
    print('startTime=' + str(startTime))
    print('endTime=' + str(datetime.now()))

if __name__ == '__main__':
    main()
