from __future__ import print_function

import copy
import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torchvision.utils

import numpy as np
from scipy.stats.mstats import gmean
from utils.config import cfg

from tensorboardX import SummaryWriter


class Loss():
    """
    Calculates the multitask loss using the defined loss function.
    In order to zero losses for predictions that should not have been made,
    the seq_loss requires reduction to be turned off.
    """

    def __init__(self):
        self.len_loss = torch.nn.CrossEntropyLoss()
        self.seq_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def calc(self, output_len, target_len, output_seq, target_seq):
        """
        Returns the multitask loss using the defined loss function.
        Loss is the sum of the log probabilities across the predicted
        sequence length and all of the expected digits (i.e., this
        ignores the left-over digits).
        """
        len_loss = self.len_loss(output_len, target_len)

        # Accumulates the loss across the sequence, ignoring predictions
        # for any number that was -1 in target (i.e., should not have been
        # predicted). We do this by zeroing the loss at those locations.
        dig_loss = 0.0

        for i in range(len(output_seq)):
            # Keep track of -1s: numbers that don't exist in ground truth.
            target_mask = target_seq[:, i].clone()
            target_mask[target_mask >= 0] = 1
            target_mask[target_mask <= 0] = 0

            # Set -1s to 0 so that we don't break the call to softmax.
            clean_targets = target_seq[:, i].clone()
            clean_targets[clean_targets == -1] = 0

            # Zero any loss that is not in the target_mask, then average.
            losses = self.seq_loss(output_seq[i], clean_targets)
            losses[target_mask == 0] = 0
            dig_loss += torch.mean(losses.float())

        # Sum of all log probabilities.
        loss = len_loss + dig_loss

        return (loss)


def count_correct_sequences(output_seq, target_seq, valid_len_mask, output_pred=False):
    """
    Sequence predictions. All elements in valid_len_mask that are
    0 are not counted.
    """
    # TODO: astype string length should be taken from cfg!
    # Store the predicted and target integers as an array of strings.
    this_batch = output_seq[0].size()[0]
    str_target = np.repeat('', this_batch).astype('<U10')
    str_pred = np.repeat('', this_batch).astype('<U10')

    for i in range(len(output_seq)):

        # Keep track of -1s: numbers that don't exist in ground truth.
        target_mask = target_seq[:, i].clone()
        target_mask[target_mask >= 0] = 1
        target_mask[target_mask <= 0] = 0

        # Get int representation of predictions and targets.
        _, seq_preds = torch.max(output_seq[i].data, 1)
        these_targets = target_seq[:, i].clone()

        # Convert house numbers to arrays of strings (also, numpy).
        seq_preds = seq_preds.cpu().numpy().astype(np.str)
        these_targets = these_targets.cpu().numpy().astype(np.str)

        # Convert mask to numpy.
        target_mask = target_mask.cpu().numpy().astype(np.uint8)

        # Remove non-ground-truth numbers.
        seq_preds[target_mask == 0] = ''
        these_targets[target_mask == 0] = ''

        # For each subject in the batch, add the string representation
        # of the target to the corresponding element of the string array.
        for j, (target, pred) in enumerate(zip(these_targets, seq_preds)):
            str_target[j] += target
            str_pred[j] += pred

    # Zero out predictions made where the predicted length was incorrect.
    str_pred[valid_len_mask == 0] = ''
    n_correct = np.sum(str_pred == str_target)

    if output_pred:
        return str_pred == str_target
    else:
        return (n_correct)


def misclassified_images(model, valid_loader, device, writer, number_max=5):
    print(f'Adding {number_max} misclassified images to TensorboardX...')
    firsts_misclassified_img = ()
    nb_misclassified_img = 0
    while nb_misclassified_img < number_max:
        for batch_idx, batch in enumerate(valid_loader):
            # Get the inputs.
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)
            target_len = targets[:, 0].long().to(device)
            target_seq = targets[:, 1:].long().to(device)

            # Forward.
            output_len, output_seq = model(inputs)

            # Length predictions.
            _, len_pred = torch.max(output_len.data, 1)
            valid_len_mask = (len_pred == target_len).cpu().numpy()

            # Sequence predictions.
            valid_seq_correct = count_correct_sequences(output_seq, target_seq, valid_len_mask, True)

            # Misclassified images
            if np.sum(valid_seq_correct == False) > 0:
                print('Dimension of inputs:',inputs.size())
                mask = torch.tensor(np.array(valid_seq_correct == False).astype(np.int))
                misclassified_img = inputs[mask == 1]
                print(f'Number of misclassified images in batch {batch_idx} : {np.sum(valid_seq_correct == False)}')
                print('Dimension of misclassified_img:', misclassified_img.size())
                firsts_misclassified_img += (misclassified_img,)
                print('\n')

            nb_misclassified_img += np.sum(valid_seq_correct == False)

    img = torch.cat(firsts_misclassified_img, dim=0)
    writer.add_image('Misclassified_images', torchvision.utils.make_grid(img))


def train_model(model, train_loader, valid_loader, device,
                num_epochs=cfg.TRAIN.NUM_EPOCHS, lr=cfg.TRAIN.LR,
                output_dir=None, track_misclassified=False):
    """
    Training loop.

    Parameters
    ----------
    model : obj
        The model.
    train_loader : obj
        The train data loader.
    valid_loader : obj
    ``    The validation data loader.
    device : str
        The type of device to use ('cpu' or 'gpu').
    num_eopchs : int
        Number of epochs to train the model.
    lr : float
        Learning rate for the optimizer.
    output_dir : str
        path to the directory where to save the model.

    """
    since = time.time()
    model = model.to(device)
    train_loss_history, valid_loss_history = [], []
    valid_accuracy_history = []
    valid_best_accuracy = 0
    valid_loss = 10000  # Initial value.

    multi_loss = Loss()

    tb = SummaryWriter('runs')
    # tb.add_scalars('Initialization', {'Learning rate': cfg.TRAIN.LR,
    #                                  'Weight decay': cfg.TRAIN.L2,
    #                                  'Max epochs': cfg.TRAIN.NUM_EPOCHS,
    #                                  'Patience': cfg.TRAIN.SCHEDULER_PATIENCE,
    #                                  'Begin': since})
    # tb.add_text('Device',device)

    # optimizer = torch.optim.SGD(
    #    model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOM,
    #    weight_decay=cfg.TRAIN.L2)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.L2)

    scheduler = ReduceLROnPlateau(optimizer,
                                  patience=cfg.TRAIN.SCHEDULER_PATIENCE)

    print("# Start training #")
    for epoch in range(num_epochs):

        # Reduces LR by factor of 10 if we don't beat best valid_loss in
        # cfg.TRAIN.SCHEDULER_PATIENCE epochs.
        scheduler.step(valid_loss)

        # Initialize values for this epoch.
        train_loss, train_n_iter = 0, 0
        valid_loss, valid_n_iter = 0, 0
        train_len_correct, train_seq_correct = 0, 0
        valid_len_correct, valid_seq_correct = 0, 0
        train_n_samples = 0
        valid_n_samples = 0

        # Train data.
        model = model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            # Get the inputs
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)
            target_len = targets[:, 0].long().to(device)
            target_seq = targets[:, 1:].long().to(device)

            # Zero the gradient buffer
            optimizer.zero_grad()

            # Forward
            output_len, output_seq = model(inputs)

            loss = multi_loss.calc(
                output_len, target_len, output_seq, target_seq)

            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            train_n_iter += 1

            # Length predictions.
            _, len_pred = torch.max(output_len.data, 1)
            train_len_mask = (len_pred == target_len).cpu().numpy()
            train_len_correct += np.sum(train_len_mask)

            # Sequence predictions.
            train_seq_correct += count_correct_sequences(
                output_seq, target_seq, train_len_mask)

            train_n_samples += target_len.size(0)

        # Validation data.
        model = model.eval()

        for batch_idx, batch in enumerate(tqdm(valid_loader)):
            # Get the inputs.
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)
            target_len = targets[:, 0].long().to(device)
            target_seq = targets[:, 1:].long().to(device)

            # Forward.
            output_len, output_seq = model(inputs)

            loss = multi_loss.calc(
                output_len, target_len, output_seq, target_seq)

            # Statistics
            valid_loss += loss.item()
            valid_n_iter += 1

            # Length predictions.
            _, len_pred = torch.max(output_len.data, 1)
            valid_len_mask = (len_pred == target_len).cpu().numpy()
            valid_len_correct += np.sum(valid_len_mask)

            # Sequence predictions.
            valid_seq_correct += count_correct_sequences(
                output_seq, target_seq, valid_len_mask)

            valid_n_samples += target_len.size(0)

        # Calculate final values
        train_loss /= train_n_iter
        valid_loss /= valid_n_iter

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        train_len_acc = train_len_correct / train_n_samples
        train_seq_acc = train_seq_correct / train_n_samples
        valid_len_acc = valid_len_correct / valid_n_samples
        valid_seq_acc = valid_seq_correct / valid_n_samples

        # For reporting purposes.
        loss_msg = 'Loss (t/v)=[{:.4f} {:.4f}]'.format(train_loss, valid_loss)
        train_acc_msg = 'Train Acc (len/seq)=[{:.4f} {:.4f}]'.format(
            train_len_acc, train_seq_acc)
        valid_acc_msg = 'Valid Acc (len/seq)=[{:.4f} {:.4f}]'.format(
            valid_len_acc, valid_seq_acc)
        print('\t[{}/{}] {} {} {}'.format(
            epoch + 1, num_epochs, loss_msg, train_acc_msg, valid_acc_msg))
        tb.add_scalar('LR', optimizer.param_groups[-1]['lr'], global_step=epoch + 1)
        tb.add_scalars('Length', {'Train len accuracy': train_len_acc,
                                  'Valid len accuracy': valid_len_acc},
                       global_step=epoch + 1)
        tb.add_scalars('Sequence', {'Train seq accuracy': train_seq_acc,
                                    'Valid seq accuracy': valid_seq_acc},
                       global_step=epoch + 1)

        tb.add_image('Sample', torchvision.utils.make_grid(inputs))

        # Early stopping on best sequence accuracy.
        if valid_seq_acc > valid_best_accuracy:
            valid_best_accuracy = valid_seq_acc
            best_model = copy.deepcopy(model)
            print('Checkpointing new model...\n')
            model_filename = output_dir + '/checkpoint.pth'
            torch.save(model, model_filename)
        valid_accuracy_history.append(valid_seq_acc)

    time_elapsed = time.time() - since

    # Error analysis on best model - Viewing the misclassified images on Tensorboard
    if track_misclassified:
        misclassified_images(model, valid_loader, device, tb)

    print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Saving model ...')
    model_filename = output_dir + '/best_model.pth'
    torch.save(best_model, model_filename)
    print('Best model saved to :', model_filename)
