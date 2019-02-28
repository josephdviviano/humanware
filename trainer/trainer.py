from __future__ import print_function

import copy
import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import numpy as np
from scipy.stats.mstats import gmean
from utils.config import cfg


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

        return(loss)

def count_correct_sequences(output_seq, target_seq, valid_len_mask):
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

    return(n_correct)


def train_model(model, optimizer, train_loader, valid_loader, device,
                num_epochs=cfg.TRAIN.NUM_EPOCHS, lr=cfg.TRAIN.LR,
                output_dir=None):
    """
    Training loop.

    Parameters
    ----------
    model : obj
        The model.
    train_loader : obj
        The train data loader.
    valid_loader : obj
        The validation data loader.
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

    history = {'train': {'acc': [], 'loss': []},
               'valid': {'acc': [], 'loss': []}}

    valid_best_accuracy = 0
    best_epoch = 0
    valid_loss = 10000 # Initial value.

    multi_loss = Loss()

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
        train_len_acc = train_len_correct / train_n_samples
        train_seq_acc = train_seq_correct / train_n_samples
        valid_len_acc = valid_len_correct / valid_n_samples
        valid_seq_acc = valid_seq_correct / valid_n_samples

        history['train']['loss'].append(train_loss)
        history['valid']['loss'].append(valid_loss)
        history['train']['acc'].append(train_seq_acc)
        history['valid']['acc'].append(valid_seq_acc)

        # For reporting purposes.
        loss_msg = 'Loss (t/v)=[{:.4f} {:.4f}]'.format(train_loss, valid_loss)
        len_acc_msg = 'Len Acc (t/v)=[{:.4f} {:.4f}]'.format(
            train_len_acc, valid_len_acc)
        seq_acc_msg = 'Seq Acc (t/v)=[{:.4f} {:.4f}]'.format(
            train_seq_acc, valid_seq_acc)
        print('\t[{}/{}] {} {} {}'.format(
            epoch+1, num_epochs, loss_msg, len_acc_msg, seq_acc_msg))

        # Early stopping on best sequence accuracy.
        if valid_seq_acc > valid_best_accuracy:
            valid_best_accuracy = valid_seq_acc
            best_epoch = epoch+1
            best_model = copy.deepcopy(model)
            print('Checkpointing new model...\n')
            model_filename = output_dir + '/checkpoint.pth'
            torch.save(model, model_filename)

        # Patience stop training after n epochs of no improvement.
        elif epoch+1 - cfg.TRAIN.EARLY_STOPPING_PATIENCE == best_epoch:
            print('No improvement in valid loss in {} epochs, stopping'.format(
                cfg.TRAIN.EARLY_STOPPING_PATIENCE))
            break

    time_elapsed = time.time() - since

    print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Saving model ...')
    model_filename = output_dir + '/best_model.pth'
    torch.save(best_model, model_filename)
    print('Best model saved to :', model_filename)

    results = {
        'best_model': best_model,
        'best_epoch': best_epoch,
        'best_acc': valid_best_acc,
        'time_elapsed': time_elapsed,
        'last_epoch': epoch+1,
        'history': history,
        'optimizer': optimizer
    }

    return(results)

