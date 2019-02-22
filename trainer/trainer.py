from __future__ import print_function

import copy
import time

import torch
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

        _, pred_len = torch.max(output_len.data, 1)

        # Don't attempt to predict sequences that are longer than the
        # sequence length predicted.
        pred_len[pred_len > len(output_seq)] = 0

        # Accumulates the loss across the sequence, ignoring predictions
        # for any number that was -1 in target (i.e., should not have been
        # predicted). We do this by zeroing the loss at those locations.
        dig_loss = 0.0

        for i in range(len(output_seq)):

            # TODO: REMOVE PREDICTIONS PAST THE LENGTH REQUESTED BY pred_len

            # Used to keep track of -1s.
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


def train_model(model, train_loader, valid_loader, device,
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
    train_loss_history, valid_loss_history = [], []
    valid_accuracy_history = []
    valid_best_accuracy = 0
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOM)

    multi_loss = Loss()

    print("# Start training #")
    for epoch in range(num_epochs):

        train_loss = 0
        train_n_iter = 0

        # Set model to train mode
        model = model.train()

        # Iterate over train data
        print("\n\n\nIterating over training data...")
        for i, batch in enumerate(tqdm(train_loader)):

            # Get the inputs
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)
            target_ndigits = targets[:, 0].long().to(device)
            target_sequence = targets[:, 1:].long().to(device)

            # Zero the gradient buffer
            optimizer.zero_grad()

            # Forward
            output_len, output_seq = model(inputs)

            loss = multi_loss.calc(
                output_len, target_ndigits, output_seq, target_sequence)

            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            train_n_iter += 1

        valid_loss = 0
        valid_n_iter = 0
        valid_len_correct, valid_seq_correct = 0, 0
        valid_n_samples = 0

        # Set model to evaluate mode
        model = model.eval()

        # Iterate over valid data
        print("Iterating over validation data...")
        for i, batch in enumerate(tqdm(valid_loader)):
            # get the inputs
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)
            target_ndigits = targets[:, 0].long().to(device)
            target_sequence = targets[:, 1:].long().to(device)

            # Forward
            output_len, output_seq = model(inputs)

            loss = multi_loss.calc(
                output_len, target_ndigits, output_seq, target_sequence)

            # Statistics
            valid_loss += loss.item()
            valid_n_iter += 1

            # TODO: astype string length should be taken from cfg!
            this_batch = output_seq[0].size()[0]
            str_target = np.repeat('', this_batch).astype('<U10')
            str_pred = np.repeat('', this_batch).astype('<U10')

            for i in range(len(output_seq)):

                # Used to keep track of -1s.
                target_mask = target_sequence[:, i].clone()
                target_mask[target_mask >= 0] = 1
                target_mask[target_mask <= 0] = 0

                # Get int representation of predictions and targets.
                _, int_pred = torch.max(output_seq[i].data, 1)
                int_target = target_sequence[:, i].clone()
                int_pred = int_pred.cpu().numpy().astype(np.str)
                int_target = int_target.cpu().numpy().astype(np.str)
                target_mask = target_mask.cpu().numpy().astype(np.uint8)

                # Remove non-digit numbers.
                int_pred[target_mask == 0] = ''
                int_target[target_mask == 0] = ''
                for j, (target, pred) in enumerate(zip(int_target, int_pred)):
                    str_target[j] = str_target[j] + target
                    str_pred[j] += pred

            valid_seq_correct += np.sum(str_pred == str_target)

            _, len_pred = torch.max(output_len.data, 1)
            valid_len_correct += (len_pred == target_ndigits).sum().item()
            valid_n_samples += target_ndigits.size(0)

        train_loss_history.append(train_loss / train_n_iter)
        valid_loss_history.append(valid_loss / valid_n_iter)
        valid_len_acc = valid_len_correct / valid_n_samples
        valid_seq_acc = valid_seq_correct / valid_n_samples
        valid_tot_acc = gmean([valid_len_acc, valid_seq_acc])

        print('\nEpoch: {}/{}'.format(epoch + 1, num_epochs))
        print('\tTrain Loss: {:.4f}'.format(train_loss / train_n_iter))
        print('\tValid Loss: {:.4f}'.format(valid_loss / valid_n_iter))
        print('\tValid Acc (Len/Seq/Tot) = [{:.4f} {:.4f} {:.4f}]'.format(
            valid_len_acc, valid_seq_acc, valid_tot_acc))

        if valid_tot_acc > valid_best_accuracy:
            valid_best_accuracy = valid_tot_acc
            best_model = copy.deepcopy(model)
            print('Checkpointing new model...')
            model_filename = output_dir + '/checkpoint.pth'
            torch.save(model, model_filename)
        valid_accuracy_history.append(valid_tot_acc)

    time_elapsed = time.time() - since

    print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Saving model ...')
    model_filename = output_dir + '/best_model.pth'
    torch.save(best_model, model_filename)
    print('Best model saved to :', model_filename)
