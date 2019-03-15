import copy
import os
import time
import random

import torch
from tqdm import tqdm
import torchvision.utils
from tensorboardX import SummaryWriter

import numpy as np
from utils.config import cfg


class Loss():
    """
    Calculates the multitask loss using the defined loss function.
    In order to zero losses for predictions that should not have been made,
    the seq_loss requires reduction to be turned off.

    Returns
    ----------
    loss : tensor
        The sum of length loss and digit loss.
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


def count_correct_sequences(output_seq, target_seq, valid_len_mask,
                            output_pred=False):
    """
    Sequence predictions. All elements in valid_len_mask that are
    0 are not counted.

    Parameters
    ----------
    output_seq: tensor
        The estimated sequence length from the model output
    target_seq: long tensor
        The target sequence we try to estimate
    valid_len_mask: numpy array
        A mask to zero-out the sequences with incorrect length prediction.
    output_pred: Bool
        If True returns a vector of correctly predicted sequence
        If False, return the sum of correctly predicted sequence

    Returns
    ----------
    n_correct : int
        The number of correct sequences.
    str_pred == str_target : numpy array
        The vector of correctly predicted sequence
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
    """
    Does a foward pass for the best model to add missclassified images to TensorboardX writer for easy visualization.

    Parameters
    ----------
    model: obj
        The model.
    valid_loader: obj
        The dataloader to load the validation dataset.
    device: string
        The torch device used either cuda of cpu
    writer: Bool
        The TensorboardX writer.
    number_max: int
        The maximum number of images to add to TensorboardX
    """

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
            len_mask = (len_pred == target_len).cpu().numpy()

            # Sequence predictions.
            seq_correct = count_correct_sequences(
                output_seq, target_seq, len_mask, output_pred=True)

            # Misclassified images
            seq_correct = seq_correct.astype(np.int)  # True = 1, False = 0
            n_misclassified = len(seq_correct) - np.sum(seq_correct)

            if n_misclassified > 0:
                seq_correct = torch.tensor(seq_correct)
                misclassified_img = inputs[seq_correct == 0]
                firsts_misclassified_img += (misclassified_img,)
                nb_misclassified_img += n_misclassified

    print('Adding {} misclassified images to TensorboardX...'.format(
        number_max))

    img = torch.cat(firsts_misclassified_img, dim=0)
    writer.add_image('Misclassified_images', torchvision.utils.make_grid(img))


def run_epoch(model, loader, optimizer, lossfxn, device, train=True):
    """
    Runs an epoch of data through model, either in training or
    evaluation mode.

    Parameters
    ----------
    model: obj
        The model.
    loader: obj
        The data loader for the training dataset.
    optimizer: torch.optim
        The optimizer used with the model.
    lossfxn: function
        The loss function to use (ex: multi-loss)
    device: string
        The device to be used (ex: cuda:0, cpu)
    train: Bool
        If True trains the model, else validation the model

    Returns
    ----------
    results : dictionary
        The loss, the length accuracy and the sequence accuaracy.
    """

    mean_loss, n_iter, n_samples, len_correct, seq_correct = 0, 0, 0, 0, 0

    if train:
        model = model.train()
    else:
        model = model.eval()

    for batch_idx, batch in enumerate(tqdm(loader)):

        # Get the inputs
        inputs, targets = batch['image'], batch['target']

        inputs = inputs.to(device)
        target_len = targets[:, 0].long().to(device)
        target_seq = targets[:, 1:].long().to(device)

        if train:
            optimizer.zero_grad()  # Zero the gradient buffer.

        # Forward.
        output_len, output_seq = model(inputs)
        loss = lossfxn.calc(output_len, target_len, output_seq, target_seq)

        # Backward.
        if train:
            loss.backward()
            optimizer.step()

        # Statistics
        mean_loss += loss.item()
        n_iter += 1

        # Length predictions.
        _, len_pred = torch.max(output_len.data, 1)
        len_mask = (len_pred == target_len).cpu().numpy()
        len_correct += np.sum(len_mask)

        # Sequence predictions.
        seq_correct += count_correct_sequences(output_seq,
                                               target_seq, len_mask)

        n_samples += target_len.size(0)

    # Final stats.
    mean_loss /= n_iter
    len_acc = len_correct / n_samples
    seq_acc = seq_correct / n_samples

    results = {'loss': mean_loss, 'len_acc': len_acc, 'seq_acc': seq_acc}

    return (results)


def train_model(model, optimizer, scheduler, hp_opt,
                train_loader, valid_loader,
                device, output_dir, iteration, resume, best_final_acc,
                num_epochs=cfg.TRAIN.NUM_EPOCHS, lr=cfg.TRAIN.LR,
                l2=cfg.TRAIN.L2, momentum=cfg.TRAIN.MOM,
                track_misclassified=False):
    """
    Training loop.

    Parameters
    ----------
    model : obj
        The model.
    optimizer : torch.optim
        The optimizer to use with the model
    scheduler: torch.optim.lr_scheduler
        Reduce learning rate when a metric has stopped improving.
    hp_opt: skopt.optimizer
        An Optimizer represents the steps of a bayesian optimisation loop used for hyper-parameter search.
    train_loader : obj
        The train data loader.
    valid_loader : obj
        The validation data loader.
    device : str
        The type of device to use ('cpu' or 'gpu').
    output_dir : str
        path to the directory where to save the model.
    iteration: int
        The iteration state for the hyper-parameter loop
    resume: Bool
        If True, resume the model where it was at checkpointing.
    best_final_acc: float
        The best accuracy from the model. Used to save the best model.
    num_epochs : int
        Number of epochs to train the model.
    lr : float or list
        Learning rate for the optimizer.
    l2 : float or list
        Weight decay.
    momentum: float or list
        Values for the momentum SGD
    track_misclassified: Bool
        If True, adds misclassified images of the best model to tensorboardX

    --------
    Results

    results : dictonary
        best_model, best_epoch, best_acc, lr ,l2, momentum, time_elapsed, last_epoch, history, optimizer.
    """
    since = time.time()
    model = model.to(device)
    multi_loss = Loss()

    # Load statistics for these n epochs.
    if resume:
        checkpoint = os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth.tar')
        state = torch.load(checkpoint)
        history = state['history']
        valid_best_accuracy = state['valid_best_accuracy']
        best_epoch = state['best_epoch']
        valid_loss = state['valid_loss']
        base_epoch = state['epoch']
    else:
        history = {'train': {'acc': [], 'loss': []},
                   'valid': {'acc': [], 'loss': []}}
        base_epoch, valid_best_accuracy, best_epoch = 0, 0, 0
        valid_loss = 10000

    # Add a summarywriter with out current hyperparameter settings.
    tb_log_dir = os.path.join(
        os.path.join(
            cfg.OUTPUT_DIR, "{:02d}".format(iteration)), 'tensorboard')

    tb = SummaryWriter(log_dir=tb_log_dir)
    tb.add_scalars('Initialization', {'Learning rate': lr,
                                      'Weight decay': l2,
                                      'Max epochs': num_epochs,
                                      'Patience': cfg.TRAIN.SCHEDULER_PATIENCE,
                                      'Begin': since})

    print("Training model: hp_iteration={} starting at epoch={}".format(
        iteration, base_epoch + 1))

    for epoch in range(base_epoch, num_epochs):

        # Reduces LR by factor of 10 if we don't beat best valid_loss in
        # cfg.TRAIN.SCHEDULER_PATIENCE epochs.
        scheduler.step(valid_loss)

        train_results = run_epoch(model, train_loader,
                                  optimizer, multi_loss, device, train=True)

        valid_results = run_epoch(model, valid_loader,
                                  optimizer, multi_loss, device, train=False)

        history['train']['loss'].append(train_results['loss'])
        history['valid']['loss'].append(valid_results['loss'])
        history['train']['acc'].append(train_results['seq_acc'])
        history['valid']['acc'].append(valid_results['seq_acc'])

        # Checkpoint!
        torch_seed = torch.random.get_rng_state()
        numpy_seed = np.random.get_state()
        python_seed = random.getstate()

        state = {'iteration': iteration,
                 'base_epoch': epoch,
                 'best_epoch': best_epoch,
                 'best_final_acc': best_final_acc,
                 'epoch': epoch + 1,
                 'history': history,
                 'hp_opt': hp_opt,
                 'l2': l2,
                 'lr': lr,
                 'momentum': momentum,
                 'numpy_seed': numpy_seed,
                 'optimizer': optimizer.state_dict(),
                 'python_seed': python_seed,
                 'scheduler': scheduler.state_dict(),
                 'state_dict': model.state_dict(),
                 'torch_seed': torch_seed,
                 'valid_best_accuracy': valid_best_accuracy,
                 'valid_loss': valid_loss,
                 'tb_log_dir': tb_log_dir}

        torch.save(state, os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth.tar'))

        # Report to terminal.
        loss_msg = 'Loss (t/v)=[{:.4f} {:.4f}]'.format(
            train_results['loss'], valid_results['loss'])
        acc_msg = 'Seq Acc (t/v)=[{:.4f} {:.4f}]'.format(
            train_results['seq_acc'], valid_results['seq_acc'])
        print('\t[{}/{}] {} {}'.format(
            epoch + 1, num_epochs, loss_msg, acc_msg))

        # Report to tensorboard.
        tb.add_scalar(
            'LR', optimizer.param_groups[-1]['lr'], global_step=epoch + 1)

        tb.add_scalars(
            'Length',
            {'Train len accuracy': train_results['len_acc'],
             'Valid len accuracy': valid_results['len_acc']},
            global_step=epoch + 1)

        tb.add_scalars(
            'Sequence',
            {'Train seq accuracy': train_results['seq_acc'],
             'Valid seq accuracy': valid_results['seq_acc']},
            global_step=epoch + 1)

        tb.add_scalars(
            'Loss',
            {'Train loss': train_results['loss'],
             'Valid loss': valid_results['loss']},
            global_step=epoch + 1)

        # Early stopping on best sequence accuracy.
        if valid_results['seq_acc'] > valid_best_accuracy:
            valid_best_accuracy = valid_results['seq_acc']
            best_epoch = epoch + 1
            best_model = copy.deepcopy(model)
            model_filename = os.path.join(
                os.path.join(
                    output_dir, "{:02d}".format(iteration), 'best_model.pth'))
            torch.save(model, model_filename)
            print('New best model saved to {}'.format(model_filename))

        # Patience stop training after n epochs of no improvement.
        elif epoch + 1 - cfg.TRAIN.EARLY_STOPPING_PATIENCE == best_epoch:
            print('No improvement in valid loss in {} epochs, stopping'.format(
                cfg.TRAIN.EARLY_STOPPING_PATIENCE))
            break

    time_elapsed = time.time() - since

    # Error analysis on best model (view misclassified images)
    if track_misclassified:
        misclassified_images(model, valid_loader, device, tb)

    print('\n\nTraining iteration {} complete in {:.0f}m {:.0f}s'.format(
        iteration, time_elapsed // 60, time_elapsed % 60))

    results = {
        'best_model': best_model,
        'best_epoch': best_epoch,
        'best_acc': valid_best_accuracy,
        'lr': lr,
        'l2': l2,
        'momentum': momentum,
        'time_elapsed': time_elapsed,
        'last_epoch': epoch + 1,
        'history': history,
        'optimizer': optimizer
    }

    return (results)
