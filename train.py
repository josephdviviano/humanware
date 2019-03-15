#!/usr/bin/env python

from __future__ import print_function
import os
import sys

import argparse
import dateutil.tz
import datetime
import numpy as np
import pprint
import random
from shutil import copyfile

import torch
from skopt import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.config import cfg, cfg_from_file
from utils.dataloader import prepare_dataloaders
from utils.misc import mkdir_p
from models.deepconv import DeepConv  # If using model from Goodfellow paper
from models.vgg import VGG  # If using model from Simonyan paper
from models.resnet import *
from trainer.trainer import train_model

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

NOW = datetime.datetime.now(dateutil.tz.tzlocal())
TIMESTAMP = NOW.strftime('%Y_%m_%d_%H_%M_%S')


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.
    '''
    parser = argparse.ArgumentParser(description='Train a CNN network')
    parser.add_argument('--cfg', type=str,
                        default=None,
                        help='''optional config file,
                             e.g. config/base_config.yml''')

    parser.add_argument("--dataset_dir", type=str,
                        default='data/SVHN/train/',
                        help='''dataset_dir will be the absolute path
                                to the directory to be used for
                                training''')

    parser.add_argument("--metadata_filename", type=str,
                        default='data/SVHN/train_metadata.pkl',
                        help='''metadata_filename will be the absolute
                                path to the directory to be used for
                                training.''')

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    parser.add_argument("--results_dir", type=str,
                        default='results/{}'.format(TIMESTAMP),
                        help="""results_dir will be the absolute
                        path to a directory where the output of your
                        training will be saved. If this folder contains
                        a checkpoint, training will resume from that point
                        on.""")

    args = parser.parse_args()
    return (args)


def load_config():
    '''
    Load the config .yml file.
    '''
    args = parse_args()

    # If a previous configuration is available in the results directory, load.
    previous_config = os.path.join(args.results_dir, 'config.yml')
    if os.path.isfile(previous_config):
        args.cfg = previous_config

    # If we don't specify a config and no previous config is available.
    if args.cfg is None:
        raise Exception("No config file specified or available.")

    # Load configuration into memory.
    cfg_from_file(args.cfg)
    cfg.TIMESTAMP = TIMESTAMP
    cfg.INPUT_DIR = args.dataset_dir
    cfg.METADATA_FILENAME = args.metadata_filename
    cfg.OUTPUT_DIR = args.results_dir

    if args.cfg != previous_config:
        mkdir_p(cfg.OUTPUT_DIR)
        copyfile(args.cfg, os.path.join(cfg.OUTPUT_DIR, 'config.yml'))

    print('Data dir: {}'.format(cfg.INPUT_DIR))
    print('Output dir: {}'.format(cfg.OUTPUT_DIR))
    print('Using config {}:'.format(args.cfg))
    pprint.pprint(cfg)


def fix_seed(seed):
    """
    Fix the seed for numpy, python random, and pytorch.

    Parameters
    ----------
    seed: int
        The seed to use.

    """
    print('pytorch/random seed: {}'.format(seed))

    # Numpy, python, pytorch (cpu), pytorch (gpu).
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False  # TODO: CHANGE BACK
    torch.backends.cudnn.benchmark = False


def load_checkpoint(model, optimizer, losslogger,
                    filename='checkpoint.pth.tar'):
    """
    To be removed.
    """
    # Note: Input model & optimizer should be pre-defined.
    # This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        writer = tb.get_file_writer()
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})".format(
            filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger


if __name__ == '__main__':

    # Load the config file.
    load_config()

    # Make the results reproductible.
    fix_seed(cfg.SEED)

    # Prepare data.
    (train_loader, valid_loader) = prepare_dataloaders(
        dataset_split=cfg.TRAIN.DATASET_SPLIT,
        dataset_path=cfg.INPUT_DIR,
        metadata_filename=cfg.METADATA_FILENAME,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sample_size=cfg.TRAIN.SAMPLE_SIZE,
        valid_split=cfg.TRAIN.VALID_SPLIT)

    # Define model architecture
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    # Check output directory for most recent checkpoint. If it exists, load.
    checkpoint = os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth.tar')
    if os.path.isfile(checkpoint):

        resume = True
        state = torch.load(checkpoint)
        base_iteration = state['iteration']
        hp_opt = state['hp_opt']
        best_final_acc = state['best_final_acc']

        # Check whether we are done. TODO: Cannot handle early stopping!
        done_epochs = state['base_epoch'] + 1 == cfg.TRAIN.NUM_EPOCHS
        done_iterations = base_iteration + 1 == cfg.TRAIN.NUM_HYPER_LOOP
        if done_epochs and done_iterations:
            sys.exit('Resuming a training session that is already complete!')

    else:
        resume = False
        base_iteration = 0
        best_final_acc = 0

        # Baysian hyperparameter optimization: [lr, l2, momentum]
        hp_opt = Optimizer(
            [cfg.TRAIN.LR, cfg.TRAIN.L2, cfg.TRAIN.MOM],
            "GP", acq_optimizer="sampling", random_state=cfg.SEED)

    for iteration in range(base_iteration, cfg.TRAIN.NUM_HYPER_LOOP):

        # Model to be used
        mdl = ResNet18(num_classes=7)
        # mdl = ConvNet(num_classes=7)
        # mdl = BaselineCNN(num_classes=7)
        # mdl = BaselineCNN_dropout(num_classes=7, p=cfg.TRAIN.DROPOUT)
        # mdl = VGG('VGG19', dropout)
        # mdl = DeepConv(dropout=cfg.TRAIN.DROPOUT)

        # For the 1st iteration of a resumed run, restore state.
        if resume:

            # Initialize trainable objects.
            lr = state['lr']
            l2 = state['l2']
            momentum = state['momentum']
            optimizer = torch.optim.SGD(
                mdl.parameters(), lr=lr, weight_decay=l2, momentum=momentum)
            scheduler = ReduceLROnPlateau(
                optimizer, patience=cfg.TRAIN.SCHEDULER_PATIENCE)

            # Restore trainable object states.
            mdl.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])

            # Fix optimizer state not loading back to cuda
            # See: https://github.com/pytorch/pytorch/issues/2830
            for opt_state in optimizer.state.values():
                for k, v in opt_state.items():
                    if isinstance(v, torch.Tensor):
                        opt_state[k] = v.to(device)

            # Restore Seed States
            torch.random.set_rng_state(state['torch_seed'])
            np.random.set_state(state['numpy_seed'])
            random.setstate(state['python_seed'])

        # Otherwise, use the hyperparameter optimizer to get the next settings.
        else:
            lr, l2, momentum = hp_opt.ask()
            optimizer = torch.optim.SGD(
                mdl.parameters(), lr=lr, weight_decay=l2, momentum=momentum)

            scheduler = ReduceLROnPlateau(
                optimizer, patience=cfg.TRAIN.SCHEDULER_PATIENCE)

        results = train_model(
            mdl, optimizer, scheduler, hp_opt,
            train_loader=train_loader, valid_loader=valid_loader,
            device=device, output_dir=cfg.OUTPUT_DIR,
            iteration=iteration, resume=resume, best_final_acc=best_final_acc,
            num_epochs=cfg.TRAIN.NUM_EPOCHS, lr=lr, l2=l2, momentum=momentum,
            track_misclassified=cfg.TRAIN.TRACK_MISCLASSIFIED)

        if resume:
            resume = False

        # Update optimizer with best accuracy obtained.
        hp_opt.tell([lr, l2, momentum], results['best_acc'])

        # Save best results.
        if results['best_acc'] > best_final_acc:
            best_results = results
            best_final_acc = results['best_acc']
            torch.save(best_results,
                       os.path.join(cfg.OUTPUT_DIR, 'best_results.pth.tar'))
