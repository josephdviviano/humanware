# Door Number Detection Project

**Quickstart:** Look at `run_on_helios.sh` for example usage.

This repository contains the code necessary for the door number detection
project.

The goal of the project is to help blind persons to find their way around by
making sure they are at the right house when they want for example visit a
friend or a family member, go to a specific store, etc.

In developing this project we must keep in mind the different constraints of
this application notably for the selection and development of the models we
will use like the execution time, online vs. offline, the memory usage (in the
case of a mobile application), etc.

## extra SVHN dataset

This model combines the `train` and `extra` data  available in
`train_extra_metadata_split.pkl`. This increases the number of available images
by a factor of 10 and therefore significantly increases training time.

## Hyperparameter search.

Hyperparameter search uses a baysian approach via the `skopt` package.

The `NUM_HYPER_LOOP` option in the `config.yml` controls how many iterations
of this occour during training.

## Checkpointing.

Checkpointing is stored in the results directory with each of the individual
training loops conducted during hyperparameter searching having it's own sub-
folder `{00, 01, 02, ... NN}`. The file `checkpoint.pth.tar` contains all
information required to:

+ Re-instantiate the seeds properly for the random number generators.
+ Recreate the model with it's weights.
+ Recreate the optimizer and it's scheduler with their state.
+ Recreate the hyperparameter optimizer with it's state.
+ Restart tensorboard logging.
+ Pick up where training left off with respect to loss tracking, etc.

`config.yml` contains the configuration used for this experiment. It will be
used if it exists regardless of whether the user specified another
configuration file at runtime.

Inside each numbered folder for each hyperparameter run, the `best_model.pth`
file contains the best model found during training (i.e, early stopping).
This also means the models can be combined later for ensemble learning if
desired.

Finally, each numbered folder contains a `tensorboard/` folder which contains
the tensorboard logs for that particular run.

Here is a view of the results directory structure:

```
results_dir/

    checkpoint.pth.tar -- the last state of training saved (epoch wise).
    config.yml         -- the configuration used to launch training.
    best_model.pth.tar -- the best model among all hyperparamater runs (so far).

    00/                -- 1st iteration of hyper parameter search.
        best_model.pth -- best model found for these hyper parameters.
        tensorboard/   -- tensorboard logs for these hyper parameters.

    01/                -- 2st iteration of hyper parameter search.
        best_model.pth -- best model found for these hyper parameters.
        tensorboard/   -- tensorboard logs for these hyper parameters.
    ...
    NN/                -- nth iteration of hyper parameter search.
        best_model.pth -- best model found for these hyper parameters.
        tensorboard/   -- tensorboard logs for these hyper parameters.
```

## Quick usage on Helios

To run the code on Helios, you can use the scripts in `scrips/helios/train_on_helios.sh`.

You can run this directly from the login node using msub:

`msub -A $GROUP_RAP -l feature=k80,nodes=1:gpus=1,walltime=2:00:00 train_on_helios.sh`

You can easily add this script to a `.pbs` file with your specific settings.

## Data
For more information about the data used and its format, consult the `README`
in the `data/` directory.

## Includes
`bbopt` cloned on March 1st 2019.

