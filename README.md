# Door Number Detection Project

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
the 'extra' XVHN dataset provided had a number of corrupt images that lead
to the following error during dataloading:

```
~/code/humanware/utils/transforms.py in __call__(self, sample)
    103         filename = sample['metadata']['filename']
    104
--> 105         h, w = np.asarray(image).shape[:2]
    106
    107         new_h, new_w = self.output_size

ValueError: not enough values to unpack (expected 2, got 0)
```

The following images were therefore removed from `extra_metadata.pkl`:


## Quick usage on Helios

To run the code on Helios, you can use the scripts in `scrips/helios/train_on_helios.sh`.

You can run this directly from the login node using msub:

`msub -A $GROUP_RAP -l feature=k80,nodes=1:gpus=1,walltime=2:00:00 train_on_helios.sh`

You can easily add this script to a `.pbs` file with your specific settings.

To change the data directories, you can modify the `train_on_helios.sh` script. To change configurations during training, use the `config/base_config.yml` file. This contains tuneable options that can be useful.

To modify the models used, modify the appropriate model declaration in `trainer/trainer.py`. Currently, VGG and Resnet as well as simple baselines are implemented.

## Best Model

The best performing model found for the first part of the project was a VGG19 with learning rate 1e-3, momentum 0.9, SGD optimizer and batch size of 32.

## Run the code interactively
For debugging purpose you might want to run your code interactively.

If you want to stop your code in a particular line you can add those
lines there: `import ipdb; ipdb.set_trace()`.
See [ipdb](https://pypi.org/project/ipdb/) for more informations.

## Data
For more information about the data used and its format, consult the `README`
in the `data/` directory.

## Includes
`bbopt` cloned on March 1st 2019.

