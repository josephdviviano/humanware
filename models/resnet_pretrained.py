from torchvision import models
import torch
import torch.nn as nn


def set_parameter_requires_grad(mdl, fine_tune):
    """turns off gradient updates if we aren't fine tuning"""
    if not fine_tune:
        for param in mdl.parameters():
            param.requires_grad = False


def resnet_pretrained(fine_tune=True):
    """Initializes a resnet that can be fine-tuned."""

    print('initializing resnet with fine_tuning={}'.format(fine_tune))

    mdl = models.resnet101(pretrained=True)  # resnet18,34,50,101,152.

    # upper layers are set to requires_grad if fine_tune is true
    set_parameter_requires_grad(model, fine_tune)

    # this new layer is always trained
    linear_in_features = model.fc.in_features
    mdl.fc = nn.Linear(linear_in_features, 31)  # 7+10+10+10+10+10.
    nn.init.xavier_uniform_(list(model.fc.parameters())[0])  # Init weights.

    # If fine_tune is set to true, sets all parameters require_grad
    # otherwise only the newly added layers (above) require_grad.
    params_to_update = model.parameters()

    LOGGER.debug("Params to learn:")
    if fine_tune:
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                LOGGER.debug("\t{}".format(str(name)))
    else:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                LOGGER.debug("\t{}".format(str(name)))

    return(mdl)
