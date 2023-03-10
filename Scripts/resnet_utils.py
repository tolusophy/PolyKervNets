import collections

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple("GlobalParams", [
    "block", "zero_init_residual",
    "groups", "width_per_group", "replace_stride_with_dilation",
    "norm_layer", "num_classes", "image_size"])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def resnet_params(model_name):
    """ Map resnet_pytorch model name to parameter coefficients. """

    params_dict = {
        # Coefficients:   block, res
        "resnet10": (BasicBlock, 224),
        "resnet10_2": (BasicBlock, 224),
        "resnet10_3": (BasicBlock, 224),
        "resnet14": (BasicBlock, 224),
        "resnet14_2": (BasicBlock, 224),
        "resnet14_3": (BasicBlock, 224),
        "resnet18": (BasicBlock, 224),
        "resnet18_2": (BasicBlock, 224),
        "resnet18_3": (BasicBlock, 224),
        "resnet32": (BasicBlock, 224),
        "resnet32_2": (BasicBlock, 224),
        "resnet32_3": (BasicBlock, 224),
        "resnet50": (Bottleneck, 224),
        "resnet50_2": (Bottleneck, 224),
        "resnet50_3": (Bottleneck, 224),
        "resnet101": (Bottleneck, 224),
        "resnet101_2": (Bottleneck, 224),
        "resnet101_3": (Bottleneck, 224),
        "resnet152": (Bottleneck, 224),
        "resnet152_2": (Bottleneck, 224),
        "resnet152_3": (Bottleneck, 224)
    }
    return params_dict[model_name]


def resnet(arch, block, num_classes=1000, zero_init_residual=False,
           groups=1, width_per_group=64, replace_stride_with_dilation=None,
           norm_layer=None, image_size=224):
    """ Creates a resnet_pytorch model. """

    global_params = GlobalParams(
        block=block,
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
        groups=groups,
        width_per_group=width_per_group,
        replace_stride_with_dilation=replace_stride_with_dilation,
        norm_layer=norm_layer,
        image_size=image_size,
    )

    layers_dict = {
        "resnet10": (1, 1, 1, 1),
        "resnet10_2": (2, 2, 0, 0),
        "resnet10_3": (2, 1, 1, 0),
        "resnet14": (2, 2, 1, 1),
        "resnet14_2": (4, 2, 0, 0),
        "resnet14_3": (2, 2, 2, 0),
        "resnet18": (2, 2, 2, 2),
        "resnet18_2": (4, 4, 0, 0),
        "resnet18_3": (4, 2, 2, 0),
        "resnet32": (3, 4, 6, 3),
        "resnet32_2": (10, 6, 0, 0),
        "resnet32_3": (8, 4, 4, 0),
        "resnet50": (3, 4, 6, 3),
        "resnet50_2": (10, 6, 0, 0),
        "resnet50_3": (8, 4, 4, 0),
        "resnet101": (3, 4, 23, 3),
        "resnet101_2": (23, 10, 0, 0),
        "resnet101_3": (17, 8, 8, 0),
        "resnet152": (3, 8, 36, 3),
        "resnet152_2": (36, 14, 0, 0),
        "resnet152_3": (18, 16, 16, 0),
    }
    layers = layers_dict[arch]

    return layers, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith("resnet"):
        b, s = resnet_params(model_name)
        layers, global_params = resnet(arch=model_name, block=b, image_size=s)
    else:
        raise NotImplementedError(f"model name is not pre-defined: {model_name}")
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return layers, global_params


urls_map = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def load_pretrained_weights(model, model_name, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(urls_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == {"fc.weight", "fc.bias"}, "issue loading pretrained weights"
    print(f"Loaded pretrained weights for {model_name}.")