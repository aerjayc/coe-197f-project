from torch import nn
from model import conv_layer

class resnet_layer(nn.Module):
    def __init__(self,
                 in_channels,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
        super(resnet_layer, self).__init__()

        self.batch_normalization = batch_normalization
        self.conv_first = conv_first

        out_channels = num_filters
        padding = kernel_size // 2      # 'same' padding

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=strides,
                              padding=padding)

        # 'he_normal' initialization
        nn.init.kaiming_normal_(self.conv.weight)

        # l2 regularization
        # not implemented

        # batch normalization
        self.batchnorm_first = nn.BatchNorm2d(in_channels)
        self.batchnorm_last = nn.BatchNorm2d(num_filters)

        # activation (only 'relu' is implemented)
        if activation:
            self.activation = nn.ReLU()
        else:
            self.activation = None


    def forward(self, x):
        if self.conv_first:
            x = self.conv(x)
            if self.batch_normalization:
                x = self.batchnorm_last(x)
            if self.activation is not None:
                x = self.activation(x)
        else:
            if self.batch_normalization:
                x = self.batchnorm_first(x)
            if self.activation is not None:
                x = self.activation(x)
            x = self.conv(x)
        
        return x


# backbone
class resnet_v2(nn.Module):
    """
    # Arguments
        input_shape (tensor): Shape of the input image tensor, assumed to be
                              (N, C, H, W) following PyTorch tensor convention
        depth (int): Number of convolutional layers
        num_classes (int): Number of classes
    """
    def __init__(self, input_shape, depth, n_layers=4):
        super(resnet_v2, self).__init__()

        # name = 'ResNet%dv2' % (depth)
        self.n_layers = n_layers

        # copied
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2')

        num_filters_in = 16
        num_filters_out = None
        self.resnet1 = resnet_layer(input_shape[1],
                                    num_filters=num_filters_in,
                                    conv_first=True)

        self.resnet_blocks = nn.ModuleList()
        self.num_res_blocks = (depth - 2) // 9
        in_channels = num_filters_in
        for stage in range(3):
            initial_in_channels = in_channels
            for res_block in range(self.num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2 # downsample

                # bottleneck residual unit
                self.resnet_blocks.append(
                    resnet_layer(in_channels,
                                 num_filters=num_filters_in,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=activation,
                                 batch_normalization=batch_normalization,
                                 conv_first=False))
                in_channels = num_filters_in

                self.resnet_blocks.append(
                    resnet_layer(in_channels,
                                 num_filters=num_filters_in,
                                 conv_first=False))
                in_channels = num_filters_in

                self.resnet_blocks.append(
                    resnet_layer(in_channels,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 conv_first=False))
                in_channels = num_filters_out

                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    self.resnet_blocks.append(
                        resnet_layer(initial_in_channels,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False))
                    in_channels = num_filters_out

            num_filters_in = num_filters_out    # don't use self.num_filters_in
                                                # as it persists between calls

        self.batchnorm = nn.BatchNorm2d(num_filters_out)
        self.activation = nn.ReLU()

        self.features_pyramid = features_pyramid(num_filters_out,
                                                 n_layers)


    def forward(self, x):
        x = self.resnet1(x)
        # Instantiate the stack of residual units
        i = 0
        for stage in range(3):
            for res_block in range(self.num_res_blocks):
                y = self.resnet_blocks[i](x)
                i += 1
                y = self.resnet_blocks[i](y)
                i += 1
                y = self.resnet_blocks[i](y)
                i += 1
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_blocks[i](x)
                    i += 1
                x = x + y

        # v2 has BN-ReLU before Pooling
        x = self.batchnorm(x)
        x = self.activation(x)
        # 1st feature map layer

        # main feature maps (160, 120)
        # succeeding feature maps scaled down by
        # 2, 4, 8
        outputs = self.features_pyramid(x)

        return outputs


class features_pyramid(nn.Module):
    def __init__(self,
                 in_channels,
                 n_layers):
        super(features_pyramid, self).__init__()

        pool_size = 2
        self.avg_pool = nn.AvgPool2d(pool_size)

        n_layers = n_layers
        n_filters = 512

        self.conv_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            postfix = "_layer" + str(i+2)
            layer = conv_layer(in_channels,
                              filters=n_filters,
                              kernel_size=3,
                              strides=2,
                              use_maxpool=False,
                              postfix=postfix)
            self.conv_layers.append(layer)
            in_channels = n_filters


    def forward(self, x):
        outputs = [x]
        conv = self.avg_pool(x)
        outputs.append(conv)
        prev_conv = conv

        # additional feature map layers
        for convlayer in self.conv_layers:
            conv = convlayer(prev_conv)
            outputs.append(conv)
            prev_conv = conv

        return outputs


def build_resnet(input_shape=(480, 640, 3),
                 n_layers=4,
                 n=6):
    depth = n * 9 + 2
    
    return resnet_v2(input_shape,
                     depth=depth,
                     n_layers=n_layers)
