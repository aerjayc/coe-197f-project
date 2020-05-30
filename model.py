from torch import nn


class conv_layer(nn.Module):
    def __init__(self,
                 in_channels,
                 filters=32,
                 kernel_size=3,
                 strides=1,
                 use_maxpool=True,
                 postfix=None,          # not implemented
                 activation=None):
        super(conv_layer, self).__init__()

        padding = kernel_size // 2      # 'same' padding
        self.conv = nn.Conv2d(in_channels,
                              filters,
                              kernel_size,
                              stride=strides,
                              padding=padding)
        # 'he_normal' initialization
        nn.init.kaiming_normal_(self.conv.weight)

        self.batchnorm = nn.BatchNorm2d(filters)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(filters)

        self.use_maxpool = use_maxpool


    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        if self.use_maxpool:
            x = self.maxpool(x)

        return x


class tconv_layer(nn.Module):
    def __init__(self,
                 in_channels,
                 filters=32,
                 kernel_size=3,
                 strides=2,
                 postfix=None):
        super(tconv_layer, self).__init__()

        padding = kernel_size // 2                  # 'same' padding
        output_padding = kernel_size - 2*padding    # for odd paddings
        self.conv_transpose = nn.ConvTranspose2d(in_channels,
                                                 filters,
                                                 kernel_size,
                                                 stride=strides,
                                                 padding=padding,
                                                 output_padding=output_padding)
        # 'he_normal' initialization
        nn.init.kaiming_normal_(self.conv_transpose.weight)

        self.batchnorm = nn.BatchNorm2d(filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batchnorm(x)
        x = self.activation(x)

        return x


# analog of `build_fcn`
class fcn(nn.Module):
    def __init__(self,
                 input_shape,
                 backbone,
                 n_layers=4,
                 n_classes=4):
        super(fcn, self).__init__()

        self.backbone = backbone

        size = (input_shape[-2] // 4, input_shape[-1] // 4)
        feature_size = 8
        scale_factor = 2
        filters = 256
        self.upsamplers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        in_channels = 16 * 16
        total_channels = in_channels
        for _ in range(n_layers):
            postfix = "fcn_" + str(feature_size)
            self.conv_layers.append(
                conv_layer(in_channels,
                           filters=filters,
                           use_maxpool=False,
                           postfix=postfix)
            )
            in_channels = 512
            total_channels += filters
            
            postfix = postfix + "_up2d"
            # self.upsamplers.append(
            #     nn.Upsample(scale_factor=scale_factor,
            #                           mode='bilinear')
            # )
            self.upsamplers.append(
                nn.Upsample(size=size,
                            mode='bilinear')
            )
            
            # scale_factor *= 2
            feature_size *= 2
        
        in_channels = total_channels
        filters = 256
        self.tconv_layer1 = tconv_layer(in_channels,
                                        filters=filters,
                                        postfix="up_x2")
        
        in_channels = filters
        self.tconv_layer2 = tconv_layer(in_channels,
                                        filters=filters,
                                        postfix="up_x4")
        
        in_channels = filters
        kernel_size = 1
        padding = kernel_size // 2      # 'same' padding
        self.conv_transpose = nn.ConvTranspose2d(in_channels,
                                                 n_classes,
                                                 kernel_size,
                                                 stride=1,
                                                 padding=padding)
        # 'he_normal' initialization
        nn.init.kaiming_normal_(self.conv_transpose.weight)

        self.logsoftmax = nn.LogSoftmax()
    

    def forward(self, x):
        features = self.backbone(x)

        main_feature = features[0]
        features = features[1:]
        out_features = [main_feature]

        # other half of the features pyramid
        # including upsampling to restore the
        # feature maps to the dimensions
        # equal to 1/4 the image size
        for i, feature in enumerate(features):
            feature = self.conv_layers[i](feature)
            feature = self.upsamplers[i](feature)
            out_features.append(feature)

        # concatenate all upsampled features
        x = torch.cat(out_features, dim=1)          # merge at channel dimension
        # perform 2 additional feature extraction
        # and upsampling
        x = self.tconv_layer1(x)
        x = self.tconv_layer2(x)

        # generate the pixel-wise classifier
        x = self.conv_transpose(x)
        x = self.logsoftmax(x)

        return x
