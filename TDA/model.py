import torch.nn as nn
import backbones

class TransferModel(nn.Module):
    def __init__(self, num_class, base_net='resnet152', use_bottleneck=True, bottleneck_width=256):
        super(TransferModel, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()
        self.classifier_layer = nn.Linear(feature_dim, num_class)

    def forward(self, x):
        features = self.base_network(x)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        outputs = self.classifier_layer(features)
        return outputs
