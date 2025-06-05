import torch.nn as nn

def block_from_token(token, in_channels, out_channels):
    if token == "conv3x3":
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    elif token == "conv5x5":
        return nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
    elif token == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown block: {token}")

def output_from_token(token, num_classes):
    if token == "softmax":
        return nn.Softmax(dim=1)
    elif token == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown output: {token}")

class GenotypeNet(nn.Module):
    def __init__(self, genotype, in_channels=3, out_channels=16, num_classes=10):
        super().__init__()
        tokens = genotype.split()
        block_tokens = tokens[:-1]
        output_token = tokens[-1]

        layers = []
        for token in block_tokens:
            layers.append(block_from_token(token, in_channels, out_channels))
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(out_channels, num_classes))
        layers.append(output_from_token(output_token, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
