import torch
import torch.nn as nn

def block_from_token(token, in_channels, out_channels):
    """Map string tokens to PyTorch layers."""
    if token == "conv3x3":
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    elif token == "conv5x5":
        return nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
    elif token == "identity":
        return nn.Identity()
    elif token == "relu":
        return nn.ReLU()
    elif token == "bn":
        return nn.BatchNorm2d(out_channels)
    else:
        raise ValueError(f"Unknown block token: {token}")

def output_from_token(token, num_classes):
    """Map final activation token to output layer."""
    if token == "softmax":
        return nn.Softmax(dim=1)
    elif token == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown output token: {token}")

def decode_genotype_to_model(genotype: str, input_channels=3, hidden_channels=64, num_classes=10):
    """
    Decodes a genotype string into a PyTorch model.

    Args:
        genotype: str, like "conv3x3 relu conv5x5 softmax"
        input_channels: input channels (e.g. 3 for RGB images)
        hidden_channels: number of output channels used in conv layers
        num_classes: output classes for the classifier

    Returns:
        nn.Sequential model
    """
    tokens = genotype.strip().split()
    if len(tokens) < 2:
        raise ValueError("Genotype must contain at least one block and one output token.")

    block_tokens = tokens[:-1]
    output_token = tokens[-1]

    layers = []
    in_channels = input_channels

    for token in block_tokens:
        layer = block_from_token(token, in_channels, hidden_channels)
        layers.append(layer)

        # update in_channels only after convolution layers
        if isinstance(layer, nn.Conv2d):
            in_channels = hidden_channels

    # Classification head
    layers.extend([
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(hidden_channels, num_classes),
        output_from_token(output_token, num_classes),
    ])

    return nn.Sequential(*layers)

# === Example usage ===
if __name__ == "__main__":
    genotype = "conv3x3 relu conv5x5 bn relu identity softmax"
    model = decode_genotype_to_model(genotype)

    x = torch.randn(1, 3, 32, 32)
    y = model(x)

    print("🔹 Genotype:", genotype)
    print("🔹 Model:\n", model)
    print("🔹 Output shape:", y.shape)