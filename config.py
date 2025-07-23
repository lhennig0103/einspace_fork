import torch.nn as nn
from einspace.layers import identity as ein_identity

# === Activations from activations.py ===
from einspace.activations import (
    identity, relu, leakyrelu, prelu, sigmoid, swish,
    tanh, softplus, softsign, sin, square, cubic, abs, softmax
)

# === Layer sets from einspace.layers ===
from einspace.layers import (
    # Structure
    sequential_module,
    computation_module,
    routing_module,

    # Linear layers
    linear32, linear64, linear128,

    # Conv layers
    conv1d3k1s1p64d,

    # Pooling
    maxpool3k2s1p,
    adaptiveavgpool,

    # Routing (tensor rearrangement)
    im2col3k1s1p,
    col2im3k1s1p,

    # Positional encoding
    positional_encoding,
    learnable_positional_encoding,

    # Tensor operations
    dot_product,
    scaled_dot_product,
    add_tensors,
    cat_tensors1d2t,
)

# === Transformer Components ===
from einspace.transformers import (
    mha_self_attention,
    transformer_encoder_block,
    ff_block,
    transformer_norm,
)

# === RNN Components ===
from einspace.rnns import (
    rnn_lstm,
    rnn_gru,
    rnn_basic,
)

# === Dropout
def dropout(**kwargs):
    return nn.Dropout(p=0.5)

def dropout_low(**kwargs):
    return nn.Dropout(p=0.2)

# === Modular Subspaces ===

# 1. Structural modules
structure_ops = [sequential_module, computation_module, routing_module]

# 2. Basic computation ops
linear_ops = [linear32, linear64, linear128]
conv_ops = [conv1d3k1s1p64d]
dropout_ops = [dropout, dropout_low]
pooling_ops = [maxpool3k2s1p, adaptiveavgpool]

# 3. Activations
activation_ops = [
    identity, relu, leakyrelu, prelu, sigmoid,
    swish, tanh, softplus, softsign, sin, square, cubic, abs, softmax,
]

# 4. Routing
routing_pre = [im2col3k1s1p, positional_encoding, ein_identity]
routing_post = [col2im3k1s1p, learnable_positional_encoding, ein_identity]

# 5. Positional encodings (standalone)
positional_ops = [positional_encoding, learnable_positional_encoding]

# 6. Tensor-level algebra
tensor_ops = [
    dot_product, scaled_dot_product,
    add_tensors, cat_tensors1d2t,
]

# 7. Transformer-related
attention_ops = [mha_self_attention, transformer_encoder_block, ff_block]
normalization_ops = [transformer_norm]

# 8. RNN-related
rnn_ops = [rnn_lstm, rnn_gru, rnn_basic]

# === Final Config ===
my_cfg = {
    "network": structure_ops,

    "first_fn": conv_ops + linear_ops + attention_ops + rnn_ops + [ein_identity],
    "second_fn": conv_ops + linear_ops + attention_ops + rnn_ops + [ein_identity],

    "computation_fn": (
        conv_ops +
        linear_ops +
        rnn_ops +
        dropout_ops +
        pooling_ops +
        activation_ops +
        tensor_ops +
        positional_ops +
        attention_ops +
        normalization_ops
    ),

    "prerouting_fn": routing_pre,
    "postrouting_fn": routing_post,
    "branching_fn": [ein_identity],
    "inner_fn": [computation_module],
    "aggregation_fn": [add_tensors, cat_tensors1d2t, ein_identity],
}
