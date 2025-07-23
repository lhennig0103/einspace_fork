import torch
import torch.nn as nn
from einspace.layers import EinLinear  # For optional projection before/after

class RNNBlock(nn.Module):
    def __init__(
        self,
        input_size=64,
        hidden_size=64,
        rnn_type="lstm",      # "gru" or "rnn"
        num_layers=2,
        bidirectional=False,
        batch_first=True,
        use_projection=False,
        **kwargs
    ):
        super().__init__()

        rnn_class = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn": nn.RNN
        }[rnn_type.lower()]

        self.use_projection = use_projection
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first
        )

        out_features = hidden_size * (2 if bidirectional else 1)

        if use_projection:
            self.proj = EinLinear(out_features, input_size)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        # Expected input shape: [B, C, L] → transpose if needed
        if not self.batch_first:
            x = x.permute(2, 0, 1)  # [L, B, C]
        else:
            x = x.permute(0, 2, 1)  # [B, L, C]

        out, _ = self.rnn(x)
        out = self.proj(out)

        # Back to [B, C, L]
        if not self.batch_first:
            out = out.permute(1, 2, 0)
        else:
            out = out.permute(0, 2, 1)
        return out
    
class UnrolledRNN(nn.Module):
    def __init__(self, cell_fn, depth=4, input_shape=None, **kwargs):
        super().__init__()
        self.depth = depth
        self.cells = nn.ModuleList([
            cell_fn(input_shape=input_shape, **kwargs)
            for _ in range(depth)
        ])

    def forward(self, x):
        for cell in self.cells:
            x = cell(x)
        return x

# Factory functions for search space
def rnn_lstm(**kwargs):
    input_dim = kwargs["input_shape"][1]
    return RNNBlock(input_size=input_dim, hidden_size=input_dim, rnn_type="lstm", **kwargs)

def rnn_gru(**kwargs):
    input_dim = kwargs["input_shape"][1]
    return RNNBlock(input_size=input_dim, hidden_size=input_dim, rnn_type="gru", **kwargs)

def rnn_basic(**kwargs):
    input_dim = kwargs["input_shape"][1]
    return RNNBlock(input_size=input_dim, hidden_size=input_dim, rnn_type="rnn", **kwargs)

def unrolled_rnn_lstm(**kwargs):
    return UnrolledRNN(cell_fn=rnn_lstm, depth=4, **kwargs)

def unrolled_rnn_gru(**kwargs):
    return UnrolledRNN(cell_fn=rnn_gru, depth=4, **kwargs)

# def unrolled_computation_rnn(**kwargs):
#     return UnrolledRNN(cell_fn=computation_module, depth=4, **kwargs)