import torch
from pprint import pprint

from einspace.compiler import Compiler
from space import CustomEinSpace
    from model_decoder import decode_genotype_to_model

if __name__ == "__main__":
    torch.manual_seed(0)
    input_shape = (1, 3, 32, 32)

    space = CustomEinSpace(
        input_shape=input_shape,
        input_mode="im",
        num_repeated_cells=1,
        computation_module_prob=0.5,
        min_module_depth=1,
        max_module_depth=3,
        device="cpu"
    )

    arch = space.sample()
    print("\n🔹 Sampled architecture:")
    pprint(arch)

    model = Compiler().compile(arch)
    print("\n🔹 Compiled model:")
    print(model)

    x = torch.randn(input_shape)
    y = model(x)
    print("\n🔹 Output shape:", y.shape)

