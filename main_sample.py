from model_decoder import GenotypeNet
from einspace.search_spaces.minimal_space import MinimalSearchSpace
from model_decoder import GenotypeNet
import torch

def main():
    space = MinimalSearchSpace()
    sample = space.sample()
    phenotype = space.decode(sample)

    print(f"Genotype: {sample}")
    print(f"Phenotype: {phenotype}")

    model = GenotypeNet(phenotype)
    print(model)

    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
