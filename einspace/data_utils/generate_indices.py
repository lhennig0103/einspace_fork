import os
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def generate_cifar100_indices(root, val_split=0.1, seed=42):
    # Folder to save index files
    save_dir = os.path.join(root, "cifar100")
    os.makedirs(save_dir, exist_ok=True)

    # Load CIFAR-100 (train split contains all 50k labeled images)
    transform = transforms.Compose([transforms.ToTensor()])
    cifar100 = datasets.CIFAR100(root=save_dir, train=True, download=False, transform=transform)

    labels = torch.tensor(cifar100.targets)
    indices = torch.arange(len(labels))

    # Stratified split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=seed
    )

    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)

    # Save indices
    train_path = os.path.join(save_dir, "cifar100_train.indices")
    val_path = os.path.join(save_dir, "cifar100_valid.indices")

    torch.save(train_idx, train_path)
    torch.save(val_idx, val_path)

    print("Saved:")
    print(f"  Train indices → {train_path}")
    print(f"  Val indices   → {val_path}")
    print(f"  Train count: {len(train_idx)}, Val count: {len(val_idx)}")

if __name__ == "__main__":
    generate_cifar100_indices(root="/hpcwork/p0025023/einspace_fork/data/", val_split=0.2)
