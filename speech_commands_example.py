"""Example usage of the Speech Commands Dataset.

This script demonstrates how to use the refactored Speech Commands dataset
with PyTorch and PyTorch Lightning.
"""

import torch
from torch.utils.data import DataLoader
import speech_commands_dataset as scd

def example_basic_usage():
    """Example 1: Basic dataset usage."""
    print("=" * 60)
    print("Example 1: Basic Dataset Usage")
    print("=" * 60)

    # Create a single dataset
    train_dataset = scd.SpeechCommandsDataset(
        root='./data',
        subset='train',
        classes='10_words',
        mfcc=True,
        download=True  # Will download if not present
    )

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Input shape: {train_dataset.input_shape}")
    print(f"Class names: {train_dataset.class_names}")

    # Get a sample
    x, y = train_dataset[0]
    print(f"\nSample shapes:")
    print(f"  Input (x): {x.shape}")
    print(f"  Target (y): {y.shape if isinstance(y, torch.Tensor) else 'scalar'}")
    print()


def example_with_dataloader():
    """Example 2: Using with PyTorch DataLoader."""
    print("=" * 60)
    print("Example 2: Using with DataLoader")
    print("=" * 60)

    # Create dataset
    train_dataset = scd.SpeechCommandsDataset(
        root='./data',
        subset='train',
        classes='digits',
        mfcc=True
    )

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    # Iterate through one batch
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Input batch shape: {batch_x.shape}")
        print(f"  Target batch shape: {batch_y.shape}")
        print(f"  Target values (first 5): {batch_y[:5]}")
        break
    print()


def example_convenience_functions():
    """Example 3: Using convenience functions."""
    print("=" * 60)
    print("Example 3: Convenience Functions")
    print("=" * 60)

    # Method 1: Create datasets separately
    train_ds, val_ds, test_ds = scd.create_datasets(
        root='./data',
        classes='10_words',
        mfcc=True
    )

    print(f"Train dataset: {len(train_ds)} samples")
    print(f"Val dataset: {len(val_ds)} samples")
    print(f"Test dataset: {len(test_ds)} samples")

    # Method 2: Create dataloaders directly
    train_loader, val_loader, test_loader = scd.create_dataloaders(
        root='./data',
        classes='10_words',
        mfcc=True,
        batch_size=64,
        num_workers=0
    )

    print(f"\nDataLoaders created with batch_size=64")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print()


def example_different_configurations():
    """Example 4: Different dataset configurations."""
    print("=" * 60)
    print("Example 4: Different Configurations")
    print("=" * 60)

    configs = [
        {"classes": "10_words", "mfcc": False, "task": "classification"},
        {"classes": "digits", "mfcc": True, "task": "classification"},
        {"classes": "all_35", "mfcc": False, "task": "classification"},
    ]

    for i, config in enumerate(configs, 1):
        dataset = scd.SpeechCommandsDataset(
            root='./data',
            subset='train',
            **config
        )
        print(f"Config {i}: {config}")
        print(f"  Samples: {len(dataset)}")
        print(f"  Classes: {dataset.num_classes}")
        print(f"  Input shape: {dataset.input_shape}")
        print()


def example_pytorch_lightning():
    """Example 5: PyTorch Lightning DataModule."""
    print("=" * 60)
    print("Example 5: PyTorch Lightning DataModule")
    print("=" * 60)

    try:
        import pytorch_lightning as pl

        # Create DataModule
        dm = scd.SpeechCommandsDataModule(
            root='./data',
            classes='10_words',
            mfcc=True,
            batch_size=32,
            num_workers=0
        )

        print(f"DataModule created!")
        print(f"  Number of classes: {dm.num_classes}")
        print(f"  Input shape: {dm.input_shape}")

        # Prepare data (download if needed)
        dm.prepare_data()

        # Setup datasets
        dm.setup('fit')

        print(f"\nDatasets ready:")
        print(f"  Train: {len(dm.train_dataset)} samples")
        print(f"  Val: {len(dm.val_dataset)} samples")

        # Get dataloaders
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        print(f"\nDataLoaders:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

        print("\nReady to use with: trainer.fit(model, dm)")

    except ImportError:
        print("PyTorch Lightning not installed, skipping this example")
    print()


def example_training_loop():
    """Example 6: Simple training loop."""
    print("=" * 60)
    print("Example 6: Simple Training Loop")
    print("=" * 60)

    # Create datasets
    train_loader, val_loader, _ = scd.create_dataloaders(
        root='./data',
        classes='10_words',
        mfcc=True,
        batch_size=32
    )

    # Simple model (just for demonstration)
    class SimpleClassifier(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            self.fc = torch.nn.Linear(input_size[0] * input_size[1], num_classes)

        def forward(self, x):
            return self.fc(self.flatten(x))

    model = SimpleClassifier(input_size=(161, 20), num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop (just one batch for demonstration)
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        print(f"Batch processed!")
        print(f"  Input shape: {batch_x.shape}")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Loss: {loss.item():.4f}")
        break

    print("\nTraining loop works correctly!")
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Speech Commands Dataset - Usage Examples")
    print("="*60 + "\n")

    # Run examples
    # Note: Commented out examples that require downloading data
    # Uncomment them when you want to test with actual data

    # example_basic_usage()
    # example_with_dataloader()
    # example_convenience_functions()
    # example_different_configurations()
    # example_pytorch_lightning()
    # example_training_loop()

    print("\nTo run these examples:")
    print("1. Uncomment the desired example in the __main__ block")
    print("2. Run: uv run python speech_commands_example.py")
    print("\nNote: First run will download the dataset (~2GB)")
