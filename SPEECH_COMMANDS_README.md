# Speech Commands Dataset

A PyTorch-compatible dataset loader for the Google Speech Commands dataset with optional MFCC preprocessing, designed for seamless integration with PyTorch and PyTorch Lightning.

## Features

- **Multiple class configurations**: 10-word subset, digits (0-9), or all 35 classes
- **Flexible preprocessing**: Raw audio (16kHz) or MFCC features (20 coefficients)
- **Task support**: Classification and generation tasks
- **Automatic caching**: Preprocessed data is cached to disk for faster subsequent loads
- **PyTorch Lightning integration**: Built-in DataModule for easy training
- **Convenient utilities**: Helper functions to create datasets and dataloaders

## Installation

The dataset requires:
- PyTorch
- torchaudio
- scikit-learn
- PyTorch Lightning (optional, for DataModule)

## Quick Start

### Basic Usage

```python
import speech_commands_dataset as scd

# Create a training dataset with MFCC features
train_dataset = scd.SpeechCommandsDataset(
    root='./data',
    subset='train',
    classes='10_words',
    mfcc=True
)

print(f"Dataset size: {len(train_dataset)}")
print(f"Number of classes: {train_dataset.num_classes}")
print(f"Input shape: {train_dataset.input_shape}")

# Get a sample
x, y = train_dataset[0]
print(f"Sample: x.shape={x.shape}, y={y}")
```

### Using with DataLoader

```python
from torch.utils.data import DataLoader

# Create datasets
train_ds, val_ds, test_ds = scd.create_datasets(
    root='./data',
    classes='10_words',
    mfcc=True
)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Or use the convenience function
train_loader, val_loader, test_loader = scd.create_dataloaders(
    root='./data',
    classes='10_words',
    mfcc=True,
    batch_size=32
)
```

### PyTorch Lightning

```python
import pytorch_lightning as pl

# Create DataModule
dm = scd.SpeechCommandsDataModule(
    root='./data',
    classes='10_words',
    mfcc=True,
    batch_size=64
)

# Use with Lightning Trainer
model = YourLightningModel(
    input_shape=dm.input_shape,
    num_classes=dm.num_classes
)

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, dm)
```

## API Reference

### SpeechCommandsDataset

Main dataset class for loading Speech Commands data.

**Parameters:**
- `root` (str): Root directory for dataset storage. Default: `"./data"`
- `subset` (str): Data split to load - `'train'`, `'val'`, or `'test'`. Default: `'train'`
- `classes` (str): Class configuration:
  - `'10_words'`: Commands [yes, no, up, down, left, right, on, off, stop, go]
  - `'digits'`: Digits [zero, one, ..., nine]
  - `'all_35'`: All 35 classes in the dataset
- `mfcc` (bool): Use MFCC features (20 coeffs, 161 timesteps) vs raw audio (16000 samples). Default: `False`
- `task` (str): Task type - `'classification'` or `'generation'`. Default: `'classification'`
- `download` (bool): Download dataset if not found. Default: `True`
- `subsample_rate` (int): Subsample every kth sample (raw audio only). Default: `1`
- `normalize` (bool): Normalize using training set statistics. Default: `True`

**Properties:**
- `num_classes`: Number of classes in the dataset
- `input_shape`: Shape of a single input sample `(length, channels)`
- `output_shape`: Shape of target/output
- `class_names`: List of class name strings

### Convenience Functions

#### `create_datasets()`

Create train, validation, and test datasets with the same configuration.

```python
train_ds, val_ds, test_ds = scd.create_datasets(
    root='./data',
    classes='10_words',
    mfcc=True
)
```

#### `create_dataloaders()`

Create train, validation, and test dataloaders with the same configuration.

```python
train_loader, val_loader, test_loader = scd.create_dataloaders(
    root='./data',
    classes='10_words',
    mfcc=True,
    batch_size=64,
    num_workers=4
)
```

### SpeechCommandsDataModule

PyTorch Lightning DataModule for Speech Commands.

**Parameters:**
- `root` (str): Root directory for dataset
- `classes` (str): Class configuration
- `mfcc` (bool): Use MFCC features
- `task` (str): Task type
- `batch_size` (int): Batch size for dataloaders. Default: `32`
- `num_workers` (int): Number of worker processes. Default: `0`

**Properties:**
- `num_classes`: Number of classes
- `input_shape`: Input shape `(length, channels)`

**Methods:**
- `prepare_data()`: Download dataset (called on 1 GPU)
- `setup(stage)`: Create datasets (called on all GPUs)
- `train_dataloader()`: Get training dataloader
- `val_dataloader()`: Get validation dataloader
- `test_dataloader()`: Get test dataloader

## Dataset Configurations

### Class Options

| Option | Classes | Count | Use Case |
|--------|---------|-------|----------|
| `10_words` | yes, no, up, down, left, right, on, off, stop, go | 10 | Command classification |
| `digits` | zero, one, two, ..., nine | 10 | Digit recognition, generation |
| `all_35` | All available classes | 35 | Full dataset |

### Feature Options

| Option | Shape | Description |
|--------|-------|-------------|
| `mfcc=False` | `(16000, 1)` | Raw audio at 16kHz |
| `mfcc=True` | `(161, 20)` | 20 MFCC coefficients, 161 timesteps |

## Data Splits

- **10_words and digits**: Stratified random splits (70% train, 15% val, 15% test)
- **all_35**: Official dataset splits using `validation_list.txt` and `testing_list.txt`

## Caching

Preprocessed data is automatically cached in:
```
{root}/SpeechCommands/processed_data/{config}/
```

Cache directories are named based on configuration:
- `raw_10_words_classification/` - Raw audio, 10 words, classification
- `mfcc_digits_classification/` - MFCC features, digits, classification
- etc.

## Examples

See `speech_commands_example.py` for comprehensive usage examples including:
1. Basic dataset usage
2. Using with DataLoader
3. Convenience functions
4. Different configurations
5. PyTorch Lightning integration
6. Simple training loop

## Notes

- First run will download the dataset (~2GB) from TensorFlow
- Preprocessing may take a few minutes on first load
- Subsequent loads are fast thanks to caching
- MFCC computation uses: `n_mfcc=20`, `n_fft=200`, `n_mels=64`, `log_mels=True`
- Normalization uses training set statistics only (prevents data leakage)

## Credits

Adapted from:
- https://github.com/dwromero/ckconv
- https://github.com/patrick-kidger/NeuralCDE

Original dataset: [Speech Commands Dataset v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
