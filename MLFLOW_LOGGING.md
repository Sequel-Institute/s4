# MLflow Logging Integration

This document describes the MLflow logging integration added to the S4 training pipeline.

## Overview

MLflow tracking has been integrated into the training script to provide comprehensive experiment tracking, including:

- **System Metrics**: CPU, GPU, memory usage tracking
- **Timing Statistics**: Epoch duration, total training time, and timing statistics
- **Hyperparameters**: Model, optimizer, scheduler, and training configuration
- **Model Artifacts**: Automatic model checkpointing and logging
- **Training Metrics**: Loss, accuracy, and custom metrics from PyTorch Lightning

## Features

The MLflow integration mimics the metrics logged in the organics library training scripts and includes:

1. **Automatic PyTorch Logging**: Uses `mlflow.pytorch.autolog()` to automatically capture:
   - Model architecture
   - Training and validation metrics
   - Model checkpoints
   - Optimizer state

2. **Custom Timing Callback**: Tracks and logs:
   - Per-epoch training time
   - Average, minimum, and maximum epoch times
   - Total training time
   - Standard deviation of epoch times

3. **System Metrics**: Automatic tracking of:
   - CPU usage
   - GPU utilization (if available)
   - Memory usage
   - Network I/O

4. **Comprehensive Hyperparameter Logging**:
   - Model configuration (d_model, n_layers, etc.)
   - Optimizer settings (lr, weight_decay, etc.)
   - Scheduler configuration
   - Data loader settings (batch_size, etc.)
   - Training configuration (seed, ema, etc.)
   - Model parameter counts

5. **Metadata Tags**:
   - Model name
   - Dataset name
   - Experiment name
   - Task type
   - Accelerator type (CPU/GPU/MPS)
   - Device information
   - Training status (completed/failed)

## Usage

### Basic Usage

MLflow logging is enabled by default. To run training with MLflow:

```bash
uv run python -m train experiment=sc/s4-sc.yaml
```

This will:
- Create an MLflow experiment named "S4 Speech Commands"
- Track all metrics and hyperparameters
- Save model checkpoints
- Log system metrics every second

### Viewing Results

After training, view results using the MLflow UI:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.

### Configuration

MLflow can be configured in your experiment YAML files or the main config:

```yaml
mlflow:
  enabled: true  # Enable/disable MLflow logging
  experiment_name: "My Experiment"  # Name of the MLflow experiment
  system_metrics_interval: 1  # System metrics sampling interval (seconds)
  log_every_n_epoch: 1  # Log model metrics every N epochs
  log_models: true  # Whether to log the model
  checkpoint: true  # Whether to log checkpoints
  # tracking_uri: "http://localhost:5000"  # Optional: Remote MLflow server
```

### Disabling MLflow

To disable MLflow logging:

```bash
uv run python -m train experiment=sc/s4-sc.yaml mlflow.enabled=false
```

Or set `~mlflow` in your config:

```yaml
~mlflow:  # This disables MLflow entirely
```

## Integration with WandB

MLflow and WandB can run simultaneously. Both loggers are independent and will track metrics in parallel.

To use both:
```bash
uv run python -m train experiment=sc/s4-sc.yaml wandb.mode=online mlflow.enabled=true
```

To use only MLflow (disable WandB):
```bash
uv run python -m train experiment=sc/s4-sc.yaml ~wandb mlflow.enabled=true
```

## Metrics Logged

### Timing Metrics
- `epoch_time_seconds`: Duration of each epoch
- `avg_epoch_time_seconds`: Running average of epoch times
- `min_epoch_time_seconds`: Minimum epoch time observed
- `max_epoch_time_seconds`: Maximum epoch time observed
- `total_training_time_seconds`: Total training duration
- `total_training_time_minutes`: Total training duration in minutes
- `epoch_time_std_dev_seconds`: Standard deviation of epoch times

### Training Metrics (via PyTorch Lightning)
- `train/loss`: Training loss
- `train/accuracy`: Training accuracy (if applicable)
- `val/loss`: Validation loss
- `val/accuracy`: Validation accuracy
- Custom metrics defined in your task

### System Metrics (automatic)
- CPU usage percentage
- GPU utilization and memory
- System memory usage
- Network I/O

## Parameters Logged

- Model architecture parameters (d_model, n_layers, dropout, etc.)
- Optimizer configuration (lr, weight_decay, betas, etc.)
- Scheduler settings
- Data loader configuration (batch_size, etc.)
- Training settings (seed, ema, max_epochs, etc.)
- Model parameter counts (total and trainable)

## Tags Logged

- `model_name`: Name of the model architecture
- `experiment_name`: Name of the experiment
- `dataset_name`: Dataset being used
- `task_name`: Task type (classification, regression, etc.)
- `accelerator`: Device type (cpu, gpu, mps)
- `device_name`: Specific device name (e.g., "NVIDIA A100")
- `status`: Training status (completed/failed)
- `error`: Error message if training failed

## Advanced Usage

### Custom Tracking URI

To use a remote MLflow server:

```yaml
mlflow:
  tracking_uri: "http://your-mlflow-server:5000"
```

Or set the environment variable:

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
uv run python -m train experiment=sc/s4-sc.yaml
```

### Custom Experiment Names

Set experiment names per run:

```bash
uv run python -m train experiment=sc/s4-sc.yaml mlflow.experiment_name="My Custom Experiment"
```

### Comparing Runs

Use the MLflow UI to:
1. Compare metrics across multiple runs
2. Visualize training curves
3. Download model artifacts
4. View system metrics
5. Compare hyperparameters

## Troubleshooting

### MLflow Not Logging

If MLflow isn't logging, check:
1. MLflow is installed: `pip install mlflow`
2. MLflow is enabled in config: `mlflow.enabled=true`
3. Check logs for MLflow-related warnings
4. Ensure write permissions for `./mlruns` directory

### Viewing Logs

MLflow-related log messages are prefixed with:
- "MLflow experiment set to: ..."
- "MLflow PyTorch autologging enabled"
- "Started MLflow run: ..."
- "MLflow run completed successfully"

Check console output or log files for these messages.

### Common Issues

**Issue**: MLflow UI shows no experiments
**Solution**: Ensure you're running `mlflow ui` in the same directory where `./mlruns` was created

**Issue**: System metrics not appearing
**Solution**: System metrics may take a minute to populate. Check the "System Metrics" tab in the MLflow UI.

**Issue**: Models not being logged
**Solution**: Ensure `mlflow.log_models=true` and `mlflow.checkpoint=true` in your config

## Comparison with Organics Library

This implementation matches the organics library MLflow integration with:

✅ Timing callback with comprehensive statistics
✅ System metrics logging
✅ PyTorch autologging
✅ Hyperparameter logging
✅ Model checkpointing
✅ Status tags and error tracking
✅ Device and accelerator information
✅ Parameter counting

## Example Output

When training starts, you'll see:

```
[INFO] MLflow experiment set to: S4 Speech Commands
[INFO] MLflow PyTorch autologging enabled
[INFO] Added MLflow timing callback
[INFO] Started MLflow run: abc123def456
[INFO] Total parameters: 123,456
[INFO] Trainable parameters: 123,456
[INFO] MLflow parameters logged
[INFO] Epoch 0 started
[INFO] Epoch 0 completed in 12.34s
[INFO]   Avg epoch time: 12.34s, Min: 12.34s, Max: 12.34s
...
[INFO] Training completed!
[INFO] Total training time: 493.20s (8.22 minutes)
[INFO] MLflow run completed successfully
[INFO] MLflow run ended
```

## Additional Resources

- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [MLflow Python API](https://www.mlflow.org/docs/latest/python_api/index.html)
- [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)

