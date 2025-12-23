# MLflow Quick Start Guide

## Installation

First, ensure mlflow is installed:

```bash
uv sync
```

This will install mlflow and all other dependencies from `pyproject.toml`.

## Quick Test

Before running your full training, verify the MLflow integration:

```bash
uv run python test_mlflow_integration.py
```

Expected output:
```
✓ MLflow experiment set successfully
✓ MLflow run started: <run_id>
✓ Parameters logged successfully
✓ Metrics logged successfully
✓ Tags logged successfully
...
✓ All tests passed! MLflow integration is working correctly.
```

## Running Training with MLflow

### Default Configuration (MLflow Enabled)

```bash
uv run python -m train experiment=sc/s4-sc.yaml
```

This will:
- Create MLflow experiment "S4 Speech Commands"
- Log all metrics, parameters, and system stats
- Save results to `./mlruns/`

### View Results

In a separate terminal, launch the MLflow UI:

```bash
mlflow ui
```

Then open your browser to: http://localhost:5000

### Disable MLflow

If you want to run without MLflow:

```bash
uv run python -m train experiment=sc/s4-sc.yaml mlflow.enabled=false
```

Or disable in config by setting `~mlflow`.

## What Gets Logged

### Timing Metrics (per epoch)
- `epoch_time_seconds` - Duration of each epoch
- `avg_epoch_time_seconds` - Running average
- `min_epoch_time_seconds` - Minimum time observed
- `max_epoch_time_seconds` - Maximum time observed

### Final Training Statistics
- `total_training_time_seconds` - Total training duration
- `total_training_time_minutes` - Total duration in minutes
- `epoch_time_std_dev_seconds` - Std deviation of epoch times

### Training Metrics (automatic via PyTorch Lightning)
- `train/loss` - Training loss per epoch
- `val/loss` - Validation loss
- `val/accuracy` - Validation accuracy
- Custom task metrics

### System Metrics (automatic, sampled every 1 second)
- CPU usage %
- GPU utilization and memory
- System memory usage
- Network I/O

### Parameters
- Model: `model_d_model`, `model_n_layers`, `model_dropout`, etc.
- Optimizer: `optimizer_lr`, `optimizer_weight_decay`, etc.
- Scheduler: `scheduler_num_training_steps`, etc.
- Training: `seed`, `max_epochs`, `loader_batch_size`, etc.
- Model stats: `total_params`, `trainable_params`

### Tags
- `model_name` - Model architecture
- `experiment_name` - Experiment identifier
- `dataset_name` - Dataset being used
- `task_name` - Task type
- `accelerator` - Device type (cpu/gpu/mps)
- `device_name` - GPU model (if applicable)
- `status` - "completed" or "failed"

## Configuration Options

You can customize MLflow behavior in your experiment configs:

```yaml
mlflow:
  enabled: true  # Enable/disable
  experiment_name: "My Custom Experiment"  # Experiment name in UI
  system_metrics_interval: 1  # Sample system metrics every N seconds
  log_every_n_epoch: 1  # Log model every N epochs
  log_models: true  # Log model artifacts
  checkpoint: true  # Log checkpoints
```

## Command-Line Overrides

Change experiment name:
```bash
uv run python -m train experiment=sc/s4-sc.yaml mlflow.experiment_name="Test Run"
```

Disable model logging:
```bash
uv run python -m train experiment=sc/s4-sc.yaml mlflow.log_models=false
```

## Using with WandB

MLflow and WandB work together. Both will log simultaneously:

```bash
# Both enabled (default)
uv run python -m train experiment=sc/s4-sc.yaml

# Only MLflow
uv run python -m train experiment=sc/s4-sc.yaml ~wandb

# Only WandB
uv run python -m train experiment=sc/s4-sc.yaml mlflow.enabled=false
```

## Console Output

During training, you'll see MLflow-related messages:

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
```

## Exploring Results in MLflow UI

1. **Compare Runs**: Select multiple runs and click "Compare"
2. **View Metrics**: Click on a run → "Metrics" tab
3. **View Parameters**: Click on a run → "Parameters" tab
4. **View System Metrics**: Click on a run → "System Metrics" tab
5. **Download Models**: Click on a run → "Artifacts" tab
6. **View Logs**: Available in the run details

## Remote MLflow Server (Optional)

To use a remote MLflow tracking server:

```bash
export MLFLOW_TRACKING_URI=http://your-server:5000
uv run python -m train experiment=sc/s4-sc.yaml
```

Or in config:
```yaml
mlflow:
  tracking_uri: "http://your-server:5000"
```

## Troubleshooting

### No experiments showing in UI
- Ensure you're in the same directory where `./mlruns` exists
- Or specify path: `mlflow ui --backend-store-uri ./mlruns`

### MLflow not logging
- Check `mlflow.enabled=true` in config
- Look for error messages in console output
- Run test script: `uv run python test_mlflow_integration.py`

### System metrics not appearing
- System metrics may take 1-2 minutes to populate
- Check "System Metrics" tab in MLflow UI
- Ensure `system_metrics_interval` is set (default: 1)

## Additional Documentation

For more details, see:
- `MLFLOW_LOGGING.md` - Comprehensive documentation
- `MLFLOW_IMPLEMENTATION_SUMMARY.md` - Technical implementation details

## Support

If you encounter issues:
1. Run the test script: `uv run python test_mlflow_integration.py`
2. Check console output for error messages
3. Verify `./mlruns` directory exists and is writable
4. Check MLflow documentation: https://www.mlflow.org/docs/latest/

