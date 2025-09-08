# Wandb Sweep Cleanup Guide

This guide provides comprehensive instructions for completely canceling and cleaning up wandb sweeps, including killing agents, clearing cache, and returning to a blank slate state.

## Quick Reference

### 1. Cancel the Sweep (Web Interface)
- Go to your wandb project dashboard
- Navigate to the "Sweeps" tab
- Find your sweep and click "Cancel" or "Stop"
- This prevents new runs from being scheduled


### 2. Cancel the Sweep (Command Line)
```bash
# Cancel a sweep to kill all running runs and stop running new runs
conda activate metabool
wandb sweep --cancel entity/project/sweep_ID

# Example for your current sweep:
conda activate metabool
wandb sweep --cancel marcello-barylli-growai/boolean_nca_cc/a2hzk5ua
```

### 3. Kill All Running Agents and Training Processes
```bash
# Find all running wandb agent processes
ps aux | grep "wandb agent"

# Find all running training processes
ps aux | grep -E "(python.*train|wandb)" | grep -v grep

# Kill all wandb agent processes
pkill -f "wandb agent"

# Kill all training processes (be careful - this kills ALL training processes)
pkill -f "python.*train"

# Or kill specific processes by PID (safer approach)
kill -9 <PID1> <PID2> <PID3> ...
```

### 4. Clear Wandb Cache
```bash
# Remove specific sweep cache directory
rm -rf wandb/sweep-<SWEEP_ID>/

# Remove all sweep cache directories
rm -rf wandb/sweep-*

# Remove all wandb run directories (optional - removes all local run data)
rm -rf wandb/run-*

# Remove wandb debug logs
rm -f wandb/debug*.log
```

### 5. Complete Clean Slate (Nuclear Option)
```bash
# Remove entire wandb directory
rm -rf wandb/

# This will force wandb to recreate everything from scratch
```

## Detailed Steps

### Step 1: Identify Running Sweeps, Agents, and Training Processes

First, check what's currently running:

```bash
# Check for running wandb agents
ps aux | grep "wandb agent"

# Check for running training processes
ps aux | grep -E "(python.*train|wandb)" | grep -v grep

# List wandb sweep directories
ls -la wandb/sweep-*

# Check wandb status
wandb status

# Check for background jobs in current terminal
jobs
```

### Step 2: Cancel the Sweep

**Option A: Web Interface (Recommended)**
1. Open your browser and go to [wandb.ai](https://wandb.ai)
2. Navigate to your project
3. Click on the "Sweeps" tab
4. Find your sweep in the list
5. Click the "Cancel" or "Stop" button

**Option B: Command Line**
```bash
# Cancel a specific sweep (replace with your sweep ID)
wandb sweep --cancel entity/project/sweep_ID

# Example for your current sweep:
wandb sweep --cancel marcello-barylli-growai/boolean_nca_cc/a2hzk5ua
```

### Step 3: Kill All Agents and Training Processes

```bash
# Method 1: Kill wandb agents by process name
pkill -f "wandb agent"

# Method 2: Kill all training processes (use with caution - kills ALL training)
pkill -f "python.*train"

# Method 3: Kill specific processes by PID (recommended for safety)
kill -9 <PID1> <PID2> <PID3>

# Method 4: Kill all python processes running wandb (use with extreme caution)
pkill -f "python.*wandb"

# Method 5: Kill processes for your user only (safer)
ps aux | grep -E "(python.*train|wandb)" | grep $USER | awk '{print $2}' | xargs kill -9
```

### Step 4: Clear Local Cache

```bash
# Remove sweep-specific cache
rm -rf wandb/sweep-<SWEEP_ID>/

# Remove all sweep caches
rm -rf wandb/sweep-*

# Remove run directories (optional - contains local run data)
rm -rf wandb/run-*

# Remove debug logs
rm -f wandb/debug*.log
```

### Step 5: Verify Clean State

```bash
# Check no agents are running
ps aux | grep "wandb agent"

# Check no training processes are running
ps aux | grep -E "(python.*train|wandb)" | grep -v grep

# Check wandb directory is clean
ls -la wandb/

# Check wandb status
wandb status

# Check for any background jobs
jobs
```

## Common Issues and Solutions

### Issue: "Sweep is not running" Error
**Cause**: Trying to connect to a canceled or non-existent sweep
**Solution**: 
1. Remove the old sweep cache: `rm -rf wandb/sweep-<OLD_SWEEP_ID>/`
2. Create a new sweep or use the correct sweep ID

### Issue: Agents Keep Restarting
**Cause**: Agents are being managed by a script or process manager
**Solution**:
1. Stop the script that launched the agents
2. Kill all agent processes: `pkill -f "wandb agent"`
3. Kill all training processes: `pkill -f "python.*train"`
4. Check for any background jobs: `jobs`

### Issue: Training Processes Still Running After Sweep Cancel
**Cause**: Training processes may continue running even after sweep cancellation
**Solution**:
1. Identify running training processes: `ps aux | grep -E "(python.*train|wandb)" | grep -v grep`
2. Kill specific training processes by PID: `kill -9 <PID1> <PID2> <PID3>`
3. Or kill all training processes for your user: `ps aux | grep -E "(python.*train|wandb)" | grep $USER | awk '{print $2}' | xargs kill -9`
4. Verify cleanup: `ps aux | grep -E "(python.*train|wandb)" | grep -v grep`

### Issue: Cache Persists After Cleanup
**Cause**: Wandb may cache information in other locations
**Solution**:
1. Complete nuclear cleanup: `rm -rf wandb/`
2. Restart your terminal session
3. Re-authenticate with wandb if needed: `wandb login`

## Prevention Tips

1. **Always use unique sweep IDs** when creating new sweeps
2. **Stop agents gracefully** before canceling sweeps when possible
3. **Keep track of running sweeps** in your project documentation
4. **Use environment variables** to manage sweep IDs in scripts

## Example Cleanup Script

Create a cleanup script for easy use:

```bash
#!/bin/bash
# cleanup_sweep.sh

SWEEP_ID=$1

if [ -z "$SWEEP_ID" ]; then
    echo "Usage: $0 <SWEEP_ID>"
    echo "Or use 'all' to clean everything"
    exit 1
fi

echo "Cleaning up sweep: $SWEEP_ID"

# Kill all agents and training processes
echo "Killing wandb agents..."
pkill -f "wandb agent"

echo "Killing training processes..."
pkill -f "python.*train"

# Remove sweep cache
if [ "$SWEEP_ID" = "all" ]; then
    echo "Removing all sweep caches..."
    rm -rf wandb/sweep-*
else
    echo "Removing sweep cache for: $SWEEP_ID"
    rm -rf wandb/sweep-$SWEEP_ID/
fi

# Remove debug logs
echo "Removing debug logs..."
rm -f wandb/debug*.log

echo "Cleanup complete!"
```

Make it executable and use:
```bash
chmod +x cleanup_sweep.sh
./cleanup_sweep.sh <SWEEP_ID>
./cleanup_sweep.sh all  # for complete cleanup
```

## Recovery After Cleanup

After a complete cleanup, you may need to:

1. **Re-authenticate**: `wandb login`
2. **Recreate sweep**: `wandb sweep sweep.yaml`
3. **Update sweep ID** in your scripts
4. **Restart agents**: `bash sweep.sh`

This guide ensures you can always return to a clean state when working with wandb sweeps.
