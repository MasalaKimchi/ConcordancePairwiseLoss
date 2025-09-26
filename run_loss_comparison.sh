#!/bin/bash

# Enhanced TorchSurv MNIST Loss Comparison Script
# Runs all loss types with batch size 64, 3 epochs, and 10% training data

echo "=========================================="
echo "Enhanced TorchSurv MNIST Loss Comparison"
echo "=========================================="
echo "Configuration:"
echo "  Batch Size: 64"
echo "  Epochs: 3"
echo "  Training Data: 10% (limit-train-batches=0.1)"
echo "  Loss Types: NLL, CPL, CPL(IPCW), CPL(IPCW batch)"
echo "=========================================="

# Create results directory
mkdir -p results

# Array of loss types to test
loss_types=("nll" "cpl" "cpl_ipcw" "cpl_ipcw_batch")

# Run each loss type
for loss_type in "${loss_types[@]}"; do
    echo ""
    echo "Running experiment: $loss_type"
    echo "----------------------------------------"
    
    # Run the experiment
    python train_torchsurv_mnist_enhanced.py \
        --loss-type "$loss_type" \
        --batch-size 256 \
        --epochs 5 \
        --limit-train-batches 0.1 \
        --output-dir results
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✅ $loss_type completed successfully"
    else
        echo "❌ $loss_type failed"
    fi
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: results/"
echo "=========================================="

# List the generated files
echo ""
echo "Generated files:"
ls -la results/*.json results/*.csv 2>/dev/null || echo "No result files found"
