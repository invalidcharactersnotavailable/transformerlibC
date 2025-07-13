#!/bin/bash

echo "=== TransformerLibC Basic Test ==="

# Clean previous build
echo "Cleaning previous build..."
make clean

# Build the project
echo "Building project..."
if ! make; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"

# Test basic functionality
echo "Testing basic functionality..."
if ! ./transformer_test --test --verbose; then
    echo "Basic test failed!"
    exit 1
fi

echo "Basic test passed!"

# Test help
echo "Testing help..."
if ! ./transformer_test --help > /dev/null; then
    echo "Help test failed!"
    exit 1
fi

echo "Help test passed!"

# Test with different model sizes
echo "Testing different model configurations..."

# Small model
echo "Testing small model (2 layers, 256 dim)..."
if ! ./transformer_test --test --n_layers 2 --embed_dim 256 --n_heads 4 --verbose; then
    echo "Small model test failed!"
    exit 1
fi

# Medium model
echo "Testing medium model (4 layers, 512 dim)..."
if ! ./transformer_test --test --n_layers 4 --embed_dim 512 --n_heads 8 --verbose; then
    echo "Medium model test failed!"
    exit 1
fi

echo "All tests passed!"
echo "=== Test completed successfully ==="