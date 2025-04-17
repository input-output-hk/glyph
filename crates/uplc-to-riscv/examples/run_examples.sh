#!/bin/bash

# Build the project
cargo build --release

# Create output directory
mkdir -p examples/output

# Run the simple example
echo "Running simple example..."
./target/release/uplc-to-risc-cli compile \
  --input examples/simple.uplc \
  --output examples/output/simple.s

echo "Evaluating simple example..."
./target/release/uplc-to-risc-cli evaluate \
  --input examples/simple.uplc

# Run the factorial example
echo "Running factorial example..."
./target/release/uplc-to-risc-cli compile \
  --input examples/factorial.uplc \
  --output examples/output/factorial.s \
  --mode optimize \
  --optimize aggressive

echo "Evaluating factorial example..."
./target/release/uplc-to-risc-cli evaluate \
  --input examples/factorial.uplc

echo "Done!" 