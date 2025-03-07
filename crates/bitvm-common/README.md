# BitVM Common

BitVM Common is a shared utility library for integrating with BitVMX, the verification protocol for Bitcoin transactions. It centralizes functionality that was previously duplicated across the `uplc-to-riscv` and `risc-v-gen` crates.

## Overview

This crate provides common utilities for:

- Memory segment management and validation
- Instruction handling and metadata generation
- Execution trace generation and formatting
- Verification and dispute resolution support

## Components

### Memory Utilities

The `memory` module provides utilities for working with memory segments in BitVMX, including:

- Memory segment types (read-only, read-write)
- Memory alignment checks
- Memory access validation
- Segment boundary constants

```rust
use bitvm_common::memory::{is_valid_memory_operation, MemorySegmentType};

fn check_memory_access(addr: u32, is_write: bool) -> Result<(), String> {
    is_valid_memory_operation(addr, is_write, 4)
}
```

### Instruction Utilities

The `instruction` module provides utilities for working with RISC-V instructions in BitVMX, including:

- BitVMX instruction format with metadata
- Register name parsing
- Immediate value parsing
- Instruction classification (jump, branch, load, store)

```rust
use bitvm_common::instruction::{BitVMXInstruction, parse_register_name};
use riscv_decode::Instruction::*;

let jal_instr = BitVMXInstruction::new(Jal(1, 0), 0x1000);
assert!(jal_instr.is_jump());
```

### Execution Trace Utilities

The `trace` module provides types and functions for generating and working with BitVMX execution traces, including:

- Execution step representation
- Execution trace generation
- Hash chain computation
- BitVMX-CPU trace conversion

```rust
use bitvm_common::trace::{ExecutionTrace, ExecutionStep};

let mut trace = ExecutionTrace::new();
let step = ExecutionStep::new(0x1000, 0x1000, 0x12345678, 0x1004, 0);
trace.add_step(step);
```

### Verification Utilities

The `verification` module provides utilities for BitVMX's verification protocol, including:

- Initial hash generation
- Step hash computation
- Hash chain validation

```rust
use bitvm_common::verification::{generate_initial_hash, generate_step_hash};

let initial_hash = generate_initial_hash();
let step_data = vec![0u8, 1u8, 2u8, 3u8];
let step_hash = generate_step_hash(&initial_hash, &step_data);
```

## Integration with BitVMX-CPU

This crate provides utilities for converting between the BitVMX-CPU trace format and our internal representation:

```rust
use bitvm_common::trace::convert_from_bitvm_cpu_trace;

let cpu_trace = /* ... */;
let our_trace = convert_from_bitvm_cpu_trace(&cpu_trace);
```

## For Contributors

The BitVM Common crate is designed to reduce code duplication and provide a consistent interface for BitVMX integration. When adding new functionality, consider whether it belongs in this crate if it:

1. Is used by multiple crates
2. Relates to BitVMX compatibility
3. Provides core functionality for trace generation or verification

### Testing

Run the tests with:

```bash
cargo test -p bitvm-common
``` 