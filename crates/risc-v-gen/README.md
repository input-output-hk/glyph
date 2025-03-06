# RISC-V Code Generator

This crate provides utilities for generating RISC-V assembly code.

## Features

- Support for all RISC-V base integer instructions (RV32I)
- Support for multiplication and division instructions (RV32M)
- Support for pseudo-instructions
- Support for assembly directives
- Error handling for invalid instructions

## Usage

```rust
use risc_v_gen::{CodeGenerator, Instruction, Register};

// Create a new code generator
let mut gen = CodeGenerator::new();

// Add instructions
gen.add_instruction(Instruction::Label("main".to_string()));
gen.add_instruction(Instruction::Li(Register::A0, 42));
gen.add_instruction(Instruction::Li(Register::A1, 58));
gen.add_instruction(Instruction::Add(Register::A2, Register::A0, Register::A1));

// Generate assembly code
let asm = gen.generate();
println!("{}", asm);
```

## Supported Instructions

### R-type Instructions

- `Add(rd, rs1, rs2)`: Add
- `Sub(rd, rs1, rs2)`: Subtract
- `And(rd, rs1, rs2)`: Bitwise AND
- `Or(rd, rs1, rs2)`: Bitwise OR
- `Xor(rd, rs1, rs2)`: Bitwise XOR
- `Slt(rd, rs1, rs2)`: Set less than
- `Sltu(rd, rs1, rs2)`: Set less than unsigned
- `Sll(rd, rs1, rs2)`: Shift left logical
- `Srl(rd, rs1, rs2)`: Shift right logical
- `Sra(rd, rs1, rs2)`: Shift right arithmetic
- `Mul(rd, rs1, rs2)`: Multiply
- `Div(rd, rs1, rs2)`: Divide
- `Rem(rd, rs1, rs2)`: Remainder

### I-type Instructions

- `Addi(rd, rs1, imm)`: Add immediate
- `Andi(rd, rs1, imm)`: Bitwise AND immediate
- `Ori(rd, rs1, imm)`: Bitwise OR immediate
- `Xori(rd, rs1, imm)`: Bitwise XOR immediate
- `Slti(rd, rs1, imm)`: Set less than immediate
- `Sltiu(rd, rs1, imm)`: Set less than immediate unsigned
- `Slli(rd, rs1, imm)`: Shift left logical immediate
- `Srli(rd, rs1, imm)`: Shift right logical immediate
- `Srai(rd, rs1, imm)`: Shift right arithmetic immediate
- `Lw(rd, offset, rs1)`: Load word
- `Lh(rd, offset, rs1)`: Load halfword
- `Lb(rd, offset, rs1)`: Load byte
- `Lhu(rd, offset, rs1)`: Load halfword unsigned
- `Lbu(rd, offset, rs1)`: Load byte unsigned
- `Jalr(rd, rs1, imm)`: Jump and link register

### S-type Instructions

- `Sw(rs2, offset, rs1)`: Store word
- `Sh(rs2, offset, rs1)`: Store halfword
- `Sb(rs2, offset, rs1)`: Store byte

### B-type Instructions

- `Beq(rs1, rs2, label)`: Branch if equal
- `Bne(rs1, rs2, label)`: Branch if not equal
- `Blt(rs1, rs2, label)`: Branch if less than
- `Bge(rs1, rs2, label)`: Branch if greater than or equal
- `Bltu(rs1, rs2, label)`: Branch if less than unsigned
- `Bgeu(rs1, rs2, label)`: Branch if greater than or equal unsigned

### U-type Instructions

- `Lui(rd, imm)`: Load upper immediate
- `Auipc(rd, imm)`: Add upper immediate to PC

### J-type Instructions

- `Jal(rd, label)`: Jump and link

### Pseudo-instructions

- `Li(rd, imm)`: Load immediate
- `La(rd, symbol)`: Load address
- `Mv(rd, rs)`: Move
- `Not(rd, rs)`: Bitwise NOT
- `Neg(rd, rs)`: Negate
- `Seqz(rd, rs)`: Set if equal to zero
- `Snez(rd, rs)`: Set if not equal to zero
- `Nop`: No operation

### Directives

- `Label(label)`: Define a label
- `Global(symbol)`: Make a symbol global
- `Section(section)`: Define a section
- `Align(align)`: Align to a power of 2
- `Word(value)`: Define a word
- `Byte(value)`: Define a byte
- `Ascii(string)`: Define an ASCII string
- `Asciiz(string)`: Define a null-terminated ASCII string
- `Space(size)`: Reserve space

### Comments

- `Comment(comment)`: Add a comment 