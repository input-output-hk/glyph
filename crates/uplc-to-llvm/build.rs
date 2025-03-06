use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();
    
    // Print debug information
    println!("OUT_DIR = {:?}", env::var("OUT_DIR"));
    println!("TARGET = {:?}", env::var("TARGET"));
    println!("HOST = {:?}", env::var("HOST"));
    
    // Set up cc builder
    let mut builder = cc::Build::new();
    
    if target == "riscv64gc-unknown-none-elf" {
        // Configure RISC-V target
        println!("cargo:rustc-cfg=target_arch=\"riscv64\"");
        println!("cargo:rustc-cfg=target_os=\"none\"");
        
        // Force using the RISC-V compiler
        builder.cpp(false)  // Ensure C mode
               .compiler("riscv64-elf-gcc")
               .opt_level(2)
               .flag("-march=rv64gc")
               .flag("-mabi=lp64d")
               .flag("-ffreestanding")
               .flag("-nostdlib")
               .flag("-nostartfiles")
               .flag("-static")
               .host("riscv64gc-unknown-none-elf")
               .target("riscv64gc-unknown-none-elf")
               .no_default_flags(true)
               .warnings(false);
    } else {
        // Use native toolchain for development
        builder.cpp(false)
               .opt_level(2)
               .warnings(false);
    }
        
    // Compile runtime support
    builder
        .file("src/runtime/uplc_runtime.c")
        .compile("uplc_runtime");
        
    // Generate linker script
    std::fs::write(
        out_dir.join("link.ld"),
        include_bytes!("src/runtime/link.ld")
    ).unwrap();
    
    println!("cargo:rerun-if-changed=src/runtime/uplc_runtime.c");
    println!("cargo:rerun-if-changed=src/runtime/link.ld");
    println!("cargo:rerun-if-changed=build.rs");
} 