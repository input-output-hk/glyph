use std::fs;
use std::path::Path;
use std::process;
use uplc_to_riscv::{Compiler};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: {} <input.uplc> <output.s> [options]", args[0]);
        eprintln!("Options:");
        eprintln!("  --bitvm-trace         Generate BitVMX execution trace");
        eprintln!("  --bitvm-verify        Generate BitVMX verification script");
        eprintln!("  --bitvm-emulate       Emulate using BitVMX emulator");
        process::exit(1);
    }
    
    let input_path = &args[1];
    let output_path = &args[2];
    
    // Parse options
    let mut generate_trace = false;
    let mut generate_verification = false;
    let mut emulate = false;
    
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--bitvm-trace" => {
                generate_trace = true;
                i += 1;
            }
            "--bitvm-verify" => {
                generate_verification = true;
                i += 1;
            }
            "--bitvm-emulate" => {
                emulate = true;
                i += 1;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                process::exit(1);
            }
        }
    }
    
    // Read input file
    let input = match fs::read_to_string(input_path) {
        Ok(content) => content,
        Err(err) => {
            eprintln!("Failed to read input file: {}", err);
            process::exit(1);
        }
    };
    
    // Create compiler
    let compiler = Compiler::new();
    
    // Compile with or without trace generation
    if generate_trace {
        // Compile with trace generation
        let (output, execution_trace) = match compiler.compile_with_trace(&input) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("Compilation failed: {}", err);
                process::exit(1);
            }
        };
        
        // Write output file
        if let Err(err) = fs::write(output_path, &output) {
            eprintln!("Failed to write output file: {}", err);
            process::exit(1);
        }
        
        println!("Successfully compiled {} to {}", input_path, output_path);
        
        // Generate trace file
        let trace_path = Path::new(output_path)
            .with_extension("trace.csv");
        
        // Convert execution trace to CSV format
        let trace_csv = execution_trace.to_csv();
        
        // Write trace file
        if let Err(err) = fs::write(&trace_path, trace_csv) {
            eprintln!("Failed to write trace file: {}", err);
            process::exit(1);
        }
        
        println!("BitVMX trace written to {}", trace_path.display());
        
        // Generate hash chain file if requested
        if generate_verification {
            let hash_chain_path = Path::new(output_path)
                .with_extension("hash_chain");
            
            // Get the hash chain
            let hash_chain = execution_trace.hash_chain();
            
            // Convert hash chain to hex strings
            let hash_chain_hex: Vec<String> = hash_chain.iter()
                .map(|hash| hash.iter().map(|b| format!("{:02x}", b)).collect::<String>())
                .collect();
            
            // Join with newlines
            let hash_chain_str = hash_chain_hex.join("\n");
            
            // Write hash chain file
            if let Err(err) = fs::write(&hash_chain_path, hash_chain_str) {
                eprintln!("Failed to write hash chain file: {}", err);
                process::exit(1);
            }
            
            println!("BitVMX hash chain written to {}", hash_chain_path.display());
        }
    } else {
        // Compile without trace generation
        let output = match compiler.compile(&input) {
            Ok(asm) => asm,
            Err(err) => {
                eprintln!("Compilation failed: {}", err);
                process::exit(1);
            }
        };
        
        // Write output file
        if let Err(err) = fs::write(output_path, &output) {
            eprintln!("Failed to write output file: {}", err);
            process::exit(1);
        }
        
        println!("Successfully compiled {} to {}", input_path, output_path);
    }
    
    // Generate BitVMX verification script if requested
    if generate_verification && !generate_trace {
        let script_path = Path::new(output_path)
            .with_extension("script");
        
        // TODO: Generate verification script
        println!("BitVMX verification script generation is not yet implemented");
        
        // For now, just create a placeholder script
        let script = "# BitVMX verification script placeholder\n";
        if let Err(err) = fs::write(&script_path, script) {
            eprintln!("Failed to write verification script: {}", err);
            process::exit(1);
        }
        
        println!("BitVMX verification script written to {}", script_path.display());
    }
    
    // Emulate using BitVMX emulator if requested
    if emulate {
        println!("BitVMX emulation is not yet implemented");
        
        // TODO: Run BitVMX emulator
        // This would involve calling the BitVMX emulator with the generated assembly
    }
} 