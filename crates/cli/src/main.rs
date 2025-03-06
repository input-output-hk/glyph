use std::fs;
use std::path::PathBuf;
use clap::{Parser, Subcommand, ValueEnum};
use uplc_to_riscv::{Compiler, OptimizationLevel};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile UPLC code to RISC-V assembly with BitVMX compatibility
    Compile {
        /// Input UPLC file
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output RISC-V assembly file
        #[arg(short, long)]
        output: PathBuf,
        
        /// Optimization level
        #[arg(short = 'O', long, value_enum, default_value_t = Optimize::None)]
        optimize: Optimize,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Optimize {
    /// No optimization
    None,
    
    /// Default optimization level
    Default,
    
    /// Aggressive optimization
    Aggressive,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Compile { input, output, optimize } => {
            let uplc_code = fs::read_to_string(input)?;
            
            let optimization_level = match optimize {
                Optimize::None => OptimizationLevel::None,
                Optimize::Default => OptimizationLevel::Default,
                Optimize::Aggressive => OptimizationLevel::Maximum,
            };
            
            let compiler = Compiler::new()
                .with_optimization_level(optimization_level);
            
            let result = compiler.compile(&uplc_code)?;
            fs::write(output, result)?;
            
            println!("Successfully compiled {} to {}", input.display(), output.display());
        },
    }
    
    Ok(())
} 