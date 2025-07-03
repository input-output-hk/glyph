use clap::Parser;

mod build;
mod compile;

pub const BANNER: &str = color_print::cstr! {
r#"

  ___  __    _  _  ____  _  _
 / __)(  )  ( \/ )(  _ \/ )( \        A modern toolkit for
( (_ \/ (_/\ )  /  ) __/) __ (      <yellow><bold>Bitcoin</bold></yellow> Smart Contracts.
 \___/\____/(__/  (__)  \_)(_/

 <magenta>repo:</magenta> <blue><italic><dim>https://github.com/input-output-hk/glyph</dim></italic></blue>
 <magenta>docs:</magenta> <blue><italic><dim>https://glyph-lang.org</dim></italic></blue>
 <magenta>chat:</magenta> <blue><italic><dim>https://discord.gg/INVITE_CODE</dim></italic></blue>
 <magenta>contribute:</magenta> <blue><italic><dim>https://github.com/input-output-hk/glyph/blob/main/CONTRIBUTING.md</dim></italic></blue>"#
};

#[derive(Parser)]
#[clap(version, about, long_about = Some(BANNER))]
#[clap(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

impl Default for Cli {
    fn default() -> Self {
        Self::parse()
    }
}

impl Cli {
    pub async fn exec(self) -> miette::Result<()> {
        self.cmd.exec().await
    }
}

#[derive(clap::Subcommand)]
pub enum Cmd {
    Build(build::Args),
    Compile(compile::Args),
}

impl Cmd {
    pub async fn exec(self) -> miette::Result<()> {
        match self {
            Cmd::Build(args) => args.exec().await,
            Cmd::Compile(args) => args.exec().await,
        }
    }
}
