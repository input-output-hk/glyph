mod cmd;

#[tokio::main]
async fn main() -> miette::Result<()> {
    let cli = cmd::Cli::default();

    cli.exec().await
}
