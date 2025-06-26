/// Build validators from a `plutus.json` file
#[derive(clap::Args)]
pub struct Args {}

impl Args {
    pub async fn exec(self) -> miette::Result<()> {
        todo!("implement build")
    }
}
