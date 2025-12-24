mod cli;
mod config;
mod error;
mod image_io;
mod logging;

use clap::Parser;
use tracing::info;

use cli::{Cli, Commands, CropTarget, ModelAction};
use config::Config;
use error::IcarusError;

fn main() {
    let exit_code = match run() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            error::exit_code(&e)
        }
    };

    std::process::exit(exit_code);
}

fn run() -> Result<(), IcarusError> {
    let cli = Cli::parse();

    cli.validate().map_err(|e| IcarusError::InvalidConfig(e))?;

    let mut config = Config::load(cli.config.as_deref())?;
    apply_cli_overrides(&mut config, &cli);
    config.validate()?;
    config.ensure_paths()?;

    logging::init_logging(cli.verbose, cli.quiet, &config.general.log_level);

    info!("Icarus - AI-powered photo cropping tool");
    info!("Version: {}", env!("CARGO_PKG_VERSION"));

    match &cli.command {
        Commands::Crop { target } => handle_crop(target, &cli, &config),
        Commands::Models { action } => handle_models(action, &cli, &config),
    }
}

fn handle_crop(target: &CropTarget, cli: &Cli, config: &Config) -> Result<(), IcarusError> {
    match target {
        CropTarget::Image { path } => {
            info!("Cropping single image: {}", path.display());

            if cli.dry_run {
                info!("[DRY RUN] Would process: {}", path.display());
            }

            println!("Not implemented yet");
            println!("Would crop: {}", path.display());

            if let Some(aspect) = cli.aspect {
                println!("Target aspect: {:?}", aspect);
            } else if cli.all_aspects {
                println!("Target: All aspects");
            }

            if let Some(output) = &cli.output {
                println!("Output directory: {}", output.display());
            } else {
                println!("Output directory: {}", config.general.output_dir);
            }
        }
        CropTarget::Dir { path, recursive } => {
            info!(
                "Cropping directory: {} (recursive: {})",
                path.display(),
                recursive
            );

            if cli.dry_run {
                info!("[DRY RUN] Would process directory: {}", path.display());
            }

            println!("Not implemented yet");
            println!("Would crop directory: {}", path.display());
            println!("Recursive: {}", recursive);
        }
    }

    Ok(())
}

fn handle_models(action: &ModelAction, _cli: &Cli, config: &Config) -> Result<(), IcarusError> {
    match action {
        ModelAction::List => {
            info!("Listing available models");
            println!("Not implemented yet");
            println!("Models directory: {}", config.paths.models_dir);
            println!("\nAvailable models:");
            println!("  (none cached yet)");
        }
        ModelAction::Download { name } => {
            info!("Downloading model: {}", name);
            println!("Not implemented yet");
            println!("Would download model: {}", name);
            println!("To: {}", config.paths.models_dir);
        }
    }

    Ok(())
}

fn apply_cli_overrides(config: &mut Config, cli: &Cli) {
    if let Some(output) = &cli.output {
        config.general.output_dir = output.to_string_lossy().to_string();
    }

    if let Some(format) = cli.format {
        config.cropping.default_format = format.as_str().to_string();
    }

    if let Some(quality) = cli.quality {
        config.cropping.default_quality = quality;
    }
}
