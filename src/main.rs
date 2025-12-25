mod cli;
mod config;
mod cropper;
mod detection;
mod error;
mod image_io;
mod logging;

use clap::Parser;
use tracing::{info, warn};

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

/// Determine which aspects are valid for the source image and request
///
/// Rules:
/// 1. Portrait source (h > w) can ONLY generate: Portrait (9:16), Mobile (9:20)
/// 2. Landscape source (w >= h) can generate all: Portrait, Mobile, Landscape
/// 3. For each aspect: must meet exact ratio + minimum dimensions
fn get_valid_aspects_for_source(
    source_width: u32,
    source_height: u32,
    requested_aspects: &[cli::AspectRatio],
    aspect_config: &config::AspectsConfig,
    crop_config: &cropper::CropConfig,
) -> Vec<cli::AspectRatio> {
    let is_portrait_source = source_height > source_width;

    // Filter by source orientation
    let orientation_filtered: Vec<cli::AspectRatio> = if is_portrait_source {
        // Portrait source: only portrait and mobile
        requested_aspects
            .iter()
            .filter(|a| matches!(**a, cli::AspectRatio::Portrait | cli::AspectRatio::Mobile))
            .copied()
            .collect()
    } else {
        // Landscape source: all aspects allowed
        requested_aspects.to_vec()
    };

    // Validate each aspect meets dimension requirements
    let mut valid_aspects = Vec::new();

    for aspect in orientation_filtered {
        let _ratio_w_h = aspect_config.get_aspect_ratio(aspect);
        let can_achieve = match aspect {
            cli::AspectRatio::Portrait => {
                // Portrait: height must be >= min_height_portrait
                let required_height = crop_config.min_height_portrait;
                source_height >= required_height
            }
            cli::AspectRatio::Mobile => {
                // Mobile: height must be >= min_height_portrait (same as portrait)
                let required_height = crop_config.min_height_portrait;
                source_height >= required_height
            }
            cli::AspectRatio::Landscape => {
                // Landscape: width must be >= min_width_landscape
                let required_width = crop_config.min_width_landscape;
                source_width >= required_width
            }
        };

        if can_achieve {
            valid_aspects.push(aspect);
        } else {
            warn!(
                "Cannot generate {:?} crop: {} source too small",
                aspect,
                if is_portrait_source {
                    "portrait"
                } else {
                    "landscape"
                }
            );
        }
    }

    valid_aspects
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
                return Ok(());
            }

            // Load image
            let image = image_io::load_image(path)?;
            info!("Loaded image: {}x{}", image.width(), image.height());

            // Create crop config to get quality validation settings
            let crop_config = cropper::CropConfig {
                min_dimension_ratio: config.cropping.min_dimension_ratio,
                min_height_portrait: config.cropping.min_height_portrait,
                min_width_landscape: config.cropping.min_width_landscape,
            };

            // Validate image quality before running detection
            if let Err(reason) =
                cropper::validate_image_quality(image.width(), image.height(), &crop_config)
            {
                warn!("Cannot process image: {}", reason);
                return Err(IcarusError::ImageIo(reason));
            }

            // Initialize detector
            let model_path = std::path::PathBuf::from(&config.model.model_path);
            let detector =
                detection::OnnxDetector::new(&model_path, config.model.confidence_threshold)?;

            // Run detection
            let detection = detector.detect(&image)?;
            info!(
                "Detection result: center=({:.1}, {:.1}), confidence={:.3}, has_person={}",
                detection.center_x, detection.center_y, detection.confidence, detection.has_person
            );

            // Determine which aspects to process
            let requested_aspects = if cli.all_aspects {
                vec![
                    cli::AspectRatio::Portrait,
                    cli::AspectRatio::Landscape,
                    cli::AspectRatio::Mobile,
                ]
            } else {
                vec![cli.aspect.unwrap_or(cli::AspectRatio::Landscape)]
            };

            // Validate which aspects are achievable for this source image
            let aspects_to_process = get_valid_aspects_for_source(
                image.width(),
                image.height(),
                &requested_aspects,
                &config.aspects,
                &crop_config,
            );

            // If no valid aspects, skip
            if aspects_to_process.is_empty() {
                warn!("No valid aspect ratios for image: {}", path.display());
                return Ok(());
            }

            // Determine output directory
            let output_dir = cli
                .output
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| config.general.output_dir.clone());

            std::fs::create_dir_all(&output_dir)
                .map_err(|e| IcarusError::ImageIo(format!("Failed to create output dir: {}", e)))?;

            // Process each aspect ratio
            for aspect in aspects_to_process {
                let (ratio_w, ratio_h) = config.aspects.get_aspect_ratio(aspect);

                // Calculate crop box for this aspect
                let crop_box = cropper::calculate_max_aspect_crop(
                    image.width(),
                    image.height(),
                    detection.center_x,
                    detection.center_y,
                    ratio_w,
                    ratio_h,
                );

                // Perform the crop
                let cropped = image::imageops::crop(
                    &mut image.to_rgba8(),
                    crop_box.x,
                    crop_box.y,
                    crop_box.width,
                    crop_box.height,
                )
                .to_image();
                let cropped = image::DynamicImage::ImageRgba8(cropped);

                info!(
                    "Cropped image for aspect {:?}: {}x{}",
                    aspect,
                    cropped.width(),
                    cropped.height()
                );

                // Determine output directory (nested if --all-aspects)
                let aspect_dir = if cli.all_aspects {
                    std::path::PathBuf::from(&output_dir)
                        .join(format!("{:?}", aspect).to_lowercase())
                } else {
                    std::path::PathBuf::from(&output_dir)
                };

                std::fs::create_dir_all(&aspect_dir).map_err(|e| {
                    IcarusError::ImageIo(format!("Failed to create output dir: {}", e))
                })?;

                // Prepare output filename (no aspect suffix when using nested dirs)
                let output_filename = path
                    .file_stem()
                    .ok_or_else(|| IcarusError::ImageIo("Invalid file path".to_string()))?;
                let output_ext = path
                    .extension()
                    .ok_or_else(|| IcarusError::ImageIo("No file extension".to_string()))?;

                let output_filename_str = format!(
                    "{}.{}",
                    output_filename.to_string_lossy(),
                    output_ext.to_string_lossy()
                );

                let output_path = aspect_dir.join(&output_filename_str);

                let format = parse_format(&config.cropping.default_format)?;
                image_io::save_image(
                    &cropped,
                    &output_path,
                    format,
                    config.cropping.default_quality,
                )?;

                info!("Saved cropped image to: {}", output_path.display());
                println!("✓ Cropped and saved to: {}", output_path.display());
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
                return Ok(());
            }

            // Initialize detector once
            let model_path = std::path::PathBuf::from(&config.model.model_path);
            let detector =
                detection::OnnxDetector::new(&model_path, config.model.confidence_threshold)?;

            let crop_config = cropper::CropConfig {
                min_dimension_ratio: config.cropping.min_dimension_ratio,
                min_height_portrait: config.cropping.min_height_portrait,
                min_width_landscape: config.cropping.min_width_landscape,
            };

            process_directory(path, *recursive, &detector, &crop_config, cli, config)?;
        }
    }

    Ok(())
}

fn process_directory(
    dir: &std::path::Path,
    recursive: bool,
    detector: &detection::OnnxDetector,
    crop_config: &cropper::CropConfig,
    cli: &Cli,
    config: &Config,
) -> Result<(), IcarusError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| IcarusError::ImageIo(format!("Failed to read directory: {}", e)))?;

    for entry in entries {
        let entry =
            entry.map_err(|e| IcarusError::ImageIo(format!("Failed to read dir entry: {}", e)))?;
        let path = entry.path();

        if path.is_dir() && recursive {
            process_directory(&path, recursive, detector, crop_config, cli, config)?;
        } else if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp" | "tiff") {
                    match process_single_file(&path, detector, crop_config, cli, config) {
                        Ok(()) => {
                            info!("Successfully processed: {}", path.display());
                        }
                        Err(e) => {
                            info!("Error processing {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn process_single_file(
    path: &std::path::Path,
    detector: &detection::OnnxDetector,
    crop_config: &cropper::CropConfig,
    cli: &Cli,
    config: &Config,
) -> Result<(), IcarusError> {
    let image = image_io::load_image(path)?;

    // Validate image quality before running detection
    if let Err(reason) = cropper::validate_image_quality(image.width(), image.height(), crop_config)
    {
        warn!("Skipping {}: {}", path.display(), reason);
        return Ok(()); // Skip this image but don't fail the entire operation
    }

    let detection = detector.detect(&image)?;

    // Determine which aspects to process
    let requested_aspects = if cli.all_aspects {
        vec![
            cli::AspectRatio::Portrait,
            cli::AspectRatio::Landscape,
            cli::AspectRatio::Mobile,
        ]
    } else {
        vec![cli.aspect.unwrap_or(cli::AspectRatio::Landscape)]
    };

    // Validate which aspects are achievable for this source image
    let aspects_to_process = get_valid_aspects_for_source(
        image.width(),
        image.height(),
        &requested_aspects,
        &config.aspects,
        crop_config,
    );

    // If no valid aspects, skip this image
    if aspects_to_process.is_empty() {
        warn!("No valid aspect ratios for image: {}", path.display());
        return Ok(());
    }

    let output_dir = cli
        .output
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| config.general.output_dir.clone());

    std::fs::create_dir_all(&output_dir)
        .map_err(|e| IcarusError::ImageIo(format!("Failed to create output dir: {}", e)))?;

    // Process each aspect ratio
    for aspect in aspects_to_process {
        let (ratio_w, ratio_h) = config.aspects.get_aspect_ratio(aspect);

        // Calculate crop box for this aspect
        let crop_box = cropper::calculate_max_aspect_crop(
            image.width(),
            image.height(),
            detection.center_x,
            detection.center_y,
            ratio_w,
            ratio_h,
        );

        // Perform the crop
        let cropped = image::imageops::crop(
            &mut image.to_rgba8(),
            crop_box.x,
            crop_box.y,
            crop_box.width,
            crop_box.height,
        )
        .to_image();
        let cropped = image::DynamicImage::ImageRgba8(cropped);

        // Determine output directory (nested if --all-aspects)
        let aspect_dir = if cli.all_aspects {
            std::path::PathBuf::from(&output_dir).join(format!("{:?}", aspect).to_lowercase())
        } else {
            std::path::PathBuf::from(&output_dir)
        };

        std::fs::create_dir_all(&aspect_dir)
            .map_err(|e| IcarusError::ImageIo(format!("Failed to create output dir: {}", e)))?;

        // Prepare output filename (no aspect suffix when using nested dirs)
        let output_filename = path
            .file_stem()
            .ok_or_else(|| IcarusError::ImageIo("Invalid file path".to_string()))?;
        let output_ext = path
            .extension()
            .ok_or_else(|| IcarusError::ImageIo("No file extension".to_string()))?;

        let output_filename_str = format!(
            "{}.{}",
            output_filename.to_string_lossy(),
            output_ext.to_string_lossy()
        );

        let output_path = aspect_dir.join(&output_filename_str);

        let format = parse_format(&config.cropping.default_format)?;
        image_io::save_image(
            &cropped,
            &output_path,
            format,
            config.cropping.default_quality,
        )?;
    }

    Ok(())
}

fn parse_format(format_str: &str) -> Result<image::ImageFormat, IcarusError> {
    match format_str.to_lowercase().as_str() {
        "jpeg" | "jpg" => Ok(image::ImageFormat::Jpeg),
        "png" => Ok(image::ImageFormat::Png),
        "webp" => Ok(image::ImageFormat::WebP),
        "tiff" => Ok(image::ImageFormat::Tiff),
        _ => Err(IcarusError::InvalidFormat(format!(
            "Unsupported format: {}. Supported: jpeg, png, webp, tiff",
            format_str
        ))),
    }
}

fn handle_models(action: &ModelAction, _cli: &Cli, config: &Config) -> Result<(), IcarusError> {
    match action {
        ModelAction::List => {
            info!("Listing available models");
            println!("Models directory: {}", config.paths.models_dir);
            println!("\nAvailable models:");

            let models_dir = std::path::Path::new(&config.paths.models_dir);
            if !models_dir.exists() {
                println!("  (directory not found)");
                return Ok(());
            }

            let mut found_any = false;
            if let Ok(entries) = std::fs::read_dir(models_dir) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        let path = entry.path();
                        if path.is_dir() {
                            if let Some(name) = path.file_name() {
                                let model_file = path.join("model.onnx");
                                if model_file.exists() {
                                    if let Ok(metadata) = std::fs::metadata(&model_file) {
                                        let size_mb = metadata.len() as f32 / (1024.0 * 1024.0);
                                        println!(
                                            "  {} ({:.1} MB)",
                                            name.to_string_lossy(),
                                            size_mb
                                        );
                                        found_any = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !found_any {
                println!("  (none cached yet)");
                println!("\nRun 'task download-models' to download models from HuggingFace");
            }
        }
        ModelAction::Download { name } => {
            info!("Downloading model: {}", name);
            println!("Model download not yet implemented");
            println!("Instead, run: task download-models");
            println!(
                "This will download models from HuggingFace to: {}",
                config.paths.models_dir
            );
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
