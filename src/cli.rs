use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "icarus")]
#[command(about = "AI-powered photo cropping tool", long_about = None)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    #[arg(long, help = "Custom config file path")]
    pub config: Option<PathBuf>,

    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity (-v, -vv, -vvv)")]
    pub verbose: u8,

    #[arg(short, long, help = "Suppress non-error output")]
    pub quiet: bool,

    #[arg(short, long, help = "Output directory")]
    pub output: Option<PathBuf>,

    #[arg(long, help = "Preview without writing files")]
    pub dry_run: bool,

    #[arg(long, help = "Generate debug visualizations")]
    pub visualize: bool,

    #[arg(long, help = "Draw detection bounding boxes on output")]
    pub r#box: bool,

    #[arg(long, help = "Target aspect (portrait|mobile|landscape)")]
    pub aspect: Option<AspectRatio>,

    #[arg(long, help = "Generate all valid aspects")]
    pub all_aspects: bool,

    #[arg(long, value_name = "FORMAT", help = "Output format (jpeg|png|webp)")]
    pub format: Option<ImageFormat>,

    #[arg(long, value_name = "QUALITY", help = "JPEG quality 1-100")]
    pub quality: Option<u8>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    #[command(about = "Crop images")]
    Crop {
        #[command(subcommand)]
        target: CropTarget,
    },
    #[command(about = "Manage ONNX models")]
    Models {
        #[command(subcommand)]
        action: ModelAction,
    },
}

#[derive(Subcommand, Debug)]
pub enum CropTarget {
    #[command(about = "Crop a single image")]
    Image {
        #[arg(help = "Path to the image file")]
        path: PathBuf,
    },
    #[command(about = "Crop all images in a directory")]
    Dir {
        #[arg(help = "Path to the directory")]
        path: PathBuf,

        #[arg(short, long, help = "Process directories recursively")]
        recursive: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum ModelAction {
    #[command(about = "List available/cached models")]
    List,
    #[command(about = "Download an ONNX model")]
    Download {
        #[arg(help = "Model name to download")]
        name: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectRatio {
    Portrait,
    Mobile,
    Landscape,
}

impl std::str::FromStr for AspectRatio {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "portrait" => Ok(AspectRatio::Portrait),
            "mobile" => Ok(AspectRatio::Mobile),
            "landscape" => Ok(AspectRatio::Landscape),
            _ => Err(format!(
                "Invalid aspect ratio '{}'. Must be one of: portrait, mobile, landscape",
                s
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Webp,
}

impl std::str::FromStr for ImageFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "jpeg" | "jpg" => Ok(ImageFormat::Jpeg),
            "png" => Ok(ImageFormat::Png),
            "webp" => Ok(ImageFormat::Webp),
            _ => Err(format!(
                "Invalid image format '{}'. Must be one of: jpeg, png, webp",
                s
            )),
        }
    }
}

impl ImageFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "jpeg",
            ImageFormat::Png => "png",
            ImageFormat::Webp => "webp",
        }
    }
}

impl Cli {
    pub fn validate(&self) -> Result<(), String> {
        if self.aspect.is_some() && self.all_aspects {
            return Err("Cannot specify both --aspect and --all-aspects".to_string());
        }

        if let Some(quality) = self.quality {
            if !(1..=100).contains(&quality) {
                return Err(format!(
                    "Quality must be between 1 and 100, got {}",
                    quality
                ));
            }
        }

        if self.verbose > 0 && self.quiet {
            return Err("Cannot specify both --verbose and --quiet".to_string());
        }

        Ok(())
    }
}
