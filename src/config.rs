use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{IcarusError, Result};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub cropping: CroppingConfig,
    #[serde(default)]
    pub aspects: AspectsConfig,
    #[serde(default)]
    pub paths: PathsConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GeneralConfig {
    #[serde(default = "default_output_dir")]
    pub output_dir: String,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_model_path")]
    pub model_path: String,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    #[serde(default = "default_nms_threshold")]
    pub nms_threshold: f32,
    #[serde(default = "default_input_size")]
    pub input_size: [u32; 2],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CroppingConfig {
    #[serde(default = "default_center_weight")]
    pub center_weight: String,
    #[serde(default = "default_min_dimension_ratio")]
    pub min_dimension_ratio: f32,
    #[serde(default = "default_format")]
    pub default_format: String,
    #[serde(default = "default_quality")]
    pub default_quality: u8,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AspectsConfig {
    #[serde(default = "default_portrait_width")]
    pub portrait_width: u32,
    #[serde(default = "default_portrait_height")]
    pub portrait_height: u32,
    #[serde(default = "default_mobile_width")]
    pub mobile_width: u32,
    #[serde(default = "default_mobile_height")]
    pub mobile_height: u32,
    #[serde(default = "default_landscape_width")]
    pub landscape_width: u32,
    #[serde(default = "default_landscape_height")]
    pub landscape_height: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PathsConfig {
    #[serde(default = "default_models_dir")]
    pub models_dir: String,
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
}

fn default_output_dir() -> String {
    "./output".to_string()
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_provider() -> String {
    "cpu".to_string()
}

fn default_model_path() -> String {
    "./models/yolov8n.onnx".to_string()
}

fn default_confidence_threshold() -> f32 {
    0.5
}

fn default_nms_threshold() -> f32 {
    0.45
}

fn default_input_size() -> [u32; 2] {
    [640, 640]
}

fn default_center_weight() -> String {
    "area".to_string()
}

fn default_min_dimension_ratio() -> f32 {
    0.75
}

fn default_format() -> String {
    "jpeg".to_string()
}

fn default_quality() -> u8 {
    90
}

fn default_portrait_width() -> u32 {
    9
}

fn default_portrait_height() -> u32 {
    16
}

fn default_mobile_width() -> u32 {
    9
}

fn default_mobile_height() -> u32 {
    20
}

fn default_landscape_width() -> u32 {
    21
}

fn default_landscape_height() -> u32 {
    9
}

fn default_models_dir() -> String {
    "./models".to_string()
}

fn default_cache_dir() -> String {
    "./cache".to_string()
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            output_dir: default_output_dir(),
            log_level: default_log_level(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model_path: default_model_path(),
            confidence_threshold: default_confidence_threshold(),
            nms_threshold: default_nms_threshold(),
            input_size: default_input_size(),
        }
    }
}

impl Default for CroppingConfig {
    fn default() -> Self {
        Self {
            center_weight: default_center_weight(),
            min_dimension_ratio: default_min_dimension_ratio(),
            default_format: default_format(),
            default_quality: default_quality(),
        }
    }
}

impl Default for AspectsConfig {
    fn default() -> Self {
        Self {
            portrait_width: default_portrait_width(),
            portrait_height: default_portrait_height(),
            mobile_width: default_mobile_width(),
            mobile_height: default_mobile_height(),
            landscape_width: default_landscape_width(),
            landscape_height: default_landscape_height(),
        }
    }
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            cache_dir: default_cache_dir(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            model: ModelConfig::default(),
            cropping: CroppingConfig::default(),
            aspects: AspectsConfig::default(),
            paths: PathsConfig::default(),
        }
    }
}

impl Config {
    pub fn load(config_path: Option<&Path>) -> Result<Self> {
        if let Some(path) = config_path {
            return Self::load_from_path(path);
        }

        if let Some(path) = Self::discover_config_file()? {
            Self::load_from_path(&path)
        } else {
            Ok(Config::default())
        }
    }

    fn load_from_path(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path).map_err(|e| {
            IcarusError::Config(format!(
                "Failed to read config file '{}': {}",
                path.display(),
                e
            ))
        })?;

        let config: Config = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    fn discover_config_file() -> Result<Option<PathBuf>> {
        let current_dir = env::current_dir().map_err(|e| {
            IcarusError::Config(format!(
                "Failed to determine current directory for config discovery: {}",
                e
            ))
        })?;

        let local_config = current_dir.join("config.toml");
        if local_config.exists() {
            return Ok(Some(local_config));
        }

        if let Some(home) = dirs::home_dir() {
            let user_config = home.join(".config/icarus/config.toml");
            if user_config.exists() {
                return Ok(Some(user_config));
            }
        }

        Ok(None)
    }

    pub fn validate(&self) -> Result<()> {
        if self.general.output_dir.trim().is_empty() {
            return Err(IcarusError::InvalidConfig(
                "output_dir cannot be empty".to_string(),
            ));
        }

        if self.paths.models_dir.trim().is_empty() {
            return Err(IcarusError::InvalidConfig(
                "models_dir cannot be empty".to_string(),
            ));
        }

        if self.paths.cache_dir.trim().is_empty() {
            return Err(IcarusError::InvalidConfig(
                "cache_dir cannot be empty".to_string(),
            ));
        }

        if !["trace", "debug", "info", "warn", "error"].contains(&self.general.log_level.as_str()) {
            return Err(IcarusError::InvalidConfig(format!(
                "Invalid log_level '{}'. Must be one of: trace, debug, info, warn, error",
                self.general.log_level
            )));
        }

        if !["area", "confidence", "equal"].contains(&self.cropping.center_weight.as_str()) {
            return Err(IcarusError::InvalidConfig(format!(
                "Invalid center_weight '{}'. Must be one of: area, confidence, equal",
                self.cropping.center_weight
            )));
        }

        if !(0.0..=1.0).contains(&self.model.confidence_threshold) {
            return Err(IcarusError::InvalidConfig(format!(
                "confidence_threshold must be between 0.0 and 1.0, got {}",
                self.model.confidence_threshold
            )));
        }

        if !(0.0..=1.0).contains(&self.model.nms_threshold) {
            return Err(IcarusError::InvalidConfig(format!(
                "nms_threshold must be between 0.0 and 1.0, got {}",
                self.model.nms_threshold
            )));
        }

        if !(0.0..=1.0).contains(&self.cropping.min_dimension_ratio) {
            return Err(IcarusError::InvalidConfig(format!(
                "min_dimension_ratio must be between 0.0 and 1.0, got {}",
                self.cropping.min_dimension_ratio
            )));
        }

        if self.aspects.portrait_width == 0 || self.aspects.portrait_height == 0 {
            return Err(IcarusError::InvalidConfig(
                "Portrait aspect dimensions must be positive integers".to_string(),
            ));
        }

        if self.aspects.mobile_width == 0 || self.aspects.mobile_height == 0 {
            return Err(IcarusError::InvalidConfig(
                "Mobile aspect dimensions must be positive integers".to_string(),
            ));
        }

        if self.aspects.landscape_width == 0 || self.aspects.landscape_height == 0 {
            return Err(IcarusError::InvalidConfig(
                "Landscape aspect dimensions must be positive integers".to_string(),
            ));
        }

        if !(1..=100).contains(&self.cropping.default_quality) {
            return Err(IcarusError::InvalidConfig(format!(
                "default_quality must be between 1 and 100, got {}",
                self.cropping.default_quality
            )));
        }

        Ok(())
    }

    pub fn ensure_paths(&self) -> Result<()> {
        ensure_dir(Path::new(&self.general.output_dir), "output directory")?;
        ensure_dir(Path::new(&self.paths.models_dir), "models directory")?;
        ensure_dir(Path::new(&self.paths.cache_dir), "cache directory")?;

        if let Some(parent) = Path::new(&self.model.model_path).parent() {
            if !parent.as_os_str().is_empty() {
                ensure_dir(parent, "model path directory")?;
            }
        }

        Ok(())
    }
}

fn ensure_dir(path: &Path, desc: &str) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(IcarusError::InvalidConfig(format!(
            "{} cannot be empty",
            desc
        )));
    }

    fs::create_dir_all(path).map_err(|e| {
        IcarusError::Config(format!(
            "Failed to create {} '{}': {}",
            desc,
            path.display(),
            e
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.general.log_level, "info");
        assert_eq!(config.model.confidence_threshold, 0.5);
        assert_eq!(config.cropping.center_weight, "area");
    }

    #[test]
    fn test_validate_invalid_log_level() {
        let mut config = Config::default();
        config.general.log_level = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_center_weight() {
        let mut config = Config::default();
        config.cropping.center_weight = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_threshold() {
        let mut config = Config::default();
        config.model.confidence_threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_quality() {
        let mut config = Config::default();
        config.cropping.default_quality = 101;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_empty_paths() {
        let mut config = Config::default();
        config.general.output_dir = "".to_string();
        assert!(config.validate().is_err());

        config.general.output_dir = "./output".to_string();
        config.paths.models_dir = "".to_string();
        assert!(config.validate().is_err());

        config.paths.models_dir = "./models".to_string();
        config.paths.cache_dir = "".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn ensure_paths_creates_directories() {
        let temp_dir = create_temp_dir();
        let output_dir = temp_dir.join("output");
        let models_dir = temp_dir.join("models");
        let cache_dir = temp_dir.join("cache");
        let model_path = models_dir.join("yolov8n.onnx");

        let mut config = Config::default();
        config.general.output_dir = output_dir.to_string_lossy().to_string();
        config.paths.models_dir = models_dir.to_string_lossy().to_string();
        config.paths.cache_dir = cache_dir.to_string_lossy().to_string();
        config.model.model_path = model_path.to_string_lossy().to_string();

        config.ensure_paths().expect("create directories");

        assert!(output_dir.exists());
        assert!(models_dir.exists());
        assert!(cache_dir.exists());

        fs::remove_dir_all(&temp_dir).expect("clean temp dir");
    }

    #[test]
    fn load_defaults_when_no_config_found() {
        let temp_dir = create_temp_dir();
        let original_home = env::var_os("HOME");
        let original_dir = env::current_dir().expect("current dir");

        env::set_var("HOME", temp_dir.display().to_string());
        env::set_current_dir(&temp_dir).expect("set cwd");

        let config = Config::load(None).expect("load default config");
        let default = Config::default();
        assert_eq!(config.general.output_dir, default.general.output_dir);

        if let Some(home) = original_home {
            env::set_var("HOME", home);
        } else {
            env::remove_var("HOME");
        }
        env::set_current_dir(&original_dir).expect("restore cwd");
        fs::remove_dir_all(&temp_dir).expect("clean temp dir");
    }

    fn create_temp_dir() -> PathBuf {
        let base = env::temp_dir();
        for attempt in 0..16 {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos();
            let path = base.join(format!("icarus-test-{}-{}", std::process::id(), stamp));
            if fs::create_dir(&path).is_ok() {
                return path;
            }
            if attempt == 15 {
                panic!("could not create temp dir");
            }
        }
        unreachable!();
    }
}
