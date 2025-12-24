use std::io;

#[derive(Debug, thiserror::Error)]
pub enum IcarusError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Image I/O error: {0}")]
    ImageIo(String),

    #[error("Invalid image format: {0}")]
    InvalidFormat(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Detection error: {0}")]
    Detection(String),

    #[error("Cropping error: {0}")]
    Cropping(String),

    #[error("Invalid configuration value: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, IcarusError>;

impl From<io::Error> for IcarusError {
    fn from(err: io::Error) -> Self {
        IcarusError::ImageIo(err.to_string())
    }
}

impl From<image::ImageError> for IcarusError {
    fn from(err: image::ImageError) -> Self {
        IcarusError::ImageIo(err.to_string())
    }
}

impl From<toml::de::Error> for IcarusError {
    fn from(err: toml::de::Error) -> Self {
        IcarusError::Config(err.to_string())
    }
}

pub fn exit_code(err: &IcarusError) -> i32 {
    match err {
        IcarusError::Config(_) | IcarusError::InvalidConfig(_) => 2,
        IcarusError::ImageIo(_) | IcarusError::InvalidFormat(_) => 3,
        IcarusError::Model(_) | IcarusError::Detection(_) | IcarusError::Cropping(_) => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageError;
    use std::io;

    #[test]
    fn io_error_maps_to_image_io_variant() {
        let err = io::Error::new(io::ErrorKind::NotFound, "boom");
        match IcarusError::from(err) {
            IcarusError::ImageIo(msg) => assert!(msg.contains("boom")),
            other => panic!("Unexpected variant: {:?}", other),
        }
    }

    #[test]
    fn image_error_maps_to_image_io_variant() {
        let err = ImageError::IoError(io::Error::new(io::ErrorKind::Other, "img err"));
        match IcarusError::from(err) {
            IcarusError::ImageIo(msg) => assert!(msg.contains("img err")),
            other => panic!("Unexpected variant: {:?}", other),
        }
    }

    #[test]
    fn toml_error_maps_to_config_variant() {
        let err = toml::from_str::<usize>("not a number").unwrap_err();
        match IcarusError::from(err) {
            IcarusError::Config(msg) => assert!(msg.contains("not a number")),
            other => panic!("Unexpected variant: {:?}", other),
        }
    }
}
