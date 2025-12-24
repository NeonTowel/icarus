use image::{DynamicImage, ImageFormat as ImgFormat};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use tracing::{debug, info, instrument};

use crate::error::{IcarusError, Result};

pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub format: ImgFormat,
    pub has_exif: bool,
}

#[instrument(skip_all, fields(path = %path.display()))]
pub fn load_image(path: &Path) -> Result<DynamicImage> {
    info!("Loading image: {}", path.display());

    if !path.exists() {
        return Err(IcarusError::ImageIo(format!(
            "File not found: {}. Check if the path is correct.",
            path.display()
        )));
    }

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let format = detect_format(&mut reader)?;
    debug!("Detected format: {:?}", format);

    let img = image::load(reader, format)?;

    validate_dimensions(img.width(), img.height())?;

    let img = apply_exif_orientation(img, path)?;

    info!(
        "Image loaded successfully: {}x{} pixels",
        img.width(),
        img.height()
    );

    Ok(img)
}

#[instrument(skip(image), fields(path = %path.display()))]
pub fn save_image(image: &DynamicImage, path: &Path, format: ImgFormat, quality: u8) -> Result<()> {
    info!("Saving image to: {}", path.display());

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    match format {
        ImgFormat::Jpeg => {
            let mut file = File::create(path)?;
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut file, quality);
            image.write_with_encoder(encoder)?;
        }
        ImgFormat::Png => {
            image.save_with_format(path, ImgFormat::Png)?;
        }
        ImgFormat::WebP => {
            image.save_with_format(path, ImgFormat::WebP)?;
        }
        _ => {
            return Err(IcarusError::InvalidFormat(format!(
                "Unsupported output format: {:?}",
                format
            )));
        }
    }

    info!("Image saved successfully");
    Ok(())
}

#[instrument(skip_all, fields(path = %path.display()))]
pub fn get_image_info(path: &Path) -> Result<ImageInfo> {
    if !path.exists() {
        return Err(IcarusError::ImageIo(format!(
            "File not found: {}",
            path.display()
        )));
    }

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let format = detect_format(&mut reader)?;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let img = image::load(reader, format)?;

    let has_exif = check_exif(path);

    Ok(ImageInfo {
        width: img.width(),
        height: img.height(),
        format,
        has_exif,
    })
}

fn detect_format<R: Read + Seek>(reader: &mut R) -> Result<ImgFormat> {
    let mut magic_bytes = [0u8; 16];
    reader
        .read_exact(&mut magic_bytes)
        .map_err(|e| IcarusError::InvalidFormat(format!("Failed to read file header: {}", e)))?;
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|e| IcarusError::ImageIo(format!("Failed to rewind file: {}", e)))?;

    if magic_bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Ok(ImgFormat::Jpeg)
    } else if magic_bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        Ok(ImgFormat::Png)
    } else if magic_bytes[8..12] == [0x57, 0x45, 0x42, 0x50] {
        Ok(ImgFormat::WebP)
    } else if magic_bytes.starts_with(&[0x49, 0x49, 0x2A, 0x00])
        || magic_bytes.starts_with(&[0x4D, 0x4D, 0x00, 0x2A])
    {
        Ok(ImgFormat::Tiff)
    } else {
        Err(IcarusError::InvalidFormat(
            "Unknown or unsupported image format. Supported formats: JPEG, PNG, WebP, TIFF"
                .to_string(),
        ))
    }
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    const MIN_DIMENSION: u32 = 100;
    const MAX_DIMENSION: u32 = 16384;

    if width < MIN_DIMENSION || height < MIN_DIMENSION {
        return Err(IcarusError::InvalidFormat(format!(
            "Image dimensions too small: {}x{}. Minimum: {}x{}",
            width, height, MIN_DIMENSION, MIN_DIMENSION
        )));
    }

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(IcarusError::InvalidFormat(format!(
            "Image dimensions too large: {}x{}. Maximum: {}x{}",
            width, height, MAX_DIMENSION, MAX_DIMENSION
        )));
    }

    Ok(())
}

fn check_exif(path: &Path) -> bool {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };

    let mut buf_reader = BufReader::new(&file);
    exif::Reader::new()
        .read_from_container(&mut buf_reader)
        .is_ok()
}

fn apply_exif_orientation(img: DynamicImage, path: &Path) -> Result<DynamicImage> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(img),
    };

    let mut buf_reader = BufReader::new(&file);
    let exif_reader = exif::Reader::new();

    let exif = match exif_reader.read_from_container(&mut buf_reader) {
        Ok(e) => e,
        Err(_) => return Ok(img),
    };

    let orientation = match exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY) {
        Some(field) => match field.value.get_uint(0) {
            Some(v) => v,
            None => return Ok(img),
        },
        None => return Ok(img),
    };

    debug!("EXIF orientation: {}", orientation);
    Ok(rotate_for_orientation(img, orientation as u16))
}

fn rotate_for_orientation(img: DynamicImage, orientation: u16) -> DynamicImage {
    match orientation {
        1 => img,
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.rotate90().fliph(),
        6 => img.rotate90(),
        7 => img.rotate270().fliph(),
        8 => img.rotate270(),
        _ => img,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GenericImageView, ImageFormat as ImgFormat};
    use std::io::Cursor;

    #[test]
    fn test_validate_dimensions() {
        assert!(validate_dimensions(100, 100).is_ok());
        assert!(validate_dimensions(1920, 1080).is_ok());
        assert!(validate_dimensions(16384, 16384).is_ok());

        assert!(validate_dimensions(99, 100).is_err());
        assert!(validate_dimensions(100, 99).is_err());
        assert!(validate_dimensions(16385, 100).is_err());
        assert!(validate_dimensions(100, 16385).is_err());
    }

    #[test]
    fn detect_format_recognizes_known_formats() {
        fn make_cursor(mut data: Vec<u8>) -> Cursor<Vec<u8>> {
            data.resize(16, 0);
            Cursor::new(data)
        }

        let jpeg_data = vec![0xFF, 0xD8, 0xFF];
        assert_eq!(
            detect_format(&mut make_cursor(jpeg_data)).unwrap(),
            ImgFormat::Jpeg
        );

        let png_data = vec![0x89, 0x50, 0x4E, 0x47];
        assert_eq!(
            detect_format(&mut make_cursor(png_data)).unwrap(),
            ImgFormat::Png
        );

        let mut webp_data = vec![0; 16];
        webp_data[8..12].copy_from_slice(b"WEBP");
        assert_eq!(
            detect_format(&mut Cursor::new(webp_data)).unwrap(),
            ImgFormat::WebP
        );

        let tiff_data = vec![0x49, 0x49, 0x2A, 0x00];
        assert_eq!(
            detect_format(&mut make_cursor(tiff_data)).unwrap(),
            ImgFormat::Tiff
        );
    }

    #[test]
    fn detect_format_rejects_unknown_headers() {
        let data = vec![0u8; 16];
        assert!(detect_format(&mut Cursor::new(data)).is_err());
    }

    #[test]
    fn rotate_for_orientation_swaps_dimensions() {
        let img = DynamicImage::new_rgb8(3, 5);
        let rotated = rotate_for_orientation(img.clone(), 6);
        assert_eq!(rotated.dimensions(), (5, 3));

        let flipped = rotate_for_orientation(img.clone(), 2);
        assert_eq!(flipped.dimensions(), (3, 5));
    }
}
