use image::{DynamicImage, GenericImageView, ImageFormat as ImgFormat};
use ndarray::Array4;
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

pub struct PreprocessedImage {
    pub tensor: Array4<f32>,
    pub scale_x: f32,
    pub scale_y: f32,
    pub pad_x: f32,
    pub pad_y: f32,
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

#[instrument(skip(image))]
pub fn preprocess_for_detection(
    image: &DynamicImage,
    target_size: u32,
) -> Result<PreprocessedImage> {
    let (orig_width, orig_height) = image.dimensions();
    debug!(
        "Preprocessing image: {}x{} -> {}x{}",
        orig_width, orig_height, target_size, target_size
    );

    // Calculate scale factors for aspect ratio preservation
    let scale = (target_size as f32) / (orig_width.max(orig_height) as f32);
    let new_width = (orig_width as f32 * scale) as u32;
    let new_height = (orig_height as f32 * scale) as u32;

    // Resize to maintain aspect ratio
    let resized = image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);

    // Get actual resized dimensions (may differ slightly from calculated due to rounding)
    let (actual_width, actual_height) = resized.dimensions();

    // Calculate padding to reach target size
    let pad_x = (target_size - actual_width) as f32;
    let pad_y = (target_size - actual_height) as f32;
    let pad_left = (pad_x / 2.0).floor() as u32;
    let pad_top = (pad_y / 2.0).floor() as u32;

    // Create padded image (with black/zero padding)
    let mut padded =
        image::ImageBuffer::from_pixel(target_size, target_size, image::Rgb([0u8, 0u8, 0u8]));

    // Copy resized image to padded location
    let resized_rgb = resized.to_rgb8();
    for y in 0..actual_height {
        for x in 0..actual_width {
            let pixel = resized_rgb.get_pixel(x, y);
            padded.put_pixel(pad_left + x, pad_top + y, *pixel);
        }
    }
    let padded_img = DynamicImage::ImageRgb8(padded);

    // Convert to RGB if needed
    let rgb_img = padded_img.to_rgb8();

    // Normalize pixels to [0, 1] and convert to ndarray in NCHW format
    // NCHW: (Batch=1, Channels=3, Height, Width)
    let mut tensor = Array4::<f32>::zeros((1, 3, target_size as usize, target_size as usize));

    for y in 0..target_size as usize {
        for x in 0..target_size as usize {
            let pixel = rgb_img.get_pixel(x as u32, y as u32);
            let r = (pixel[0] as f32) / 255.0;
            let g = (pixel[1] as f32) / 255.0;
            let b = (pixel[2] as f32) / 255.0;

            tensor[[0, 0, y, x]] = r;
            tensor[[0, 1, y, x]] = g;
            tensor[[0, 2, y, x]] = b;
        }
    }

    debug!(
        "Preprocessing complete: scale={:.4}, pad=({:.2}, {:.2})",
        scale, pad_x, pad_y
    );

    Ok(PreprocessedImage {
        tensor,
        scale_x: scale,
        scale_y: scale,
        pad_x: pad_left as f32,
        pad_y: pad_top as f32,
    })
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

    #[test]
    fn preprocess_creates_correct_tensor_shape() {
        let img = DynamicImage::new_rgb8(640, 480);
        let result = preprocess_for_detection(&img, 640).unwrap();

        // Should produce a 4D tensor in NCHW format (batch=1, channels=3, height=640, width=640)
        assert_eq!(result.tensor.dim(), (1, 3, 640, 640));
    }

    #[test]
    fn preprocess_normalizes_pixel_values() {
        let img = DynamicImage::new_rgb8(100, 100);
        let result = preprocess_for_detection(&img, 100).unwrap();

        // All pixels should be between 0 and 1
        let min = result.tensor.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = result.tensor.iter().cloned().fold(0.0, f32::max);

        assert!(min >= 0.0);
        assert!(max <= 1.0);
    }

    #[test]
    fn preprocess_maintains_aspect_ratio() {
        let img = DynamicImage::new_rgb8(1920, 1080);
        let result = preprocess_for_detection(&img, 640).unwrap();

        // Scale should be around 0.333... (640 / 1920)
        let expected_scale = 640.0 / 1920.0;
        assert!((result.scale_x - expected_scale).abs() < 0.01);
        assert_eq!(result.scale_x, result.scale_y);
    }

    #[test]
    fn preprocess_handles_small_images() {
        let img = DynamicImage::new_rgb8(320, 240);
        let result = preprocess_for_detection(&img, 640).unwrap();

        assert_eq!(result.tensor.dim(), (1, 3, 640, 640));
        // Scale should be 2.0 (640 / 320)
        assert!((result.scale_x - 2.0).abs() < 0.01);
    }
}
