use crate::error::IcarusError;

/// Represents a crop box with position and dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Crop configuration for smart center-based cropping
#[derive(Debug, Clone)]
pub struct CropConfig {
    /// Minimum crop dimension as ratio of image size (e.g., 0.75 = 75% of image)
    pub min_dimension_ratio: f32,
    /// Minimum height required for portrait-oriented images
    pub min_height_portrait: u32,
    /// Minimum width required for landscape-oriented images
    pub min_width_landscape: u32,
}

impl Default for CropConfig {
    fn default() -> Self {
        Self {
            min_dimension_ratio: 0.75,
            min_height_portrait: 1800,
            min_width_landscape: 1800,
        }
    }
}

/// Validates whether an image meets minimum quality/dimension requirements
///
/// Determines orientation based on image dimensions:
/// - Portrait: height > width → requires min_height_portrait
/// - Landscape: width >= height → requires min_width_landscape
///
/// Returns Ok(()) if image meets quality standards, Err(String) with reason if not
pub fn validate_image_quality(width: u32, height: u32, config: &CropConfig) -> Result<(), String> {
    let is_portrait = height > width;

    if is_portrait {
        if height < config.min_height_portrait {
            return Err(format!(
                "Image height ({} px) is below minimum required for portrait orientation ({} px)",
                height, config.min_height_portrait
            ));
        }
    } else {
        if width < config.min_width_landscape {
            return Err(format!(
                "Image width ({} px) is below minimum required for landscape orientation ({} px)",
                width, config.min_width_landscape
            ));
        }
    }

    Ok(())
}

/// Calculate the maximum crop rectangle for a given aspect ratio, centered on detection.
///
/// Finds the largest rectangle that:
/// - Maintains exact aspect ratio (ratio_w : ratio_h)
/// - Fits entirely within image bounds
/// - Is centered on (or near) the detection center point
///
/// When the subject is near an image edge, the crop shifts to stay in bounds
/// while preserving maximum resolution.
///
/// # Arguments
/// * `image_width` - Width of source image in pixels
/// * `image_height` - Height of source image in pixels
/// * `detection_center_x` - X coordinate of detection center
/// * `detection_center_y` - Y coordinate of detection center
/// * `ratio_w` - Aspect ratio width component (e.g., 9 for 9:16)
/// * `ratio_h` - Aspect ratio height component (e.g., 16 for 9:16)
///
/// # Returns
/// A `CropBox` representing the maximum crop rectangle
pub fn calculate_max_aspect_crop(
    image_width: u32,
    image_height: u32,
    detection_center_x: f32,
    detection_center_y: f32,
    ratio_w: u32,
    ratio_h: u32,
) -> CropBox {
    let img_w = image_width as f32;
    let img_h = image_height as f32;
    let ratio_w = ratio_w as f32;
    let ratio_h = ratio_h as f32;

    // Step 1: Calculate maximum crop dimensions that fit the aspect ratio
    // Try to use full width first
    let crop_h_if_full_width = img_w * (ratio_h / ratio_w);

    let (crop_width, crop_height) = if crop_h_if_full_width <= img_h {
        // Image is wide enough: use full width
        (img_w, crop_h_if_full_width)
    } else {
        // Image is too tall for full width: use full height
        let crop_w_if_full_height = img_h * (ratio_w / ratio_h);
        (crop_w_if_full_height, img_h)
    };

    // Round to integer dimensions while maintaining aspect ratio
    // Prefer rounding down to avoid exceeding bounds
    let crop_width = crop_width.floor() as u32;
    let crop_height = crop_height.floor() as u32;

    // Ensure we don't exceed image bounds (paranoid check)
    let crop_width = crop_width.min(image_width);
    let crop_height = crop_height.min(image_height);

    // Step 2: Center on detection point (with clamping)
    let half_width = crop_width as f32 / 2.0;
    let half_height = crop_height as f32 / 2.0;

    let mut x_start = (detection_center_x - half_width).floor() as i32;
    let mut y_start = (detection_center_y - half_height).floor() as i32;

    // Step 3: Clamp to image bounds (this is where decentering happens)
    let max_x_start = image_width as i32 - crop_width as i32;
    let max_y_start = image_height as i32 - crop_height as i32;

    x_start = x_start.max(0).min(max_x_start);
    y_start = y_start.max(0).min(max_y_start);

    CropBox {
        x: x_start as u32,
        y: y_start as u32,
        width: crop_width,
        height: crop_height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageBuffer;

    #[test]
    fn test_calculate_max_aspect_crop_portrait_fits_in_portrait() {
        // Portrait image 1000x2000, portrait aspect 9:16
        let crop = calculate_max_aspect_crop(1000, 2000, 500.0, 1000.0, 9, 16);

        // Should use full width, calculate height from aspect
        // width = 1000, height = 1000 * (16/9) = 1777.78 -> 1777
        assert_eq!(crop.width, 1000);
        assert_eq!(crop.height, 1777);
        assert!(crop.x + crop.width <= 1000);
        assert!(crop.y + crop.height <= 2000);
    }

    #[test]
    fn test_calculate_max_aspect_crop_landscape_fits_landscape() {
        // Landscape image 3000x2000, landscape aspect 20:9
        let crop = calculate_max_aspect_crop(3000, 2000, 1500.0, 1000.0, 20, 9);

        // Should use full height, calculate width from aspect
        // height = 2000, width = 2000 * (20/9) = 4444.44 > 3000
        // So use full width: width = 3000, height = 3000 * (9/20) = 1350
        assert_eq!(crop.width, 3000);
        assert_eq!(crop.height, 1350);
        assert!(crop.x + crop.width <= 3000);
        assert!(crop.y + crop.height <= 2000);
    }

    #[test]
    fn test_calculate_max_aspect_crop_mobile_from_landscape() {
        // Landscape image 2560x1440, mobile aspect 9:20
        let crop = calculate_max_aspect_crop(2560, 1440, 1280.0, 720.0, 9, 20);

        // height = 1440, width = 1440 * (9/20) = 648
        // Can fit: width = 648, height = 1440
        assert_eq!(crop.width, 648);
        assert_eq!(crop.height, 1440);
        assert!(crop.x + crop.width <= 2560);
        assert!(crop.y + crop.height <= 1440);
    }

    #[test]
    fn test_calculate_max_aspect_crop_centered_on_detection() {
        // 2000x2000 square, centered detection at (1000, 1000)
        // Aspect 1:1 (square), should place crop centered
        let crop = calculate_max_aspect_crop(2000, 2000, 1000.0, 1000.0, 1, 1);

        // Should use full dimensions
        assert_eq!(crop.width, 2000);
        assert_eq!(crop.height, 2000);
        assert_eq!(crop.x, 0);
        assert_eq!(crop.y, 0);
    }

    #[test]
    fn test_calculate_max_aspect_crop_detection_near_corner() {
        // 2000x2000 image, detection at top-left (100, 100)
        // Portrait aspect 9:16
        // Full width 2000 needs height 2000*(16/9)=3556 (exceeds)
        // So use full height 2000, width = 2000*(9/16)=1125
        let crop = calculate_max_aspect_crop(2000, 2000, 100.0, 100.0, 9, 16);

        // Crop centered on (100, 100) would start at:
        // x = 100 - 562.5 = -462.5 -> clamped to 0
        // y = 100 - 1000 = -900 -> clamped to 0
        // Result should be at (0, 0) with calculated max dimensions
        assert_eq!(crop.x, 0);
        assert_eq!(crop.y, 0);
        assert_eq!(crop.width, 1125);
        assert_eq!(crop.height, 2000);
    }

    #[test]
    fn test_calculate_max_aspect_crop_detection_near_edge() {
        // 2000x2000 image, detection at right edge (1900, 1000)
        // Portrait aspect 9:16
        // Max crop: full height 2000, width = 2000*(9/16)=1125
        let crop = calculate_max_aspect_crop(2000, 2000, 1900.0, 1000.0, 9, 16);

        // Would try to center at (1900, 1000), but width 1125 centered there
        // would put start at 1900 - 562.5 = 1337.5, end at ~2462 (exceeds)
        // Clamped to fit: x = 2000 - 1125 = 875 (so end is at 2000)
        assert_eq!(crop.x, 875);
        assert_eq!(crop.width, 1125);
        assert_eq!(crop.height, 2000);
        assert!(crop.x + crop.width <= 2000);
    }

    #[test]
    fn test_calculate_max_aspect_crop_tiny_image() {
        // Tiny 100x100 image, portrait aspect 9:16
        let crop = calculate_max_aspect_crop(100, 100, 50.0, 50.0, 9, 16);

        // width = 100, height = 100 * (16/9) = 177.78 > 100
        // So use height = 100, width = 100 * (9/16) = 56.25 -> 56
        assert_eq!(crop.width, 56);
        assert_eq!(crop.height, 100);
        assert!(crop.x + crop.width <= 100);
        assert!(crop.y + crop.height <= 100);
    }

    #[test]
    fn test_calculate_max_aspect_crop_preserves_aspect_ratio() {
        // Verify the output maintains exact aspect ratio
        let crop = calculate_max_aspect_crop(3840, 2160, 1920.0, 1080.0, 20, 9);

        // Check aspect ratio is maintained (allowing for rounding)
        let crop_aspect = crop.width as f32 / crop.height as f32;
        let target_aspect = 20.0 / 9.0;
        let aspect_diff = (crop_aspect - target_aspect).abs();
        // Allow small rounding difference
        assert!(
            aspect_diff < 0.01,
            "Aspect ratio not maintained: got {}, expected {}",
            crop_aspect,
            target_aspect
        );
    }

    #[test]
    fn test_crop_config_default() {
        let config = CropConfig::default();
        assert_eq!(config.min_dimension_ratio, 0.75);
        assert_eq!(config.min_height_portrait, 1800);
        assert_eq!(config.min_width_landscape, 1800);
    }

    #[test]
    fn test_validate_portrait_passes() {
        let config = CropConfig {
            min_dimension_ratio: 0.75,
            min_height_portrait: 1800,
            min_width_landscape: 1800,
        };

        // Portrait image that passes: 1000x2000
        let result = validate_image_quality(1000, 2000, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_portrait_fails() {
        let config = CropConfig {
            min_dimension_ratio: 0.75,
            min_height_portrait: 1800,
            min_width_landscape: 1800,
        };

        // Portrait image that fails: 1000x1500 (height too low)
        let result = validate_image_quality(1000, 1500, &config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("1500"));
        assert!(err_msg.contains("1800"));
    }

    #[test]
    fn test_validate_landscape_passes() {
        let config = CropConfig {
            min_dimension_ratio: 0.75,
            min_height_portrait: 1800,
            min_width_landscape: 1800,
        };

        // Landscape image that passes: 2000x1000
        let result = validate_image_quality(2000, 1000, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_landscape_fails() {
        let config = CropConfig {
            min_dimension_ratio: 0.75,
            min_height_portrait: 1800,
            min_width_landscape: 1800,
        };

        // Landscape image that fails: 1500x1000 (width too low)
        let result = validate_image_quality(1500, 1000, &config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("1500"));
        assert!(err_msg.contains("1800"));
    }

    #[test]
    fn test_validate_square_image() {
        let config = CropConfig {
            min_dimension_ratio: 0.75,
            min_height_portrait: 1800,
            min_width_landscape: 1800,
        };

        // Square image is treated as landscape (width >= height)
        // So it should check width >= 1800
        let result = validate_image_quality(2000, 2000, &config);
        assert!(result.is_ok());

        let result = validate_image_quality(1500, 1500, &config);
        assert!(result.is_err());
    }
}
