use crate::detection::{BoundingBox, DetectionResult};
use image::{DynamicImage, Rgb, RgbImage};

/// Draw a detection bounding box marker on an image
///
/// Creates a rectangle outline around the detected person bounding box.
/// This is useful for debugging detection results during finetuning.
///
/// # Arguments
/// * `image` - The original image to draw on
/// * `detection` - The detection result to visualize
///
/// # Returns
/// A new image with drawn detection marker, or the original if no detection
pub fn draw_detection_marker(image: &DynamicImage, detection: &DetectionResult) -> DynamicImage {
    if !detection.has_person {
        return image.clone();
    }

    let mut img = image.to_rgb8();
    let box_color = Rgb([0u8, 255u8, 0u8]); // Green

    // Use actual bounding box coordinates from detection
    let x_min = detection.bbox_xmin.max(0.0) as u32;
    let y_min = detection.bbox_ymin.max(0.0) as u32;
    let x_max = detection.bbox_xmax.min(img.width() as f32) as u32;
    let y_max = detection.bbox_ymax.min(img.height() as f32) as u32;

    // Only draw if box is valid
    if x_max > x_min && y_max > y_min {
        // Draw rectangle outline (2 pixel thickness)
        draw_hollow_rect(&mut img, x_min, y_min, x_max, y_max, box_color, 2);
    }

    DynamicImage::ImageRgb8(img)
}

/// Draw bounding boxes with confidence scores on an image
///
/// Creates outlined rectangles for each detection.
/// Draws on a copy of the image, leaving the original unchanged.
///
/// # Arguments
/// * `image` - The original image to draw boxes on
/// * `boxes` - Slice of bounding boxes to draw
///
/// # Returns
/// A new image with drawn boxes, or the original if no boxes provided
pub fn draw_bounding_boxes(image: &DynamicImage, boxes: &[BoundingBox]) -> DynamicImage {
    if boxes.is_empty() {
        return image.clone();
    }

    let mut img = image.to_rgb8();
    let box_color = Rgb([0u8, 255u8, 0u8]); // Green for box outlines

    for bbox in boxes {
        let x_min = bbox.xmin.max(0.0) as u32;
        let y_min = bbox.ymin.max(0.0) as u32;
        let x_max = bbox.xmax.min(img.width() as f32) as u32;
        let y_max = bbox.ymax.min(img.height() as f32) as u32;

        if x_max > x_min && y_max > y_min {
            draw_hollow_rect(&mut img, x_min, y_min, x_max, y_max, box_color, 2);
        }
    }

    DynamicImage::ImageRgb8(img)
}

/// Draw a hollow rectangle on an image
/// Uses a simple pixel-based approach without external dependencies
fn draw_hollow_rect(
    img: &mut RgbImage,
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    color: Rgb<u8>,
    thickness: u32,
) {
    let width = img.width();
    let height = img.height();

    // Draw horizontal lines (top and bottom)
    for x in x_min..x_max {
        // Top line
        for t in 0..thickness {
            if y_min + t < height && x < width {
                img.put_pixel(x, y_min + t, color);
            }
        }
        // Bottom line
        for t in 0..thickness {
            if y_max > t && y_max - t - 1 < height && x < width {
                img.put_pixel(x, y_max - t - 1, color);
            }
        }
    }

    // Draw vertical lines (left and right)
    for y in y_min..y_max {
        // Left line
        for t in 0..thickness {
            if x_min + t < width && y < height {
                img.put_pixel(x_min + t, y, color);
            }
        }
        // Right line
        for t in 0..thickness {
            if x_max > t && x_max - t - 1 < width && y < height {
                img.put_pixel(x_max - t - 1, y, color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_detection_marker() {
        let img = DynamicImage::new_rgb8(100, 100);
        let detection = DetectionResult {
            center_x: 50.0,
            center_y: 50.0,
            bbox_xmin: 30.0,
            bbox_ymin: 30.0,
            bbox_xmax: 70.0,
            bbox_ymax: 70.0,
            confidence: 0.9,
            has_person: true,
        };

        let result = draw_detection_marker(&img, &detection);

        assert_eq!(result.width(), img.width());
        assert_eq!(result.height(), img.height());
    }

    #[test]
    fn test_draw_detection_marker_no_detection() {
        let img = DynamicImage::new_rgb8(100, 100);
        let detection = DetectionResult {
            center_x: 0.0,
            center_y: 0.0,
            bbox_xmin: 0.0,
            bbox_ymin: 0.0,
            bbox_xmax: 0.0,
            bbox_ymax: 0.0,
            confidence: 0.0,
            has_person: false,
        };

        let result = draw_detection_marker(&img, &detection);

        assert_eq!(result.width(), img.width());
        assert_eq!(result.height(), img.height());
    }

    #[test]
    fn test_draw_bounding_boxes_empty() {
        let img = DynamicImage::new_rgb8(100, 100);
        let boxes = vec![];
        let result = draw_bounding_boxes(&img, &boxes);

        assert_eq!(result.width(), img.width());
        assert_eq!(result.height(), img.height());
    }

    #[test]
    fn test_draw_bounding_boxes_single() {
        let img = DynamicImage::new_rgb8(200, 200);
        let boxes = vec![BoundingBox {
            xmin: 50.0,
            ymin: 50.0,
            xmax: 150.0,
            ymax: 150.0,
            confidence: 0.95,
            class_id: Some(0),
        }];

        let result = draw_bounding_boxes(&img, &boxes);

        assert_eq!(result.width(), img.width());
        assert_eq!(result.height(), img.height());
    }
}
