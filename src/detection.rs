use crate::error::IcarusError;
use image::DynamicImage;
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

/// Enum to identify the ONNX model type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    YOLOv10,
    DETR,
}

/// Represents a bounding box with confidence score
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub confidence: f32,
    pub class_id: Option<u32>,
}

impl BoundingBox {
    pub fn center_x(&self) -> f32 {
        (self.xmin + self.xmax) / 2.0
    }

    pub fn center_y(&self) -> f32 {
        (self.ymin + self.ymax) / 2.0
    }

    pub fn width(&self) -> f32 {
        self.xmax - self.xmin
    }

    pub fn height(&self) -> f32 {
        self.ymax - self.ymin
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

/// Result of running detection on an image
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Center X coordinate of highest-confidence person detection
    pub center_x: f32,
    /// Center Y coordinate of highest-confidence person detection
    pub center_y: f32,
    /// Confidence score of the detection (0.0-1.0)
    pub confidence: f32,
    /// Whether a person was detected above confidence threshold
    pub has_person: bool,
}

impl Default for DetectionResult {
    fn default() -> Self {
        Self {
            center_x: 0.0,
            center_y: 0.0,
            confidence: 0.0,
            has_person: false,
        }
    }
}

/// ONNX-based person detector using tract
pub struct OnnxDetector {
    model_path: std::path::PathBuf,
    model_type: ModelType,
    input_size: (u32, u32),
    confidence_threshold: f32,
    id2label: HashMap<u32, String>,
}

impl OnnxDetector {
    /// Load an ONNX model from disk and prepare for inference
    pub fn new(model_path: &Path, confidence_threshold: f32) -> Result<Self, IcarusError> {
        // 1. Verify file exists
        if !model_path.exists() {
            return Err(IcarusError::Model(format!(
                "ONNX model not found: {}. Expected from config.model.model_path",
                model_path.display()
            )));
        }

        info!("Loading ONNX model from: {}", model_path.display());

        // 3. Detect model type from path
        let model_type = if model_path.to_string_lossy().contains("yolov10") {
            ModelType::YOLOv10
        } else if model_path.to_string_lossy().contains("detr") {
            ModelType::DETR
        } else {
            ModelType::YOLOv10 // Default to YOLOv10
        };

        info!("Detected model type: {:?}", model_type);

        // 4. Parse config.json for input size and id2label
        let config_path = model_path
            .parent()
            .ok_or_else(|| IcarusError::Model("Model path has no parent".to_string()))?
            .join("config.json");

        let (input_size, id2label) = if config_path.exists() {
            parse_model_config(&config_path)?
        } else {
            // Default values based on model type
            match model_type {
                ModelType::YOLOv10 => ((640, 640), default_id2label()),
                ModelType::DETR => ((800, 800), default_id2label()),
            }
        };

        info!("Model input size: {:?}", input_size);
        info!("Found {} classes in id2label mapping", id2label.len());

        Ok(OnnxDetector {
            model_path: model_path.to_path_buf(),
            model_type,
            input_size,
            confidence_threshold,
            id2label,
        })
    }

    /// Run detection on an image and return the center of the highest-confidence person
    pub fn detect(&self, image: &DynamicImage) -> Result<DetectionResult, IcarusError> {
        // Note: For Phase 3, we load and run ONNX models
        // For now, this is a placeholder that will be fully implemented with actual ONNX inference
        // The infrastructure is in place to load the model at new() time, but inference requires
        // runtime model loading to avoid type system issues with tract's runtime API

        info!(
            "Running detection on {}x{} image",
            image.width(),
            image.height()
        );

        // For now, return center as fallback
        let fallback_center_x = image.width() as f32 / 2.0;
        let fallback_center_y = image.height() as f32 / 2.0;

        warn!("Detection inference not yet fully implemented, falling back to image center");

        Ok(DetectionResult {
            center_x: fallback_center_x,
            center_y: fallback_center_y,
            confidence: 0.0,
            has_person: false,
        })
    }
}

/// Parse config.json to extract input_size and id2label mapping
fn parse_model_config(
    config_path: &Path,
) -> Result<((u32, u32), HashMap<u32, String>), IcarusError> {
    let contents = std::fs::read_to_string(config_path)
        .map_err(|e| IcarusError::Model(format!("Failed to read config.json: {}", e)))?;

    let config: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|e| IcarusError::Model(format!("Failed to parse config.json: {}", e)))?;

    // Try different keys for input size
    let input_size = if let Some(size) = config.get("input_size").and_then(|v| v.as_array()) {
        if size.len() >= 2 {
            let w = size[0].as_u64().unwrap_or(640) as u32;
            let h = size[1].as_u64().unwrap_or(640) as u32;
            (w, h)
        } else {
            (640, 640)
        }
    } else if let Some(size) = config.get("image_size").and_then(|v| v.as_array()) {
        if size.len() >= 2 {
            let w = size[0].as_u64().unwrap_or(640) as u32;
            let h = size[1].as_u64().unwrap_or(640) as u32;
            (w, h)
        } else {
            (640, 640)
        }
    } else {
        (640, 640)
    };

    // Parse id2label mapping
    let mut id2label = HashMap::new();
    if let Some(mapping) = config.get("id2label").and_then(|v| v.as_object()) {
        for (id_str, label) in mapping {
            if let Ok(id) = id_str.parse::<u32>() {
                if let Some(label_str) = label.as_str() {
                    id2label.insert(id, label_str.to_string());
                }
            }
        }
    }

    // Ensure "person" class exists
    if !id2label.values().any(|l| l == "person") {
        id2label.insert(0, "person".to_string());
    }

    Ok((input_size, id2label))
}

fn default_id2label() -> HashMap<u32, String> {
    let mut map = HashMap::new();
    map.insert(0, "person".to_string());
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_result_default() {
        let result = DetectionResult::default();
        assert_eq!(result.center_x, 0.0);
        assert_eq!(result.center_y, 0.0);
        assert_eq!(result.confidence, 0.0);
        assert!(!result.has_person);
    }

    #[test]
    fn test_detection_result_creation() {
        let result = DetectionResult {
            center_x: 100.0,
            center_y: 150.0,
            confidence: 0.95,
            has_person: true,
        };
        assert_eq!(result.center_x, 100.0);
        assert_eq!(result.center_y, 150.0);
        assert_eq!(result.confidence, 0.95);
        assert!(result.has_person);
    }

    #[test]
    fn test_bounding_box_center() {
        let bbox = BoundingBox {
            xmin: 10.0,
            ymin: 20.0,
            xmax: 110.0,
            ymax: 220.0,
            confidence: 0.9,
            class_id: Some(0),
        };

        assert_eq!(bbox.center_x(), 60.0);
        assert_eq!(bbox.center_y(), 120.0);
        assert_eq!(bbox.width(), 100.0);
        assert_eq!(bbox.height(), 200.0);
        assert_eq!(bbox.area(), 20000.0);
    }

    #[test]
    fn test_default_id2label() {
        let map = default_id2label();
        assert!(map.contains_key(&0));
        assert_eq!(map.get(&0).map(|s| s.as_str()), Some("person"));
    }
}
