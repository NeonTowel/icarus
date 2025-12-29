use crate::error::IcarusError;
use crate::image_io::{preprocess_for_detection, PreprocessedImage};
use image::DynamicImage;
use ndarray::{Array3, Array4};
use ort::session::Session;
use ort::value::Value;
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

/// Internal enum to represent different ONNX model outputs
enum InferenceOutput {
    YOLOv10(Array3<f32>),
    DETR {
        logits: Array3<f32>,
        boxes: Array3<f32>,
    },
}

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
    /// Left edge of bounding box
    pub bbox_xmin: f32,
    /// Top edge of bounding box
    pub bbox_ymin: f32,
    /// Right edge of bounding box
    pub bbox_xmax: f32,
    /// Bottom edge of bounding box
    pub bbox_ymax: f32,
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
            bbox_xmin: 0.0,
            bbox_ymin: 0.0,
            bbox_xmax: 0.0,
            bbox_ymax: 0.0,
            confidence: 0.0,
            has_person: false,
        }
    }
}

/// ONNX-based person detector using ort runtime
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

    /// Load and run ONNX session for inference
    /// Returns output tensors based on model type (YOLOv10 vs DETR)
    fn run_inference(&self, input_tensor: Array4<f32>) -> Result<InferenceOutput, IcarusError> {
        debug!(
            "Loading and running ONNX session from: {}",
            self.model_path.display()
        );

        // Verify model file exists
        if !self.model_path.exists() {
            return Err(IcarusError::Model(format!(
                "Model file not found: {}",
                self.model_path.display()
            )));
        }

        debug!("Input tensor shape: {:?}", input_tensor.dim());
        debug!("Model type: {:?}", self.model_type);

        // Create ONNX Runtime session directly (ort 2.0.0-rc.1 API)
        // No separate Environment needed - it's managed internally
        let mut session = Session::builder()
            .map_err(|e| IcarusError::Model(format!("Failed to create session builder: {}", e)))?
            .commit_from_file(&self.model_path)
            .map_err(|e| {
                IcarusError::Model(format!(
                    "Failed to load ONNX model from {}: {}",
                    self.model_path.display(),
                    e
                ))
            })?;

        debug!("ONNX session created successfully");

        // Convert ndarray tensor to ort::Value for inference
        // Value::from_array expects a (Shape, Vec<T>) tuple where shape is Vec<i64>
        let shape = input_tensor.dim();
        let shape_vec = vec![
            shape.0 as i64,
            shape.1 as i64,
            shape.2 as i64,
            shape.3 as i64,
        ];
        let data_vec: Vec<f32> = input_tensor.iter().cloned().collect();
        let input_value = Value::from_array((shape_vec, data_vec))
            .map_err(|e| IcarusError::Model(format!("Failed to create input value: {}", e)))?;

        // Run inference using the ort::inputs! macro with the tensor value
        // The inputs! macro doesn't return a Result, so no ? operator needed
        let outputs = session
            .run(ort::inputs![input_value])
            .map_err(|e| IcarusError::Model(format!("ONNX inference failed: {}", e)))?;

        debug!("Inference completed, extracting outputs");

        // Parse outputs based on model type
        // SessionOutputs implements Deref<Target=BTreeMap<&str, DynValue>>
        match self.model_type {
            ModelType::YOLOv10 => {
                // YOLOv10 typically outputs: [batch, num_detections, 6] with [x, y, w, h, conf, class_id]
                // Access first output by index and extract as tensor
                let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
                    IcarusError::Model(format!("Failed to extract output tensor: {}", e))
                })?;

                // Reconstruct Array3 from shape and data slice
                let shape_vec: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                let output_tensor = match shape_vec.len() {
                    3 => {
                        let (b, n, f) = (shape_vec[0], shape_vec[1], shape_vec[2]);
                        Array3::from_shape_vec((b, n, f), data.to_vec()).map_err(|e| {
                            IcarusError::Model(format!("Failed to reshape output tensor: {}", e))
                        })?
                    }
                    _ => {
                        return Err(IcarusError::Model(format!(
                            "Unexpected output shape dimensions: {}",
                            shape_vec.len()
                        )))
                    }
                };

                debug!("YOLOv10 output shape: {:?}", output_tensor.dim());

                Ok(InferenceOutput::YOLOv10(output_tensor))
            }
            ModelType::DETR => {
                // DETR outputs multiple tensors: logits and boxes
                if outputs.len() < 2 {
                    return Err(IcarusError::Model(format!(
                        "DETR expected 2+ outputs, got {}",
                        outputs.len()
                    )));
                }

                // Extract logits and boxes from outputs by index
                let (logits_shape, logits_data) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| IcarusError::Model(format!("Failed to extract logits: {}", e)))?;
                let (boxes_shape, boxes_data) = outputs[1]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| IcarusError::Model(format!("Failed to extract boxes: {}", e)))?;

                // Reconstruct Array3s from shapes and data slices
                let logits_shape_vec: Vec<usize> =
                    logits_shape.iter().map(|&s| s as usize).collect();
                let boxes_shape_vec: Vec<usize> = boxes_shape.iter().map(|&s| s as usize).collect();

                let logits_tensor = match logits_shape_vec.len() {
                    3 => {
                        let (b, q, c) = (
                            logits_shape_vec[0],
                            logits_shape_vec[1],
                            logits_shape_vec[2],
                        );
                        Array3::from_shape_vec((b, q, c), logits_data.to_vec()).map_err(|e| {
                            IcarusError::Model(format!("Failed to reshape logits: {}", e))
                        })?
                    }
                    _ => {
                        return Err(IcarusError::Model(format!(
                            "Unexpected logits shape dimensions: {}",
                            logits_shape_vec.len()
                        )))
                    }
                };

                let boxes_tensor = match boxes_shape_vec.len() {
                    3 => {
                        let (b, q, f) =
                            (boxes_shape_vec[0], boxes_shape_vec[1], boxes_shape_vec[2]);
                        Array3::from_shape_vec((b, q, f), boxes_data.to_vec()).map_err(|e| {
                            IcarusError::Model(format!("Failed to reshape boxes: {}", e))
                        })?
                    }
                    _ => {
                        return Err(IcarusError::Model(format!(
                            "Unexpected boxes shape dimensions: {}",
                            boxes_shape_vec.len()
                        )))
                    }
                };

                debug!(
                    "DETR logits shape: {:?}, boxes shape: {:?}",
                    logits_tensor.dim(),
                    boxes_tensor.dim()
                );

                Ok(InferenceOutput::DETR {
                    logits: logits_tensor,
                    boxes: boxes_tensor,
                })
            }
        }
    }

    /// Run detection on an image and return the center of the highest-confidence person
    pub fn detect(&self, image: &DynamicImage) -> Result<DetectionResult, IcarusError> {
        info!(
            "Running detection on {}x{} image with {:?} model",
            image.width(),
            image.height(),
            self.model_type
        );

        // Preprocess image
        let preprocessed = preprocess_for_detection(image, self.input_size.0)?;
        debug!(
            "Image preprocessed: scale=({}, {}), padding=({}, {})",
            preprocessed.scale_x, preprocessed.scale_y, preprocessed.pad_x, preprocessed.pad_y
        );

        // Run inference and get output tensors
        let inference_output = match self.run_inference(preprocessed.tensor.clone()) {
            Ok(output) => output,
            Err(e) => {
                warn!("Inference failed: {}, falling back to image center", e);
                return Ok(DetectionResult {
                    center_x: image.width() as f32 / 2.0,
                    center_y: image.height() as f32 / 2.0,
                    bbox_xmin: 0.0,
                    bbox_ymin: 0.0,
                    bbox_xmax: 0.0,
                    bbox_ymax: 0.0,
                    confidence: 0.0,
                    has_person: false,
                });
            }
        };

        // Parse output and find person detection
        let result = match inference_output {
            InferenceOutput::YOLOv10(boxes) => {
                self.parse_yolov10_boxes(&boxes, image, &preprocessed)
            }
            InferenceOutput::DETR { logits, boxes } => {
                self.parse_detr_outputs(&logits, &boxes, image, &preprocessed)
            }
        };

        info!(
            "Detection result: has_person={}, confidence={:.4}",
            result.has_person, result.confidence
        );

        Ok(result)
    }

    /// Parse YOLOv10m output: [batch, num_boxes, 6] where values are [x, y, w, h, conf, class_id]
    /// Returns highest-confidence person detection or image center fallback
    fn parse_yolov10_boxes(
        &self,
        boxes: &Array3<f32>,
        original_image: &DynamicImage,
        preprocessed: &PreprocessedImage,
    ) -> DetectionResult {
        debug!(
            "Parsing YOLOv10m output boxes with shape: {:?}",
            boxes.dim()
        );

        // Find the highest confidence person (class_id = 0) detection
        let mut best_box: Option<(BoundingBox, f32)> = None;

        for box_idx in 0..boxes.dim().1 {
            let x = boxes[[0, box_idx, 0]];
            let y = boxes[[0, box_idx, 1]];
            let w = boxes[[0, box_idx, 2]];
            let h = boxes[[0, box_idx, 3]];
            let confidence = boxes[[0, box_idx, 4]];
            let class_id = boxes[[0, box_idx, 5]] as u32;

            // Only consider person detections (class_id = 0)
            if class_id != 0 {
                continue;
            }

            // Skip low confidence detections
            if confidence < self.confidence_threshold {
                debug!(
                    "Skipping detection with low confidence: {:.4} < {:.4}",
                    confidence, self.confidence_threshold
                );
                continue;
            }

            // Convert from center coords (x, y, w, h) to corner coords (xmin, ymin, xmax, ymax)
            // First subtract padding (to get coords in resized image space), then scale up
            let xmin = (x - preprocessed.pad_x - w / 2.0) / preprocessed.scale_x;
            let ymin = (y - preprocessed.pad_y - h / 2.0) / preprocessed.scale_y;
            let xmax = (x - preprocessed.pad_x + w / 2.0) / preprocessed.scale_x;
            let ymax = (y - preprocessed.pad_y + h / 2.0) / preprocessed.scale_y;

            let bbox = BoundingBox {
                xmin: xmin.max(0.0),
                ymin: ymin.max(0.0),
                xmax: xmax.min(original_image.width() as f32),
                ymax: ymax.min(original_image.height() as f32),
                confidence,
                class_id: Some(class_id),
            };

            // Track the highest confidence box
            if best_box.is_none() || confidence > best_box.as_ref().unwrap().1 {
                best_box = Some((bbox, confidence));
            }
        }

        // Return the best detection or fallback to image center
        match best_box {
            Some((bbox, conf)) => {
                info!(
                    "Found person detection: center=({:.2}, {:.2}), confidence={:.4}",
                    bbox.center_x(),
                    bbox.center_y(),
                    conf
                );
                DetectionResult {
                    center_x: bbox.center_x(),
                    center_y: bbox.center_y(),
                    bbox_xmin: bbox.xmin,
                    bbox_ymin: bbox.ymin,
                    bbox_xmax: bbox.xmax,
                    bbox_ymax: bbox.ymax,
                    confidence: conf,
                    has_person: true,
                }
            }
            None => {
                debug!("No person detections found above confidence threshold");
                DetectionResult {
                    center_x: original_image.width() as f32 / 2.0,
                    center_y: original_image.height() as f32 / 2.0,
                    bbox_xmin: 0.0,
                    bbox_ymin: 0.0,
                    bbox_xmax: 0.0,
                    bbox_ymax: 0.0,
                    confidence: 0.0,
                    has_person: false,
                }
            }
        }
    }

    /// Parse DETR output: Extracts logits and boxes from multiple output tensors
    /// DETR outputs: class_logits [batch, num_queries, num_classes], boxes [batch, num_queries, 4]
    /// where boxes are [cx, cy, w, h] in normalized coordinates [0.0, 1.0]
    fn parse_detr_outputs(
        &self,
        logits: &Array3<f32>,
        boxes: &Array3<f32>,
        original_image: &DynamicImage,
        _preprocessed: &PreprocessedImage,
    ) -> DetectionResult {
        debug!(
            "Parsing DETR output with logits shape: {:?}, boxes shape: {:?}",
            logits.dim(),
            boxes.dim()
        );

        let mut best_detection: Option<(BoundingBox, f32)> = None;

        for query_idx in 0..logits.dim().1.min(100) {
            // Get person class logit (softmax applied in preprocessing)
            // Person is typically class_id = 0 in COCO
            let person_logit = logits[[0, query_idx, 0]];

            // Skip if confidence below threshold
            if person_logit < self.confidence_threshold {
                continue;
            }

            // Get box coordinates [cx, cy, w, h] in normalized coordinates
            let cx = boxes[[0, query_idx, 0]];
            let cy = boxes[[0, query_idx, 1]];
            let w = boxes[[0, query_idx, 2]];
            let h = boxes[[0, query_idx, 3]];

            // Convert from normalized coordinates to pixel coordinates
            let scale_w = original_image.width() as f32;
            let scale_h = original_image.height() as f32;

            let xmin = ((cx - w / 2.0) * scale_w).max(0.0);
            let ymin = ((cy - h / 2.0) * scale_h).max(0.0);
            let xmax = ((cx + w / 2.0) * scale_w).min(scale_w);
            let ymax = ((cy + h / 2.0) * scale_h).min(scale_h);

            let bbox = BoundingBox {
                xmin,
                ymin,
                xmax,
                ymax,
                confidence: person_logit,
                class_id: Some(0),
            };

            if best_detection.is_none() || person_logit > best_detection.as_ref().unwrap().1 {
                best_detection = Some((bbox, person_logit));
            }
        }

        match best_detection {
            Some((bbox, conf)) => {
                info!(
                    "Found person detection: center=({:.2}, {:.2}), confidence={:.4}",
                    bbox.center_x(),
                    bbox.center_y(),
                    conf
                );
                DetectionResult {
                    center_x: bbox.center_x(),
                    center_y: bbox.center_y(),
                    bbox_xmin: bbox.xmin,
                    bbox_ymin: bbox.ymin,
                    bbox_xmax: bbox.xmax,
                    bbox_ymax: bbox.ymax,
                    confidence: conf,
                    has_person: true,
                }
            }
            None => {
                debug!("No person detections found above confidence threshold");
                DetectionResult {
                    center_x: original_image.width() as f32 / 2.0,
                    center_y: original_image.height() as f32 / 2.0,
                    bbox_xmin: 0.0,
                    bbox_ymin: 0.0,
                    bbox_xmax: 0.0,
                    bbox_ymax: 0.0,
                    confidence: 0.0,
                    has_person: false,
                }
            }
        }
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
        assert_eq!(result.bbox_xmin, 0.0);
        assert_eq!(result.bbox_ymin, 0.0);
        assert_eq!(result.bbox_xmax, 0.0);
        assert_eq!(result.bbox_ymax, 0.0);
        assert_eq!(result.confidence, 0.0);
        assert!(!result.has_person);
    }

    #[test]
    fn test_detection_result_creation() {
        let result = DetectionResult {
            center_x: 100.0,
            center_y: 150.0,
            bbox_xmin: 50.0,
            bbox_ymin: 100.0,
            bbox_xmax: 150.0,
            bbox_ymax: 200.0,
            confidence: 0.95,
            has_person: true,
        };
        assert_eq!(result.center_x, 100.0);
        assert_eq!(result.center_y, 150.0);
        assert_eq!(result.bbox_xmin, 50.0);
        assert_eq!(result.bbox_ymin, 100.0);
        assert_eq!(result.bbox_xmax, 150.0);
        assert_eq!(result.bbox_ymax, 200.0);
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

    #[test]
    fn test_parse_yolov10_boxes_with_person_detection() {
        use crate::image_io::PreprocessedImage;
        use image::DynamicImage;
        use ndarray::Array3;

        // Create a detector instance
        let detector = OnnxDetector {
            model_path: std::path::PathBuf::from("test.onnx"),
            model_type: ModelType::YOLOv10,
            input_size: (640, 640),
            confidence_threshold: 0.5,
            id2label: default_id2label(),
        };

        // Create test image
        let test_image = DynamicImage::new_rgb8(640, 480);

        // Create test boxes: [batch=1, num_boxes=2, features=6]
        let mut boxes = Array3::zeros((1, 2, 6));

        // First box: person with high confidence at (320, 240) in preprocessed space
        // With scale=1.0, pad_x=0, pad_y=80, the detection is at the image center
        boxes[[0, 0, 0]] = 320.0; // x in preprocessed space
        boxes[[0, 0, 1]] = 320.0; // y in preprocessed space (240 + 80 padding)
        boxes[[0, 0, 2]] = 100.0; // w
        boxes[[0, 0, 3]] = 200.0; // h
        boxes[[0, 0, 4]] = 0.9; // confidence
        boxes[[0, 0, 5]] = 0.0; // class_id (person)

        // Second box: non-person with high confidence (should be ignored)
        boxes[[0, 1, 0]] = 100.0;
        boxes[[0, 1, 1]] = 100.0;
        boxes[[0, 1, 2]] = 50.0;
        boxes[[0, 1, 3]] = 50.0;
        boxes[[0, 1, 4]] = 0.95; // high confidence but wrong class
        boxes[[0, 1, 5]] = 5.0; // class_id (not person)

        // Preprocessed image: scale=1.0, pad_y=80 (for centering 640x480 in 640x640)
        let preprocessed = PreprocessedImage {
            tensor: Array4::zeros((1, 3, 640, 640)),
            scale_x: 1.0,
            scale_y: 1.0,
            pad_x: 0.0,
            pad_y: 80.0,
        };

        let result = detector.parse_yolov10_boxes(&boxes, &test_image, &preprocessed);

        assert!(result.has_person);
        assert!(result.confidence > 0.85);
        // With pad_y=80, a detection at y=320 should translate to y=240 in original image
        assert!(result.center_x > 250.0 && result.center_x < 390.0);
        assert!(result.center_y > 150.0 && result.center_y < 330.0); // Adjusted for padding
    }

    #[test]
    fn test_parse_yolov10_boxes_low_confidence_filtered() {
        use crate::image_io::PreprocessedImage;
        use image::DynamicImage;
        use ndarray::Array3;

        let mut detector = OnnxDetector {
            model_path: std::path::PathBuf::from("test.onnx"),
            model_type: ModelType::YOLOv10,
            input_size: (640, 640),
            confidence_threshold: 0.7,
            id2label: default_id2label(),
        };

        let test_image = DynamicImage::new_rgb8(640, 480);
        let mut boxes = Array3::zeros((1, 1, 6));

        // Person detection with confidence below threshold
        boxes[[0, 0, 0]] = 320.0;
        boxes[[0, 0, 1]] = 240.0;
        boxes[[0, 0, 2]] = 100.0;
        boxes[[0, 0, 3]] = 200.0;
        boxes[[0, 0, 4]] = 0.5; // below threshold of 0.7
        boxes[[0, 0, 5]] = 0.0;

        let preprocessed = PreprocessedImage {
            tensor: Array4::zeros((1, 3, 640, 640)),
            scale_x: 1.0,
            scale_y: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        };

        let result = detector.parse_yolov10_boxes(&boxes, &test_image, &preprocessed);

        assert!(!result.has_person);
        assert_eq!(result.confidence, 0.0);
        // Should return image center
        assert!((result.center_x - 320.0).abs() < 1.0);
        assert!((result.center_y - 240.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_detr_outputs_with_person_detection() {
        use crate::image_io::PreprocessedImage;
        use image::DynamicImage;
        use ndarray::Array3;

        let detector = OnnxDetector {
            model_path: std::path::PathBuf::from("test.onnx"),
            model_type: ModelType::DETR,
            input_size: (800, 800),
            confidence_threshold: 0.3,
            id2label: default_id2label(),
        };

        let test_image = DynamicImage::new_rgb8(800, 600);

        // Create test logits and boxes
        let mut logits = Array3::zeros((1, 10, 1)); // 10 queries, 1 class (person only)
        let mut boxes = Array3::zeros((1, 10, 4));

        // First detection: person with high confidence at center
        logits[[0, 0, 0]] = 0.85; // high confidence for person
        boxes[[0, 0, 0]] = 0.5; // cx (normalized)
        boxes[[0, 0, 1]] = 0.5; // cy
        boxes[[0, 0, 2]] = 0.2; // w
        boxes[[0, 0, 3]] = 0.4; // h

        let preprocessed = PreprocessedImage {
            tensor: Array4::zeros((1, 3, 800, 800)),
            scale_x: 1.0,
            scale_y: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        };

        let result = detector.parse_detr_outputs(&logits, &boxes, &test_image, &preprocessed);

        assert!(result.has_person);
        assert!(result.confidence > 0.8);
        // Center should be roughly at image center (0.5 * width, 0.5 * height)
        assert!(result.center_x > 300.0 && result.center_x < 500.0);
        assert!(result.center_y > 200.0 && result.center_y < 400.0);
    }

    #[test]
    fn test_parse_detr_outputs_no_detection() {
        use crate::image_io::PreprocessedImage;
        use image::DynamicImage;
        use ndarray::Array3;

        let detector = OnnxDetector {
            model_path: std::path::PathBuf::from("test.onnx"),
            model_type: ModelType::DETR,
            input_size: (800, 800),
            confidence_threshold: 0.5,
            id2label: default_id2label(),
        };

        let test_image = DynamicImage::new_rgb8(800, 600);

        // All detections below threshold
        let logits = Array3::zeros((1, 10, 1)); // all zeros (low confidence)
        let boxes = Array3::zeros((1, 10, 4));

        let preprocessed = PreprocessedImage {
            tensor: Array4::zeros((1, 3, 800, 800)),
            scale_x: 1.0,
            scale_y: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        };

        let result = detector.parse_detr_outputs(&logits, &boxes, &test_image, &preprocessed);

        assert!(!result.has_person);
        assert_eq!(result.confidence, 0.0);
        // Should return image center
        assert!((result.center_x - 400.0).abs() < 1.0);
        assert!((result.center_y - 300.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_yolov10_boxes_with_scaling_and_padding() {
        use crate::image_io::PreprocessedImage;
        use image::DynamicImage;
        use ndarray::Array3;

        // Create a detector instance
        let detector = OnnxDetector {
            model_path: std::path::PathBuf::from("test.onnx"),
            model_type: ModelType::YOLOv10,
            input_size: (640, 640),
            confidence_threshold: 0.5,
            id2label: default_id2label(),
        };

        // Create test image: 2000x1500 (landscape)
        let test_image = DynamicImage::new_rgb8(2000, 1500);

        // Simulate preprocessing:
        // - scale = 640 / 2000 = 0.32
        // - resized = 640x480
        // - padding = (0, 80) for vertical centering
        // So detection in preprocessed space at (320, 240+80=320) should map back to image center (1000, 750)

        let mut boxes = Array3::zeros((1, 1, 6));
        // Person at center of preprocessed image
        boxes[[0, 0, 0]] = 320.0; // x in 640x640 space
        boxes[[0, 0, 1]] = 320.0; // y in 640x640 space (240 + 80 padding)
        boxes[[0, 0, 2]] = 100.0; // w
        boxes[[0, 0, 3]] = 150.0; // h
        boxes[[0, 0, 4]] = 0.9; // confidence
        boxes[[0, 0, 5]] = 0.0; // class_id (person)

        let preprocessed = PreprocessedImage {
            tensor: Array4::zeros((1, 3, 640, 640)),
            scale_x: 0.32,
            scale_y: 0.32,
            pad_x: 0.0,
            pad_y: 80.0,
        };

        let result = detector.parse_yolov10_boxes(&boxes, &test_image, &preprocessed);

        assert!(result.has_person);
        // Detection should be at the center of the original image
        // x: (320 - 0) / 0.32 ≈ 1000
        // y: (320 - 80) / 0.32 ≈ 750
        assert!((result.center_x - 1000.0).abs() < 50.0, "Expected center_x ≈ 1000, got {}", result.center_x);
        assert!((result.center_y - 750.0).abs() < 50.0, "Expected center_y ≈ 750, got {}", result.center_y);
    }
}
