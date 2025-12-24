# AI-Powered Photo Cropping Plan (Refined)

## Objectives
- Deliver a CLI that automatically crops photos around detected people using ONNX models with desktop, mobile, and landscape presets.
- Keep cropping decisions deterministic by exposing configurable detection thresholds, center weighting, and aspect-specific clamps while surfacing thoughtful defaults.
- Provide first-class CLI workflows for single files, directories, recursive batches, auto/visualization modes, and model management, backed by solid documentation and metrics.

## Feature Scope
### Detection and Model Management
- Integrate ONNX runtime for YOLOv8, YOLO-NAS, and R-CNN models while allowing users to swap in custom ONNX files.
- Detect only the `person` class with configurable confidence and NMS thresholds, and expose model listing/downloading commands for local caching.

### Cropping and Aspect Ratio Handling
- Support portrait (9:16), mobile (9:20), and landscape (21:9) presets plus the ability to generate every aspect via `--all-aspects`.
- Center crops on the computed point of focus (single-person center, weighted multi-person center, or image center fallback) while honoring `center_weight` config setting.
- Clamp crops to the source image bounds, respect automatic minimum dimension requirements (derived from source image), and output in user-selected formats/quality levels.

### CLI Experience
- Commands cover single image, directory, recursive traversal, dry-run, visualization, and model management workflows.
- Auto-aspect heuristics choose the best crop ratio per image, while `--all-aspects` mode generates all valid aspect ratios (organized into subfolders: `portrait/`, `mobile/`, `landscape/` preserving original filenames) based on automatic dimension validation.
- Model downloads use hardcoded URLs for curated, supported models with basic retry and validation logic.
- Verbosity, logging, progress reporting, and error handling give users insight into detection/cropping decisions, and `--dry-run`/`--visualize` keep the pipeline transparent.

## Architecture and Pipeline
- Image loading handles JPEG, PNG, WebP, TIFF formats, reads EXIF orientation, and keeps data in memory-efficient buffers.
- Preprocessing resizes inputs to model dimensions, normalizes pixels, and prepares batches; inference runs with optimized ONNX sessions; postprocessing filters for person boxes and converts coordinates back to the source frame.
- Cropping derives `CropParams` from selected `AspectRatio`, computes center points (including weighted centers for multi-person scenes), and ensures the final crop rectangle is valid against the source dimensions.
- Configuration (TOML) drives general settings, CPU-only model parameters (MVP scope), cropping/parsing rules, aspect ratio definitions, and `center_weight` mode; CLI flags override config defaults as needed.
- Error handling distinguishes between image/model/detection/cropping failures and opts for graceful degradation (warnings, skip, fallback crops, or CLI hints).

## Phased Implementation
1. **Core infrastructure**: set up CLI entrypoint, clap-based parsing, configuration loader, logging, basic error types, and image I/O utilities.
2. **ONNX integration**: initialize runtime sessions, load/validate models, build preprocessing pipelines, run inference, and postprocess detections for `person` items.
3. **Detection and cropping logic**: extract bounding boxes, calculate centers (single/multi-person), resolve aspect ratios, clamp/adjust crop regions, and implement center-weight fallbacks.
4. **CLI polish and productivity**: support single/batch/recursive commands, add progress reporting, surface `--dry-run`/`--visualize`, and wire up model-management commands (`models list`/`download`).
5. **Advanced workflows**: introduce auto-aspect heuristics, multi-aspect `--all-aspects`, caching/performance optimizations (rayon for batches, session caching), and export options (quality, format, output directories).
6. **Testing documentation**: author unit/integration/benchmark suites, collect sample fixtures, draft user/developer docs, and summarize release instructions (download OS binaries, ONNX Runtime prerequisites).

## Testing and Validation
- Unit coverage for preprocessing, aspect ratio math, center-weight vectors, configuration parsing, and error conversions.
- Integration tests spanning single/multiple/no-person scenes, different image formats, recursive directories, and boundary/clamp edge cases.
- Benchmarks targeting inference latency, throughput, memory usage, and batch scaling.
- Acceptance criteria documenting behavior for each CLI command, auto-aspect decisions, and visualization output (logs/screenshots).

## Operational Considerations
- Performance targets: <500ms per image on CPU and 10+ images/sec in batch via rayon and session reuse (CPU-only for MVP; GPU support documented as future enhancement).
- Resource guardrails: cap image dimensions/batch size, timeout inference, and skip corrupted assets with helpful warnings.
- Security: sanitize input paths/types, validate ONNX payloads, and document licensing obligations for downstream models.
- Deployment: publish Linux/Windows binaries, hardcode curated model download URLs with basic retry logic, and detail ONNX Runtime CPU installation per platform.
- Output organization: `--all-aspects` creates aspect-specific subdirectories (`portrait/`, `mobile/`, `landscape/`) preserving original filenames, with automatic minimum dimension validation determining which aspects are generated.

## Design Decisions (Finalized)
1. **GPU Support**: MVP is CPU-only with ONNX Runtime. GPU providers (CUDA, TensorRT) are documented as future enhancements. Configuration `[model] provider` defaults to `"cpu"`.
2. **Mobile Aspect Ratio**: Standardized on **9:20** for all mobile cropping calculations and documentation.
3. **Center Weight Mode**: Single configuration-driven mode (`area`, `confidence`, or `equal`) set in `[cropping] center_weight`. No per-run CLI overrides in MVP. Users modify config file to change behavior.
4. **Multi-Aspect Output Organization**: `--all-aspects` generates separate subdirectories (`portrait/`, `mobile/`, `landscape/`) within the output directory, preserving original filenames. Minimum dimension thresholds are automatically derived from source image dimensions (e.g., requires 75% of target aspect dimensions). Only aspects that meet minimum thresholds are generated.
5. **Model Management**: `models download` uses hardcoded URLs for curated, supported models (YOLOv8n, YOLO-NAS variants). Basic HTTP downloads with retry logic and progress bars. No authenticated HuggingFace API support in MVP—users can manually download and place custom models in the models directory.
