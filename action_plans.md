# Action Plan for Real-time 2D Video to 3D Point-Cloud Converter

Based on `instructions.md` (PRD Version 1.1).

## Stage 1: Project Setup and Core Dependencies
*   **Goal:** Initialize the Python project structure, set up dependency management with `uv`, and install core libraries for image processing and numerical operations.
*   **Details:**
    *   Create `pyproject.toml` for project metadata and dependencies.
    *   Initialize `uv` for Python package management (`uv pip init`).
    *   Add `numpy` (for numerical operations) and `opencv-python` (for image loading/manipulation) as dependencies.
    *   Create a basic directory structure (e.g., `src/vide_to_3d_converter`, `tests/`).
*   **Reviewable/Testable:**
    *   `action_plans.md` contains this stage description.
    *   `pyproject.toml` is created and configured.
    *   `uv lock` and `uv pip sync` (or `uv venv && source .venv/bin/activate && uv pip install numpy opencv-python`) complete without errors.
    *   A test script can successfully `import numpy` and `import cv2`.
    *   Basic project directory structure is in place.

## Stage 2: Depth Estimation Module
*   **Goal:** Implement the monocular depth estimation using a pre-trained model (MiDaS or Depth-Anything).
*   **Details:**
    *   Research and select suitable pre-trained models from the MiDaS family or Depth-Anything. Prioritize models with good performance/accuracy trade-offs that can run on an RTX 4080.
    *   Integrate the chosen depth estimation model(s) into the project. This will likely involve adding new dependencies (e.g., `timm`, `torch`, `torchvision`).
    *   Create a Python module (e.g., `src/video_to_3d_converter/depth_estimator.py`).
    *   Implement a function `load_rgb_frame(image_path: str) -> np.ndarray` that reads an image using OpenCV.
    *   Implement a core function `estimate_depth(rgb_frame: np.ndarray, model_variant: str) -> np.ndarray`.
    *   This function should:
        *   Preprocess the frame to match the chosen model's input requirements (resize, normalize).
        *   Perform inference using the selected model variant.
        *   Post-process the output to get a depth map (e.g., resize to original frame dimensions if necessary).
    *   Support at least two model variants (e.g., a faster one and a more accurate one) as per FR3. This could be handled by a parameter in `estimate_depth`.
    *   Ensure the module can handle various input resolutions (VGA, HD, FullHD) as per FR7, potentially by resizing internally or ensuring model compatibility.
*   **Reviewable/Testable:**
    *   `action_plans.md` includes this stage description.
    *   New dependencies (e.g., `torch`, `timm`) are added to `pyproject.toml` and installable via `uv pip install -e .`.
    *   `depth_estimator.py` contains the specified functions.
    *   Unit tests in `tests/test_depth_estimator.py` for:
        *   Loading a sample RGB image.
        *   Successfully running `estimate_depth` with at least two model variants on a test image.
        *   Verifying that the output depth map is a 2D `np.ndarray` with expected dimensions.
    *   Visual inspection of an output depth map from a sample image shows plausible depth estimation (e.g., closer objects are brighter/darker depending on convention).

## Stage 3: Point Cloud Generation
*   **Goal:** Convert the depth map and RGB frame into an XYZRGB point cloud and save it.
*   **Details:**
    *   Create a new Python module (e.g., `src/video_to_3d_converter/point_cloud_generator.py`).
    *   Implement a function `depth_to_points(depth_map: np.ndarray, intrinsics: dict) -> np.ndarray`.
        *   This function will take a 2D depth map and camera intrinsics (`fx`, `fy`, `cx`, `cy`).
        *   It will generate a 3D point cloud (Nx3 array for XYZ coordinates) using the pinhole camera model un-projection formula (FR4, FR5).
            *   `X = (u - cx) * Z / fx`
            *   `Y = (v - cy) * Z / fy`
            *   `Z = depth_map[v, u]`
    *   Implement a function `combine_xyz_rgb(points_xyz: np.ndarray, rgb_frame: np.ndarray, intrinsics: dict) -> np.ndarray`.
        *   This function will take the Nx3 XYZ point cloud, the original RGB frame, and the camera intrinsics.
        *   It will sample colors from the RGB frame corresponding to each point to create an Nx6 XYZRGB point cloud (FR6). Ensure that points outside the valid image area (if any due to distortion or projection) are handled.
    *   Implement a function `save_pcd(point_cloud_xyzrgb: np.ndarray, filepath: str)`.
        *   This function will save the Nx6 point cloud to a file in binary PCD format (little-endian, float32, XYZRGB fields) as per FR8. This will require writing the PCD header and then the binary data.
        *   PCD Header Example:
            ```
            # .PCD v0.7 - Point Cloud Data file format
            VERSION 0.7
            FIELDS x y z rgb
            SIZE 4 4 4 4
            TYPE F F F F
            COUNT 1 1 1 1
            WIDTH num_points
            HEIGHT 1
            VIEWPOINT 0 0 0 1 0 0 0
            POINTS num_points
            DATA binary
            ```
        *   The RGB data needs to be packed into a single float32. This is often done by shifting bits: `(r << 16) | (g << 8) | b`.
    *   Consider adding a dependency like `pypcd` or `open3d` if writing binary PCD files manually becomes too complex, but the PRD implies direct implementation. For now, attempt manual creation.
*   **Reviewable/Testable:**
    *   `action_plans.md` includes this stage description.
    *   `point_cloud_generator.py` contains the specified functions.
    *   New dependencies (if any, like `pypcd` or `open3d`) are added to `pyproject.toml`.
    *   Unit tests in `tests/test_point_cloud_generator.py` for:
        *   `depth_to_points`: Given a sample depth map and intrinsics, verify the shape and some sample XYZ values.
        *   `combine_xyz_rgb`: Given sample XYZ points and an RGB frame, verify the shape of the output and that RGB values are correctly assigned.
        *   `save_pcd` and `load_pcd` (a helper for testing): Save a sample point cloud and then load it back, verifying the data integrity. Check PCD header fields.
    *   Visual inspection of a generated PCD file (e.g., using `pcl_viewer` or Open3D) from a sample depth map shows a recognizable 3D structure.

## Stage 4: Performance Optimization and Temporal Smoothing
*   **Goal:** Optimize the pipeline for performance and latency, and implement an optional temporal smoothing filter.
*   **Details:**
    *   **Benchmarking and Profiling:**
        *   Develop a benchmarking script (e.g., `scripts/benchmark.py`) that processes a sample video or a sequence of frames through the full pipeline (load RGB -> estimate depth -> generate point cloud -> save PCD).
        *   Measure end-to-end throughput (FPS) and latency (ms per frame) for different resolutions (especially 1080p) as per NFR1, NFR2.
        *   Use profiling tools (e.g., `cProfile`, `torch.profiler`, `py-spy`) to identify bottlenecks in the Python code.
        *   Assess GPU utilization and VRAM usage (NFR4), e.g., using `nvidia-smi`.
    *   **Optimization:**
        *   Based on profiling, optimize critical code sections. This might involve:
            *   Algorithm improvements (e.g., more efficient NumPy operations).
            *   Ensuring data is efficiently transferred between CPU and GPU.
            *   For depth estimation, ensure the model is running optimally on the GPU (e.g., using TensorRT for NVIDIA models if applicable, though MiDaS/Depth-Anything might need specific paths for this). The PRD mentions TensorRT-LLM for future work, implying core TensorRT is an option.
        *   If Python optimizations are insufficient, identify "hot loops" for potential C++/CUDA implementation (Scope 4.1). Setup `vcpkg` for C++ dependency management if this path is taken. For this stage, we will first focus on Python-level and PyTorch optimizations.
    *   **Temporal Smoothing:**
        *   Create a new module (e.g., `src/video_to_3d_converter/temporal_smoother.py`).
        *   Implement at least one temporal consistency filter for point clouds or depth maps (e.g., Exponential Moving Average - EMA on depth maps or per-point XYZ coordinates).
        *   The filter should be optional and configurable.
        *   Ensure the added latency from smoothing is ≤ 33ms (NFR2).
        *   The smoother should aim to reduce flickering or jitter in the output point clouds between frames.
*   **Reviewable/Testable:**
    *   `action_plans.md` includes this stage description.
    *   Benchmarking script `scripts/benchmark.py` is created and functional.
    *   Profiling reports or summaries are available, identifying key bottlenecks.
    *   Optimization efforts are documented (e.g., code changes, new techniques applied).
    *   `temporal_smoother.py` module is created with a functional smoothing filter.
    *   Unit tests in `tests/test_temporal_smoother.py` for the smoothing logic (e.g., applying EMA to a sequence of depth maps and verifying the output).
    *   Performance metrics after optimization:
        *   Throughput ≥ 30 FPS @ 1080p on RTX 4080 (NFR1).
        *   End-to-end latency ≤ 100ms (NFR2), with smoothing filter latency ≤ 33ms.
        *   GPU utilization and VRAM usage within limits (NFR4).
    *   Visual inspection of output with and without temporal smoothing shows improved consistency.

## Stage 5: Packaging, API, Error Handling, and Documentation
*   **Goal:** Deliver the module as a robust, well-documented, and importable Python package with a stable API and proper error handling.
*   **Details:**
    *   **Public API Definition:**
        *   Define the main public interface for the `video_to_3d_converter` package. This might be a facade class or a few key functions in `src/video_to_3d_converter/__init__.py`.
        *   Example API:
            ```python
            # src/video_to_3d_converter/__init__.py
            class Converter:
                def __init__(self, model_variant: str, intrinsics: dict, use_temporal_smoother: bool = False, smoothing_alpha: float = 0.5):
                    # Initialize depth estimator, (optional) smoother
                    pass

                def process_rgb_frame(self, rgb_frame: np.ndarray) -> np.ndarray:
                    # Run full pipeline: depth -> (smooth) -> points_xyz -> combine_rgb
                    # Returns Nx4 XYZRGB point cloud
                    pass

                def process_image_file(self, image_path: str) -> np.ndarray:
                    # Load image, then call process_rgb_frame
                    pass

            # Utility functions might also be exposed if needed.
            ```
    *   **Error Handling (FR9):**
        *   Define structured JSON error codes for recoverable faults (e.g., `E_MODEL_LOAD`, `E_OOM`, `E_NO_INPUT`).
        *   Create custom exception classes (e.g., `ConverterError(Exception)`) that can store these error codes and messages.
        *   Integrate this error handling into the core modules (depth estimator, point cloud generator, API).
        *   Example JSON error structure: `{"error_code": "E_MODEL_LOAD", "message": "Failed to load depth model variant X."}`
    *   **Packaging (NFR5):**
        *   Ensure `pyproject.toml` is complete for `pip install -e .` and potentially for building a wheel.
        *   Add a basic `README.md` at the project root with setup and usage instructions.
    *   **Documentation & Linting (NFR7):**
        *   Write comprehensive docstrings for all public modules, classes, and functions (target ≥ 80% coverage). Tools like `coverage.py` (for code) and `interrogate` (for docstrings) can measure this.
        *   Ensure code conforms to PEP 8 using `ruff format .` and `ruff check .`. Add `ruff` configuration to `pyproject.toml` or `ruff.toml` if specific rules need adjustment.
    *   **Stability & Security (NFR6, NFR8) - Partial Implementation/Guidance:**
        *   **Stability (NFR6):** The benchmark script can be adapted into a soak test (run continuously for 2 hours, monitoring for crashes/leaks). Actual memory leak detection for Python often uses `tracemalloc` or external tools; C++ parts would need `valgrind`. This subtask will note the requirement; full execution of soak tests is beyond typical CI subtask scope but the script should be runnable.
        *   **Security (NFR8):**
            *   SBOM (Software Bill of Materials): Can be generated using tools like `cyclonedx-python`.
            *   Dependency Scanning: Tools like `pip-audit` or GitHub's Dependabot.
            *   Camera Access Sandboxing: This is more of an OS/deployment concern, note it in documentation.
            *   For this stage, we will ensure `ruff` linting includes security rules (e.g., via `ruff-flake8-security`) and add `pip-audit` to dev dependencies.
*   **Reviewable/Testable:**
    *   `action_plans.md` includes this stage description.
    *   Public API is defined in `src/video_to_3d_converter/__init__.py` and is usable.
    *   Custom exceptions and error handling (FR9) are implemented and can be triggered.
    *   Project `README.md` exists with basic instructions.
    *   Docstring coverage meets or approaches the ≥ 80% target.
    *   Code passes `ruff format` and `ruff check` (including security linters).
    *   The package can be installed in editable mode.
    *   A basic soak test script (modified benchmark script) is available.
    *   `pip-audit` is added, and guidance for SBOM generation is noted.
