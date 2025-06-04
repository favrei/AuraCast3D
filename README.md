# Real-time 2D Video to 3D Point-Cloud Converter

This module converts a monocular RGB video stream (or individual image frames)
of a conference participant into a temporally consistent 3-D point cloud in real time.

## Project Status

This project is currently under development. Key features include:
- Depth estimation using MiDaS models (DPT_Large, MiDaS_small).
- Point cloud generation (XYZRGB format).
- Saving point clouds in binary PCD format.
- Optional temporal smoothing (Exponential Moving Average).
- Basic benchmarking script.
- Public API via `Converter` class.

## Setup

This project uses `uv` for package management and Python 3.11+.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
    uv pip install -e ".[dev]"
    ```
    The `[dev]` option includes development tools like `ruff`, `pytest`, and `pip-audit`.

## Usage

The primary interface is the `Converter` class from the `video_to_3d_converter` package.

```python
from video_to_3d_converter import Converter, ConverterError, save_pcd # save_pcd also exposed

# Example camera intrinsics (replace with actual values for your setup)
# For a 1920x1080 resolution image:
intrinsics = {'fx': 1000.0, 'fy': 1000.0, 'cx': (1920-1)/2.0, 'cy': (1080-1)/2.0}

try:
    # Initialize converter (choose model, intrinsics, optional smoothing)
    converter = Converter(model_variant="MiDaS_small", intrinsics=intrinsics, use_temporal_smoother=True)

    # Process an image file
    image_path = "sample_frames/frame_0000.png" # Create this dummy file or use your own
    # (Ensure sample_frames directory exists and has an image if running this example)
    # You can create dummy frames using: scripts/benchmark.py --num_frames 1 --resolution 1080p

    # Check if the dummy image exists, if not, maybe skip or guide user
    import os
    if not os.path.exists(image_path):
        print(f"Test image {image_path} not found. Create dummy frames using benchmark script or provide valid path.")
        # Example: Create dummy frames if sample_frames is empty and image_path points there
        if image_path.startswith("sample_frames/") and not os.listdir("sample_frames" if os.path.exists("sample_frames") else "."):
             print("Attempting to create dummy frames for example...")
             os.system(f"uv run python scripts/benchmark.py --num_frames 1 --resolution 1080p --image_dir sample_frames")


    if os.path.exists(image_path):
        point_cloud = converter.process_image_file(image_path)

        if point_cloud.shape[0] > 0:
            print(f"Generated point cloud with {point_cloud.shape[0]} points.")
            output_pcd_path = "output.pcd"
            save_pcd(point_cloud, output_pcd_path)
            print(f"Saved point cloud to {output_pcd_path}")
        else:
            print("No points generated for the image.")
    else:
        print(f"Skipping processing for {image_path} as it was not found/created.")


    # To process a raw RGB frame (NumPy array HxWxC):
    # import numpy as np
    # dummy_rgb_frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    # point_cloud_from_frame = converter.process_rgb_frame(dummy_rgb_frame)
    # if point_cloud_from_frame.shape[0] > 0:
    #    print(f"Generated point cloud with {point_cloud_from_frame.shape[0]} points from raw frame.")


except ConverterError as e:
    print(f"A converter error occurred: {e.to_json()}")
except Exception as e:
    print(f"A general exception occurred: {e}")

```

## Development

-   **Linting & Formatting:** Run `uv run ruff format . && uv run ruff check . --fix`
-   **Tests:** Run `uv run python -m unittest discover tests`
-   **Benchmarking:** See `scripts/benchmark.py`. Configure and run with `uv run python scripts/benchmark.py [args]`.
-   **Security Audit:** Run `pip-audit` (after installing dev dependencies).

## TODO / Future Work (from PRD)
- Detailed performance tuning for RTX 4080 to meet 30 FPS @ 1080p.
- C++/CUDA kernels for hot loops if Python isn't fast enough.
- More sophisticated temporal smoothing.
- Person-specific fine-tuning of depth models.
- TensorRT integration.
- SBOM generation and regular dependency scanning setup.

This README provides a basic overview. See docstrings for more detailed API documentation.
