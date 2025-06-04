import argparse  # For command-line arguments
import os
import time

import numpy as np
from PIL import Image

# Assuming modules are accessible via editable install or PYTHONPATH
from video_to_3d_converter.depth_estimator import _load_model, estimate_depth, load_rgb_frame
from video_to_3d_converter.point_cloud_generator import combine_xyz_rgb, depth_to_points, save_pcd
from video_to_3d_converter.temporal_smoother import ExponentialMovingAverageSmoother

# --- Default Configuration ---
DEFAULT_IMAGE_DIR = "sample_frames/"
DEFAULT_OUTPUT_DIR = "benchmark_output/"
DEFAULT_NUM_FRAMES = 10
DEFAULT_MODEL_VARIANT = "MiDaS_small"
DEFAULT_RESOLUTION_MODE = "1080p"  # (1920x1080). Others: "720p" (1280x720), "VGA" (640x480)
DEFAULT_CAMERA_INTRINSICS = {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 959.5,
    "cy": 539.5,
}  # cx=(W-1)/2, cy=(H-1)/2 for 1920x1080


# --- Helper Functions ---
def get_image_resolution(resolution_mode_str):
    if resolution_mode_str == "1080p":
        return 1920, 1080
    if resolution_mode_str == "720p":
        return 1280, 720
    if resolution_mode_str == "VGA":
        return 640, 480
    try:
        # Custom format WxH
        w, h = map(int, resolution_mode_str.split("x"))
        return w, h
    except ValueError:
        print(f"Warning: Invalid resolution string '{resolution_mode_str}'. Defaulting to 1080p.")
        return 1920, 1080


def get_intrinsics_for_resolution(width, height):
    # Basic assumption: FoV is roughly constant, fx, fy scale with width/height
    # And cx, cy are at the center. This is a simplification.
    # Example: If 1080p (1920x1080) has fx=1000, then for VGA (640x480) fx_vga = 1000 * (640/1920)
    scale_factor_w = width / 1920.0
    # scale_factor_h = height / 1080.0 # Not used if fx=fy and aspect ratio maintained by model

    # Using a common assumption that fx and fy are similar
    # Adjust fx, fy based on the ratio to a reference resolution (e.g. 1080p width)
    # This is a heuristic. Real intrinsics depend on the actual camera.
    ref_fx = 1000.0

    return {
        "fx": ref_fx * scale_factor_w,
        "fy": ref_fx * scale_factor_w,  # Assuming square pixels and fx approx fy after scaling
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }


def create_dummy_frames(image_dir, num_frames, width=1920, height=1080):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    print(f"Ensuring {num_frames} dummy frames in {image_dir} ({width}x{height})...")
    for i in range(num_frames):
        img_path = os.path.join(image_dir, f"frame_{i:04d}.png")
        if not os.path.exists(img_path):
            img = Image.new(
                "RGB",
                (width, height),
                color=(
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                ),
            )
            img.save(img_path)
    print(f"Dummy frames ready in {image_dir}.")


def list_frames(image_dir):
    return sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))]
    )


# --- Main Pipeline ---
def process_frame(image_path, model_variant, intrinsics, output_dir, smoother=None):
    # 1. Load RGB Frame
    t_load_start = time.perf_counter()
    rgb_frame = load_rgb_frame(image_path)
    t_load_end = time.perf_counter()
    load_time = (t_load_end - t_load_start) * 1000

    # 2. Estimate Depth
    t_depth_start = time.perf_counter()
    depth_map = estimate_depth(rgb_frame, model_variant=model_variant)
    t_depth_end = time.perf_counter()
    depth_time = (t_depth_end - t_depth_start) * 1000

    # 3. Temporal Smoothing (Optional)
    t_smooth_start = time.perf_counter()
    if smoother:
        depth_map = smoother.smooth(depth_map)  # Apply to depth map
    t_smooth_end = time.perf_counter()
    smooth_time = (t_smooth_end - t_smooth_start) * 1000 if smoother else 0

    # 4. Generate Point Cloud
    t_pc_gen_start = time.perf_counter()
    points_xyz = depth_to_points(depth_map, intrinsics)
    points_xyzrgb = combine_xyz_rgb(points_xyz, rgb_frame, intrinsics)
    t_pc_gen_end = time.perf_counter()
    pc_gen_time = (t_pc_gen_end - t_pc_gen_start) * 1000

    # 5. Save Point Cloud
    t_save_start = time.perf_counter()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    frame_basename = os.path.splitext(os.path.basename(image_path))[0]
    pcd_filename = f"{frame_basename}_{model_variant}"
    if smoother:
        pcd_filename += "_smoothed"
    pcd_filepath = os.path.join(output_dir, f"{pcd_filename}.pcd")

    if points_xyzrgb.shape[0] > 0:
        save_pcd(points_xyzrgb, pcd_filepath)
    # else:
    #     print(f"Skipping save for {frame_basename} as no points were generated.")
    t_save_end = time.perf_counter()
    save_time = (t_save_end - t_save_start) * 1000

    total_frame_time = load_time + depth_time + smooth_time + pc_gen_time + save_time
    return {
        "load_time_ms": load_time,
        "depth_time_ms": depth_time,
        "smooth_time_ms": smooth_time,
        "pc_gen_time_ms": pc_gen_time,
        "save_time_ms": save_time,
        "total_frame_time_ms": total_frame_time,
        "num_points": points_xyzrgb.shape[0],
    }


def run_benchmark(args):
    print(f"Starting benchmark with model: {args.model_variant}, resolution: {args.resolution}")
    print(
        f"Number of frames: {args.num_frames}, Output dir: {args.output_dir}, Image_dir: {args.image_dir}"
    )
    if args.enable_smoothing:
        print(f"Temporal smoothing enabled with alpha={args.smoothing_alpha}")

    frame_width, frame_height = get_image_resolution(args.resolution)
    camera_intrinsics = get_intrinsics_for_resolution(frame_width, frame_height)
    print(f"Using intrinsics: {camera_intrinsics} for resolution {frame_width}x{frame_height}")

    create_dummy_frames(args.image_dir, args.num_frames, width=frame_width, height=frame_height)

    image_paths = list_frames(args.image_dir)[: args.num_frames]
    if not image_paths:
        print(f"No images found in {args.image_dir}. Exiting.")
        return

    # Initialize smoother if enabled
    smoother = None
    if args.enable_smoothing:
        smoother = ExponentialMovingAverageSmoother(alpha=args.smoothing_alpha)

    # Warm-up run
    print("Performing warm-up run (processing first frame)...")
    _load_model(model_name=args.model_variant)
    process_frame(
        image_paths[0], args.model_variant, camera_intrinsics, args.output_dir, smoother=smoother
    )
    if smoother:
        smoother.reset()  # Reset smoother after warm-up if it was used
    print("Warm-up complete.")

    all_frame_stats = []
    # Using perf_counter for overall time is better than summing individual stats due to potential small gaps
    overall_processing_start_time = time.perf_counter()

    for _, image_path in enumerate(image_paths): # Changed i to _
        # print(f"Processing frame {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        stats = process_frame(
            image_path, args.model_variant, camera_intrinsics, args.output_dir, smoother=smoother
        )
        all_frame_stats.append(stats)
        # print(f"  Frame time: {stats['total_frame_time_ms']:.2f} ms, Points: {stats['num_points']}")
        # print(f"  Timings (ms): Load={stats['load_time_ms']:.2f}, Depth={stats['depth_time_ms']:.2f}, Smooth={stats['smooth_time_ms']:.2f}, PCGen={stats['pc_gen_time_ms']:.2f}, Save={stats['save_time_ms']:.2f}")

    overall_processing_end_time = time.perf_counter()
    total_elapsed_s = overall_processing_end_time - overall_processing_start_time

    # Calculate averages from collected stats
    # Note: total_frame_time_ms from stats includes individual perf_counters.
    # For overall FPS, using the total_elapsed_s is more robust.
    avg_total_frame_time_from_stats = np.mean([s["total_frame_time_ms"] for s in all_frame_stats])

    # FPS calculated from total time taken for all frames (excluding warm-up)
    actual_fps = len(all_frame_stats) / total_elapsed_s if total_elapsed_s > 0 else 0
    actual_avg_latency_ms = (
        (total_elapsed_s * 1000) / len(all_frame_stats) if len(all_frame_stats) > 0 else 0
    )

    print("\n--- Benchmark Summary ---")
    print(f"Processed {len(all_frame_stats)} frames.")
    print(f"Total processing time (excluding warm-up): {total_elapsed_s:.2f} seconds.")
    print(f"Average FPS (from total time): {actual_fps:.2f}")
    print(f"Average latency per frame (from total time): {actual_avg_latency_ms:.2f} ms")
    print(
        f"Average total_frame_time_ms (sum of components): {avg_total_frame_time_from_stats:.2f} ms"
    )

    # Further breakdown from stats
    timing_keys = [
        "load_time_ms",
        "depth_time_ms",
        "smooth_time_ms",
        "pc_gen_time_ms",
        "save_time_ms",
    ]
    for key in timing_keys:
        avg_val = np.mean([s[key] for s in all_frame_stats])
        # percentage = (avg_val / avg_total_frame_time_from_stats * 100) if avg_total_frame_time_from_stats > 0 else 0
        # print(f"Average {key.replace('_ms','').replace('_',' ')}: {avg_val:.2f} ms ({percentage:.1f}%)")
        print(f"Average {key.replace('_ms', '').replace('_', ' ')}: {avg_val:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark script for 2D Video to 3D Point Cloud conversion."
    )
    parser.add_argument(
        "--image_dir", type=str, default=DEFAULT_IMAGE_DIR, help="Directory containing test frames."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output PCD files.",
    )
    parser.add_argument(
        "--num_frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of frames to process."
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default=DEFAULT_MODEL_VARIANT,
        choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"],
        help="Depth estimation model variant.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=DEFAULT_RESOLUTION_MODE,
        help="Resolution mode (e.g., '1080p', '720p', 'VGA', or 'WIDTHxHEIGHT').",
    )
    parser.add_argument(
        "--enable_smoothing", action="store_true", help="Enable temporal smoothing."
    )
    parser.add_argument(
        "--smoothing_alpha", type=float, default=0.5, help="Alpha value for EMA smoother."
    )

    args = parser.parse_args()
    run_benchmark(args)
