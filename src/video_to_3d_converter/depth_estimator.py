"""Module for estimating depth from RGB images using MiDaS models."""

import cv2
import numpy as np
import torch

# Initialize globals with a default that indicates no model is loaded yet
MODEL_NAME = None
MODEL = None
TRANSFORM = None


def _load_model(model_name: str):  # Removed default model_name here
    global MODEL, TRANSFORM, MODEL_NAME
    # No need to check if MODEL is not None and MODEL_NAME == model_name here,
    # as this check is done in estimate_depth before calling _load_model.

    try:
        # Attempt to load the model from torch.hub
        loaded_model = torch.hub.load(
            "intel-isl/MiDaS", model_name, trust_repo=True
        )  # Added trust_repo=True
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        loaded_model.to(device)
        loaded_model.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )  # Added trust_repo=True
        if model_name in ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]:  # Common MiDaS models
            transform_to_use = midas_transforms.dpt_transform
        else:  # Potentially other models might use a different transform, or a generic one
            transform_to_use = (
                midas_transforms.small_transform
            )  # Fallback or specific for other small models

        # Update globals only after successful loading and transform selection
        MODEL = loaded_model
        TRANSFORM = transform_to_use
        MODEL_NAME = model_name  # Update the global model name

        # print(f"Model {model_name} loaded successfully on {device}.") # Consider logging

    except Exception:
        # print(f"Error loading model {model_name}: {e}") # Consider logging
        # Reset globals if loading fails to ensure a clean state
        MODEL = None
        TRANSFORM = None
        MODEL_NAME = None  # Explicitly reset MODEL_NAME as well
        raise  # Re-raise the exception so the caller can handle it


def load_rgb_frame(image_path: str) -> np.ndarray:
    if not image_path:
        raise ValueError("Image path cannot be empty.")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def estimate_depth(
    rgb_frame: np.ndarray, model_variant: str
) -> np.ndarray:  # Removed default here, should be explicit
    global MODEL, TRANSFORM, MODEL_NAME

    # Check if the requested model is different from the loaded one or if no model is loaded
    if MODEL is None or model_variant != MODEL_NAME:
        # print(f"Loading model {model_variant}...") # Consider logging
        _load_model(model_variant)  # This will update MODEL, TRANSFORM, and MODEL_NAME globals

    # After attempting to load, check if it was successful
    if MODEL is None or TRANSFORM is None or MODEL_NAME != model_variant:
        # Use model_variant in the error message as that's what was requested
        raise RuntimeError(
            f"Model {model_variant} is not loaded properly. Check logs for loading errors."
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Ensure frame is contiguous, which can be an issue with some transforms/models
    rgb_frame_contiguous = np.ascontiguousarray(rgb_frame)

    with torch.no_grad():
        transformed_img = TRANSFORM(rgb_frame_contiguous).to(device)  # Use contiguous frame
        prediction = MODEL(transformed_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_frame_contiguous.shape[:2],  # Use contiguous frame shape
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map
