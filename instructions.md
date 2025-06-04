# Product Requirements Document (PRD)

**Module:** Real‑time 2D Video ➜ 3D Point‑Cloud Converter
**Version:** 1.1 — 2025‑06‑04
**Author:** Peter Hsu
**Status:** Draft for review

---

## 1  Introduction

This module converts a monocular RGB video stream of a conference participant into a temporally consistent 3‑D point cloud in real time. It will be integrated later into a broader 3‑D video‑conferencing stack but is currently developed as a self‑contained backend component.  Depth is estimated with a model from the **MiDaS** family or an equivalent modern network (e.g. **Depth‑Anything ViT‑S**).

---

## 2  Goals

| Category        | Objective                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| **Primary**     | Generate a per‑frame XYZRGB point cloud from an incoming RGB frame at real‑time speed.                       |
| **Performance** | Sustain **≥ 30 FPS** at **1920 × 1080** resolution on the reference GPU (RTX 4080, 16 GB).                   |
| **Quality**     | Output must represent the human subject recognisably (see objective metric below).                           |
| **Latency**     | End‑to‑end latency ≤ **100 ms** (capture → point cloud available on disk), including any temporal smoothing. |
| **Modularity**  | Deliver as an importable Python package with a narrow public API and clear I/O contracts.                    |

### Objective quality metric

*Average point‑to‑mesh distance on the internal “ConfFaces‑10” validation set ≤ 15 mm and ≥ 50 k points inside the person mask per frame.*

---

## 3  Target Users

* **Primary:** In‑house developers integrating this module into the full conferencing pipeline.
* **Indirect:** End‑users of the final 3‑D conferencing product.

---

## 4  Scope

### 4.1  In‑scope

* Frame‑by‑frame ingestion of an RGB video stream.
* Monocular depth estimation via pre‑trained MiDaS/Depth‑Anything model.
* Un‑projection to XYZRGB point cloud using supplied camera intrinsics.
* Optional temporal consistency filter (EMA, bilateral‑temporal, or similar) that adds ≤ 33 ms.
* Storage of each point cloud frame as a file on local or network storage.
* Implementation primarily in **Python 3.11**; C++/CUDA kernels permitted for hot loops (via **vcpkg**).
* Dependency management: **uv** (Python) and **vcpkg** (C++).

### 4.2  Out‑of‑scope

* Rendering or visualisation of point clouds.
* Network streaming or transport of data.
* Multi‑person segmentation, SLAM, volumetric fusion, or camera self‑calibration.
* UI/UX for end users.

---

## 5  Success Metrics

| Metric          | Target                                                             |
| --------------- | ------------------------------------------------------------------ |
| **Throughput**  | ≥ 30 FPS at 1080p on RTX 4080.                                     |
| **Latency**     | ≤ 100 ms end‑to‑end (with ≤ 33 ms allocated to optional smoother). |
| **Quality**     | See metric in §2.                                                  |
| **Stability**   | No crashes / leaks over a 2‑hour soak test.                        |
| **Integration** | Public API consumed without change by reference app.               |

---

## 6  Functional Requirements

| ID  | Requirement                                                                                             | Priority |
| --- | ------------------------------------------------------------------------------------------------------- | -------- |
| FR1 | Accept a single RGB frame as input.                                                                     | MUST     |
| FR2 | Estimate depth map using the configured model.                                                          | MUST     |
| FR3 | Allow choice between at least two model variants (speed vs. quality).                                   | SHOULD   |
| FR4 | Accept calibrated camera intrinsics (fx, fy, cx, cy).                                                   | MUST     |
| FR5 | Generate XYZ coordinates via pinhole un‑projection.                                                     | MUST     |
| FR6 | Attach RGB colour to each point.                                                                        | SHOULD   |
| FR7 | Configurable input resolution (VGA, HD, FullHD).                                                        | MUST     |
| FR8 | Persist each frame’s point cloud to storage **in binary PCD (little‑endian, float32, XYZRGB)**.         | MUST     |
| FR9 | Return structured JSON error codes for recoverable faults (e.g. `E_MODEL_LOAD`, `E_OOM`, `E_NO_INPUT`). | MUST     |

---

## 7  Non‑Functional Requirements

| ID   | Requirement                                                                                             | Priority |
| ---- | ------------------------------------------------------------------------------------------------------- | -------- |
| NFR1 | ≥ 30 FPS @ 1080p on reference GPU.                                                                      | MUST     |
| NFR2 | Latency ≤ 100 ms (breakdown: ≤ 33 ms capture→cloud, ≤ 33 ms optional smoothing, remainder for I/O).     | MUST     |
| NFR3 | Depth accuracy sufficient for metric in §2.                                                             | MUST     |
| NFR4 | Operate within 70 % GPU utilisation and ≤ 3 GB additional VRAM on RTX 4080; CPU load ≤ 4 logical cores. | MUST     |
| NFR5 | Package exposes a stable API and is importable via `pip install -e .`.                                  | MUST     |
| NFR6 | Uptime ≥ 2 h continuous without leaks (tested with `valgrind` for C++ parts, `tracemalloc` for Python). | SHOULD   |
| NFR7 | Code conforms to **PEP 8 + Ruff**; doc‑strings ≥ 80 % coverage; C++ follows **Google style**.           | SHOULD   |
| NFR8 | Security: SBOM generated; dependencies scanned weekly; camera access sandboxed.                         | SHOULD   |

---

## 8  Assumptions

* **Hardware:** Development & deployment on **NVIDIA RTX 4080 16 GB, CUDA 12.5, cuDNN 9.0, driver 555.xx**.
* **Single primary subject** per video stream.
* **Adequate lighting** and near‑frontal camera angle typical of webcam setups.
* **FX, FY, CX, CY** supplied externally by the conferencing application.

---

## 9  Limitations & Constraints

* Monocular depth scale ambiguity — absolute units unknown; downstream components may rescale via user height or table plane.
* MiDaS & Depth‑Anything can mis‑estimate texture‑less, reflective, or transparent regions.
* Only visible surfaces included (no occlusion completion).
* Achievable throughput tied to reference GPU; lower‑tier cards will require down‑sampling or lower FPS.

---

## 10  Open Questions / Future Work

1. Will a person‑specific fine‑tuning of Depth‑Anything further improve face fidelity?
2. Should we bundle a tiny Web UI demo for quick visual sanity checks?
3. Investigate **TensorRT‑LLM** style layer fusion for additional throughput margin at 1080p.

---

## 11  Glossary

* **PCD:** Point Cloud Data file format defined by the PCL library.
* **EMA:** Exponential Moving Average.
* **XYZRGB:** 3‑D coordinate + colour tuple.
