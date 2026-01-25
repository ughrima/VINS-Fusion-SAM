# `vins` package (`vins_estimator/`)

This ROS package contains the core VINS-Fusion estimator plus the SAM integration.

## What you run

- **`vins_node`** (built from `src/rosNodeTest.cpp`)
  - Subscribes to IMU + images (mono or stereo)
  - Calls into `Estimator` (`src/estimator/estimator.*`)
  - Publishes odometry, paths, point clouds, and debug images

## Key folders

- **`src/estimator/`**: sliding-window VIO estimator + feature manager + parameter loading
- **`src/featureTracker/`**: feature tracking (optical flow) + SAM-mask-aware feature selection
- **`src/sam_service/`**: SAM Python service + C++ client
- **`src/factor/`**: Ceres residual factors (IMU, projection, marginalization)
- **`src/initial/`**: initialization (SFM, alignment, extrinsic rotation)
- **`src/utility/`**: math + visualization helpers

## Launch files

- `launch/sam_service.launch`: start the SAM Python service
- `launch/vins_sam.launch`: start SAM service + VINS node + RViz
- `launch/vins_rviz.launch`: RViz only

## Where to read next

- High-level architecture: `docs/ARCHITECTURE.md`
- File-by-file index: `docs/FILE_INDEX.md`

