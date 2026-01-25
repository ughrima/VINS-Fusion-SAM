# `loop_fusion` package (`loop_fusion/`)

This ROS package performs **loop closure** and **pose-graph optimization** on top of the VINS odometry output.

## What you run

- **`loop_fusion_node`** (built from `src/pose_graph_node.cpp`)
  - Buffers incoming keyframe images + feature point clouds + odometry
  - Detects loop candidates using bag-of-words (DBoW2 + BRIEF)
  - Runs pose-graph optimization in a background thread
  - Publishes corrected trajectories and visualization topics

## Key files

- `src/pose_graph_node.cpp`: ROS wiring (subscriptions, buffers, publishing)
- `src/pose_graph.h/.cpp`: `PoseGraph` class (loop detection + optimization + drift)
- `src/keyframe.h/.cpp`: keyframe container + descriptor handling
- `src/ThirdParty/`: vendored DBoW2/DVision/DUtils used for place recognition

## Where to read next

- High-level architecture: `docs/ARCHITECTURE.md`
- File-by-file index: `docs/FILE_INDEX.md`

