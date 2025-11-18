# SAM Integration with VINS Fusion

This document describes the integration of Segment Anything Model (SAM) with VINS Fusion SLAM system.

## Overview

The SAM integration enhances VINS Fusion by using semantic segmentation masks to guide feature detection. This allows the SLAM system to focus on meaningful objects and regions in the scene, potentially improving tracking robustness and accuracy.

## Architecture

The integration consists of three main components:

1. **SAM ROS Service** (`sam_ros_service.py`): A Python service that runs the SAM model and provides segmentation masks
2. **SAM Client** (`sam_client.cpp/h`): A C++ client that communicates with the SAM service
3. **Feature Tracker Integration**: Modified feature tracker that uses SAM masks to guide feature detection

## Prerequisites

### 1. Install Segment Anything Model

```bash
cd segment-anything
pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

### 2. Download SAM Model Checkpoints

Download one of the SAM model checkpoints:
- [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (largest, most accurate)
- [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) (smallest, fastest)

Place the checkpoint file in `segment-anything/checkpoints/` directory.

### 3. Install PyTorch

Make sure PyTorch is installed with CUDA support (if using GPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Building

1. Build the VINS Fusion package:

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

2. Make the SAM service executable:

```bash
chmod +x ~/catkin_ws/src/VINS-Fusion-SAM/vins_estimator/src/sam_service/sam_ros_service.py
```

## Configuration

### Config File Parameters

Add the following parameters to your VINS Fusion config file:

```yaml
# SAM Integration Parameters
use_sam: 1              # Enable (1) or disable (0) SAM integration
sam_update_frequency: 5 # Update SAM mask every N frames
```

- `use_sam`: Set to 1 to enable SAM integration, 0 to disable
- `sam_update_frequency`: Controls how often the SAM mask is updated. Lower values mean more frequent updates but higher computational cost. Recommended: 5-10 frames.

### SAM Service Parameters

Edit `launch/sam_service.launch` to configure the SAM service:

```xml
<param name="sam_model_type" value="vit_b" />  <!-- Options: vit_h, vit_l, vit_b -->
<param name="sam_checkpoint_path" value="/path/to/checkpoint.pth" />
<param name="use_automatic_mask" value="true" />
<param name="device" value="cuda" />  <!-- Options: cuda, cpu -->
```

## Usage

### 1. Start SAM Service

In one terminal:

```bash
roslaunch vins sam_service.launch
```

Or manually:

```bash
rosrun vins sam_ros_service.py _sam_model_type:=vit_b _sam_checkpoint_path:=/path/to/checkpoint.pth
```

### 2. Run VINS Fusion with SAM

In another terminal:

```bash
roslaunch vins vins_sam.launch config_path:=/path/to/config.yaml
```

Or use the standard VINS node with SAM-enabled config:

```bash
rosrun vins vins_node /path/to/config_with_sam.yaml
```

### 3. Play Dataset (if using bag file)

```bash
rosbag play /path/to/dataset.bag
```

## How It Works

1. **Image Processing**: When VINS Fusion receives a new image, it periodically (based on `sam_update_frequency`) sends it to the SAM service.

2. **Segmentation**: The SAM service generates segmentation masks identifying objects and regions in the image.

3. **Feature Detection**: The feature tracker uses the SAM mask to guide feature detection:
   - Features are preferentially detected in segmented regions (where SAM identified objects)
   - This focuses tracking on meaningful scene elements rather than background or noise

4. **Mask Updates**: The SAM mask is updated periodically to adapt to scene changes while maintaining real-time performance.

## Performance Considerations

- **Update Frequency**: Higher `sam_update_frequency` values reduce computational load but may miss rapid scene changes
- **Model Size**: ViT-B is fastest but less accurate; ViT-H is most accurate but slowest
- **GPU Usage**: SAM benefits significantly from GPU acceleration. Use CPU only if GPU is unavailable.
- **Real-time Performance**: SAM processing adds latency. Adjust `sam_update_frequency` to balance accuracy and performance.

## Troubleshooting

### SAM Service Not Available

If you see "SAM service not available" warnings:

1. Check that the SAM service is running: `rostopic list | grep sam`
2. Verify the checkpoint path is correct
3. Check ROS service: `rosservice list | grep sam`
4. Review service logs for errors

### Out of Memory Errors

- Use a smaller model (ViT-B instead of ViT-H)
- Increase `sam_update_frequency` to process fewer frames
- Reduce image resolution if possible

### Slow Performance

- Use GPU acceleration (`device: cuda`)
- Use smaller model (ViT-B)
- Increase `sam_update_frequency`
- Consider using automatic mask generation with lower quality settings

## Advanced Configuration

### Custom Mask Generation

You can modify `sam_ros_service.py` to customize mask generation:

- Adjust `points_per_side` for automatic mask generation
- Modify `pred_iou_thresh` and `stability_score_thresh` for quality filtering
- Change `crop_n_layers` for multi-scale processing

### Point-based Segmentation

The service also supports point-based segmentation. You can modify the feature tracker to use detected features as prompts for more targeted segmentation.

## Citation

If you use this integration, please cite:

- VINS Fusion: [VINS-Fusion Paper](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
- Segment Anything: [SAM Paper](https://arxiv.org/abs/2304.02643)

## License

This integration follows the licenses of the original projects:
- VINS Fusion: GPLv3
- Segment Anything: Apache 2.0

