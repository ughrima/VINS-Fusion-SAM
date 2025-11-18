# Quick Start Guide: SAM Integration with VINS Fusion

## Prerequisites Checklist

- [ ] ROS installed (Kinetic or Melodic)
- [ ] VINS Fusion built successfully
- [ ] Segment Anything Model installed (`pip install -e segment-anything`)
- [ ] SAM checkpoint downloaded (e.g., `sam_vit_b_01ec64.pth`)
- [ ] PyTorch installed with CUDA support (recommended)

## Step-by-Step Setup

### 1. Download SAM Checkpoint

```bash
cd segment-anything
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 2. Update Launch File

Edit `vins_estimator/launch/sam_service.launch` and update the checkpoint path:

```xml
<param name="sam_checkpoint_path" value="/absolute/path/to/segment-anything/checkpoints/sam_vit_b_01ec64.pth" />
```

### 3. Update Config File

Add SAM parameters to your VINS config file (e.g., `config/euroc/euroc_stereo_imu_sam_config.yaml`):

```yaml
use_sam: 1
sam_update_frequency: 5
```

### 4. Build the Package

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 5. Make Service Executable

```bash
chmod +x ~/catkin_ws/src/VINS-Fusion-SAM/vins_estimator/src/sam_service/sam_ros_service.py
```

## Running the System

### Terminal 1: Start SAM Service

```bash
roslaunch vins sam_service.launch
```

You should see:
```
[INFO] Loading SAM model: vit_b from /path/to/checkpoint.pth
[INFO] SAM model loaded successfully
[INFO] SAM service ready
```

### Terminal 2: Start VINS with SAM

```bash
rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion-SAM/config/euroc/euroc_stereo_imu_sam_config.yaml
```

### Terminal 3: Start RViz (Optional)

```bash
roslaunch vins vins_rviz.launch
```

### Terminal 4: Play Dataset (if using bag file)

```bash
rosbag play /path/to/your/dataset.bag
```

## Verification

Check that SAM is working:

1. **Check service is running:**
   ```bash
   rosservice list | grep sam
   ```
   Should show: `/sam_segmentation`

2. **Check VINS logs:**
   Look for: `[INFO] SAM integration initialized successfully`

3. **Monitor feature tracking:**
   If `show_track: 1` in config, you can visualize features being detected primarily in segmented regions.

## Troubleshooting

### Issue: "SAM service not available"

**Solution:**
- Ensure SAM service is running before starting VINS
- Check checkpoint path is correct and file exists
- Verify PyTorch is installed: `python -c "import torch; print(torch.__version__)"`

### Issue: Out of memory

**Solution:**
- Use smaller model (vit_b instead of vit_h)
- Increase `sam_update_frequency` to 10 or higher
- Close other GPU applications

### Issue: Slow performance

**Solution:**
- Use GPU: Set `device: cuda` in launch file
- Use smaller model (vit_b)
- Increase `sam_update_frequency`

## Performance Tips

- **For real-time performance:** Use `vit_b` model with `sam_update_frequency: 10`
- **For best accuracy:** Use `vit_h` model with `sam_update_frequency: 5`
- **For CPU-only systems:** Use `vit_b` with `device: cpu` and `sam_update_frequency: 20`

## Next Steps

- Read the full documentation: `SAM_INTEGRATION_README.md`
- Experiment with different `sam_update_frequency` values
- Try different SAM models (vit_b, vit_l, vit_h)
- Customize mask generation parameters in `sam_ros_service.py`

