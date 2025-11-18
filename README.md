# VINS-Fusion-SAM
## An optimization-based multi-sensor state estimator with Segment Anything Model integration

<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/vins_logo.png" width = 55% height = 55% div align=left />
<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/kitti.png" width = 34% height = 34% div align=center />

VINS-Fusion is an optimization-based multi-sensor state estimator, which achieves accurate self-localization for autonomous applications (drones, cars, and AR/VR). VINS-Fusion is an extension of [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono), which supports multiple visual-inertial sensor types (mono camera + IMU, stereo cameras + IMU, even stereo cameras only). We also show a toy example of fusing VINS with GPS.

**This repository includes integration with Segment Anything Model (SAM)**, which enhances feature tracking by using semantic segmentation to guide feature detection on meaningful objects and regions.

**Features:**
- multiple sensors support (stereo cameras / mono camera+IMU / stereo cameras+IMU)
- online spatial calibration (transformation between camera and IMU)
- online temporal calibration (time offset between camera and IMU)
- visual loop closure
- **SAM integration for semantic-aware feature tracking**

<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/kitti_rank.png" width = 80% height = 80% />

We are the **top** open-sourced stereo algorithm on [KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (12.Jan.2019).

**Authors:** [Tong Qin](http://www.qintonguav.com), Shaozu Cao, Jie Pan, [Peiliang Li](https://peiliangli.github.io/), and [Shaojie Shen](http://www.ece.ust.hk/ece.php/profile/facultydetail/eeshaojie) from the [Aerial Robotics Group](http://uav.ust.hk/), [HKUST](https://www.ust.hk/)

**Videos:**

<a href="https://www.youtube.com/embed/1qye82aW7nI" target="_blank"><img src="http://img.youtube.com/vi/1qye82aW7nI/0.jpg" 
alt="VINS" width="320" height="240" border="10" /></a>

**Related Paper:** (paper is not exactly same with code)

* **Online Temporal Calibration for Monocular Visual-Inertial Systems**, Tong Qin, Shaojie Shen, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS, 2018), **best student paper award** [pdf](https://ieeexplore.ieee.org/abstract/document/8593603)

* **VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator**, Tong Qin, Peiliang Li, Shaojie Shen, IEEE Transactions on Robotics [pdf](https://ieeexplore.ieee.org/document/8421746/?arnumber=8421746&source=authoralert)

* **Segment Anything**, Alexander Kirillov et al., arXiv:2304.02643 [pdf](https://arxiv.org/abs/2304.02643)

*If you use VINS-Fusion for your academic research, please cite our related papers. [bib](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/paper_bib.txt)*

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start with SAM Integration](#2-quick-start-with-sam-integration)
3. [SAM Integration Overview](#3-sam-integration-overview)
4. [Building VINS-Fusion](#4-building-vins-fusion)
5. [Configuration](#5-configuration)
6. [Usage Examples](#6-usage-examples)
7. [Troubleshooting](#7-troubleshooting)
8. [Acknowledgements](#8-acknowledgements)
9. [License](#9-license)

---

## 1. Prerequisites

### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04 or 18.04.
ROS Kinetic or Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 1.3. **Segment Anything Model (for SAM integration)**
If you want to use the SAM integration feature:

```bash
cd segment-anything
pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

**Download SAM Model Checkpoints:**
- [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (largest, most accurate)
- [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) (smallest, fastest)

Place the checkpoint file in `segment-anything/checkpoints/` directory.

**Install PyTorch:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 2. Quick Start with SAM Integration

### Prerequisites Checklist

- [ ] ROS installed (Kinetic or Melodic)
- [ ] Ceres Solver installed
- [ ] Segment Anything Model installed (`pip install -e segment-anything`)
- [ ] SAM checkpoint downloaded (e.g., `sam_vit_b_01ec64.pth`)
- [ ] PyTorch installed with CUDA support (recommended)

### Step-by-Step Setup

#### 1. Download SAM Checkpoint

```bash
cd segment-anything
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

#### 2. Clone and Build

```bash
cd ~/catkin_ws/src
git clone https://github.com/your-username/VINS-Fusion-SAM.git
cd ../
catkin_make
source devel/setup.bash
```

#### 3. Update Launch File

Edit `vins_estimator/launch/sam_service.launch` and update the checkpoint path:

```xml
<param name="sam_checkpoint_path" value="/absolute/path/to/segment-anything/checkpoints/sam_vit_b_01ec64.pth" />
```

#### 4. Update Config File

Add SAM parameters to your VINS config file (e.g., `config/euroc/euroc_stereo_imu_sam_config.yaml`):

```yaml
use_sam: 1
sam_update_frequency: 5
```

#### 5. Make Service Executable

```bash
chmod +x ~/catkin_ws/src/VINS-Fusion-SAM/vins_estimator/src/sam_service/sam_ros_service.py
```

### Running the System

**Terminal 1: Start SAM Service**
```bash
roslaunch vins sam_service.launch
```

You should see:
```
[INFO] Loading SAM model: vit_b from /path/to/checkpoint.pth
[INFO] SAM model loaded successfully
[INFO] SAM service ready
```

**Terminal 2: Start VINS with SAM**
```bash
rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion-SAM/config/euroc/euroc_stereo_imu_sam_config.yaml
```

**Terminal 3: Start RViz (Optional)**
```bash
roslaunch vins vins_rviz.launch
```

**Terminal 4: Play Dataset (if using bag file)**
```bash
rosbag play /path/to/your/dataset.bag
```

### Verification

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

---

## 3. SAM Integration Overview

### What is SAM Integration?

The SAM integration enhances VINS Fusion by using semantic segmentation masks to guide feature detection. Instead of detecting features randomly across the image, the system focuses on meaningful objects and regions identified by SAM, potentially improving tracking robustness and accuracy.

### Architecture

The integration consists of three main components:

1. **SAM ROS Service** (`sam_ros_service.py`): A Python service that runs the SAM model and provides segmentation masks
2. **SAM Client** (`sam_client.cpp/h`): A C++ client that communicates with the SAM service
3. **Feature Tracker Integration**: Modified feature tracker that uses SAM masks to guide feature detection

### How It Works

1. **Image Processing**: When VINS Fusion receives a new image, it periodically (based on `sam_update_frequency`) sends it to the SAM service.

2. **Segmentation**: The SAM service generates segmentation masks identifying objects and regions in the image.

3. **Feature Detection**: The feature tracker uses the SAM mask to guide feature detection:
   - Features are preferentially detected in segmented regions (where SAM identified objects)
   - This focuses tracking on meaningful scene elements rather than background or noise

4. **Mask Updates**: The SAM mask is updated periodically to adapt to scene changes while maintaining real-time performance.

### Performance Considerations

- **Update Frequency**: Higher `sam_update_frequency` values reduce computational load but may miss rapid scene changes
- **Model Size**: ViT-B is fastest but less accurate; ViT-H is most accurate but slowest
- **GPU Usage**: SAM benefits significantly from GPU acceleration. Use CPU only if GPU is unavailable.
- **Real-time Performance**: SAM processing adds latency. Adjust `sam_update_frequency` to balance accuracy and performance.

**Performance Tips:**
- **For real-time performance:** Use `vit_b` model with `sam_update_frequency: 10`
- **For best accuracy:** Use `vit_h` model with `sam_update_frequency: 5`
- **For CPU-only systems:** Use `vit_b` with `device: cpu` and `sam_update_frequency: 20`

---

## 4. Building VINS-Fusion

Clone the repository and catkin_make:

```bash
cd ~/catkin_ws/src
git clone https://github.com/your-username/VINS-Fusion-SAM.git
cd ../
catkin_make
source devel/setup.bash
```

(if you fail in this step, try to find another computer with clean system or reinstall Ubuntu and ROS)

---

## 5. Configuration

### VINS Fusion Config File

Write a config file for your device. You can take config files of EuRoC and KITTI as examples.

### SAM Integration Parameters

Add the following parameters to your VINS Fusion config file:

```yaml
# SAM Integration Parameters
use_sam: 1              # Enable (1) or disable (0) SAM integration
sam_update_frequency: 5 # Update SAM mask every N frames
```

- `use_sam`: Set to 1 to enable SAM integration, 0 to disable
- `sam_update_frequency`: Controls how often the SAM mask is updated. Lower values mean more frequent updates but higher computational cost. Recommended: 5-10 frames.

### SAM Service Parameters

Edit `vins_estimator/launch/sam_service.launch` to configure the SAM service:

```xml
<param name="sam_model_type" value="vit_b" />  <!-- Options: vit_h, vit_l, vit_b -->
<param name="sam_checkpoint_path" value="/path/to/checkpoint.pth" />
<param name="use_automatic_mask" value="true" />
<param name="device" value="cuda" />  <!-- Options: cuda, cpu -->
```

### Camera Calibration

VINS-Fusion support several camera models (pinhole, mei, equidistant). You can use [camera model](https://github.com/hengli/camodocal) to calibrate your cameras. We put some example data under `/camera_models/calibrationdata` to tell you how to calibrate.

```bash
cd ~/catkin_ws/src/VINS-Fusion-SAM/camera_models/camera_calib_example/
rosrun camera_models Calibrations -w 12 -h 8 -s 80 -i calibrationdata --camera-model pinhole
```

---

## 6. Usage Examples

### 6.1 EuRoC Example

Download [EuRoC MAV Dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) to YOUR_DATASET_FOLDER. Take MH_01 for example, you can run VINS-Fusion with three sensor types (monocular camera + IMU, stereo cameras + IMU and stereo cameras).

Open four terminals, run vins odometry, visual loop closure(optional), rviz and play the bag file respectively.
Green path is VIO odometry; red path is odometry under visual loop closure.

#### 6.1.1 Monocular camera + IMU

```bash
roslaunch vins vins_rviz.launch
rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion-SAM/config/euroc/euroc_mono_imu_config.yaml 
(optional) rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/VINS-Fusion-SAM/config/euroc/euroc_mono_imu_config.yaml 
rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

#### 6.1.2 Stereo cameras + IMU

```bash
roslaunch vins vins_rviz.launch
rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion-SAM/config/euroc/euroc_stereo_imu_config.yaml 
(optional) rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/VINS-Fusion-SAM/config/euroc/euroc_stereo_imu_config.yaml 
rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

#### 6.1.3 Stereo cameras + IMU with SAM

```bash
# Terminal 1: Start SAM service
roslaunch vins sam_service.launch

# Terminal 2: Start VINS with SAM
rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion-SAM/config/euroc/euroc_stereo_imu_sam_config.yaml

# Terminal 3: Start RViz
roslaunch vins vins_rviz.launch

# Terminal 4: Play dataset
rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

#### 6.1.4 Stereo cameras

```bash
roslaunch vins vins_rviz.launch
rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion-SAM/config/euroc/euroc_stereo_config.yaml 
(optional) rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/VINS-Fusion-SAM/config/euroc/euroc_stereo_config.yaml 
rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/euroc.gif" width = 430 height = 240 />

### 6.2 KITTI Example

#### 6.2.1 KITTI Odometry (Stereo)

Download [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to YOUR_DATASET_FOLDER. Take sequences 00 for example,
Open two terminals, run vins and rviz respectively.
(We evaluated odometry on KITTI benchmark without loop closure function)

```bash
roslaunch vins vins_rviz.launch
(optional) rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/VINS-Fusion-SAM/config/kitti_odom/kitti_config00-02.yaml
rosrun vins kitti_odom_test ~/catkin_ws/src/VINS-Fusion-SAM/config/kitti_odom/kitti_config00-02.yaml YOUR_DATASET_FOLDER/sequences/00/ 
```

#### 6.2.2 KITTI GPS Fusion (Stereo + GPS)

Download [KITTI raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) to YOUR_DATASET_FOLDER. Take [2011_10_03_drive_0027_synced](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0027/2011_10_03_drive_0027_sync.zip) for example.
Open three terminals, run vins, global fusion and rviz respectively.
Green path is VIO odometry; blue path is odometry under GPS global fusion.

```bash
roslaunch vins vins_rviz.launch
rosrun vins kitti_gps_test ~/catkin_ws/src/VINS-Fusion-SAM/config/kitti_raw/kitti_10_03_config.yaml YOUR_DATASET_FOLDER/2011_10_03_drive_0027_sync/ 
rosrun global_fusion global_fusion_node
```

<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/kitti.gif" width = 430 height = 240 />

### 6.3 VINS-Fusion on car demonstration

Download [car bag](https://drive.google.com/open?id=10t9H1u8pMGDOI6Q2w2uezEq5Ib-Z8tLz) to YOUR_DATASET_FOLDER.
Open four terminals, run vins odometry, visual loop closure(optional), rviz and play the bag file respectively.
Green path is VIO odometry; red path is odometry under visual loop closure.

```bash
roslaunch vins vins_rviz.launch
rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion-SAM/config/vi_car/vi_car.yaml 
(optional) rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/VINS-Fusion-SAM/config/vi_car/vi_car.yaml 
rosbag play YOUR_DATASET_FOLDER/car.bag
```

<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/car_gif.gif" width = 430 height = 240 />

### 6.4 Run with your devices

VIO is not only a software algorithm, it heavily relies on hardware quality. For beginners, we recommend you to run VIO with professional equipment, which contains global shutter cameras and hardware synchronization.

### 6.5 Docker Support

To further facilitate the building process, we add docker in our code. Docker environment is like a sandbox, thus makes our code environment-independent. To run with docker, first make sure [ros](http://wiki.ros.org/ROS/Installation) and [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) are installed on your machine. Then add your account to `docker` group by `sudo usermod -aG docker $YOUR_USER_NAME`. **Relaunch the terminal or logout and re-login if you get `Permission denied` error**, type:

```bash
cd ~/catkin_ws/src/VINS-Fusion-SAM/docker
make build
```

Note that the docker building process may take a while depends on your network and machine. After VINS-Fusion successfully built, you can run vins estimator with script `run.sh`.
Script `run.sh` can take several flags and arguments. Flag `-k` means KITTI, `-l` represents loop fusion, and `-g` stands for global fusion. You can get the usage details by `./run.sh -h`. Here are some examples with this script:

```bash
# Euroc Monocular camera + IMU
./run.sh ~/catkin_ws/src/VINS-Fusion-SAM/config/euroc/euroc_mono_imu_config.yaml

# Euroc Stereo cameras + IMU with loop fusion
./run.sh -l ~/catkin_ws/src/VINS-Fusion-SAM/config/euroc/euroc_mono_imu_config.yaml

# KITTI Odometry (Stereo)
./run.sh -k ~/catkin_ws/src/VINS-Fusion-SAM/config/kitti_odom/kitti_config00-02.yaml YOUR_DATASET_FOLDER/sequences/00/

# KITTI Odometry (Stereo) with loop fusion
./run.sh -kl ~/catkin_ws/src/VINS-Fusion-SAM/config/kitti_odom/kitti_config00-02.yaml YOUR_DATASET_FOLDER/sequences/00/

# KITTI GPS Fusion (Stereo + GPS)
./run.sh -kg ~/catkin_ws/src/VINS-Fusion-SAM/config/kitti_raw/kitti_10_03_config.yaml YOUR_DATASET_FOLDER/2011_10_03_drive_0027_sync/
```

In Euroc cases, you need open another terminal and play your bag file. If you need modify the code, simply re-run `./run.sh` with proper arguments after your changes.

---

## 7. Troubleshooting

### General VINS Issues

If you fail in building step, try to find another computer with clean system or reinstall Ubuntu and ROS.

### SAM Integration Issues

#### Issue: "SAM service not available"

**Solution:**
- Ensure SAM service is running before starting VINS
- Check checkpoint path is correct and file exists
- Verify PyTorch is installed: `python -c "import torch; print(torch.__version__)"`
- Check that the SAM service is running: `rostopic list | grep sam`
- Check ROS service: `rosservice list | grep sam`
- Review service logs for errors

#### Issue: Out of memory

**Solution:**
- Use smaller model (vit_b instead of vit_h)
- Increase `sam_update_frequency` to 10 or higher
- Close other GPU applications
- Reduce image resolution if possible

#### Issue: Slow performance

**Solution:**
- Use GPU: Set `device: cuda` in launch file
- Use smaller model (vit_b)
- Increase `sam_update_frequency`
- Consider using automatic mask generation with lower quality settings

### Advanced Configuration

#### Custom Mask Generation

You can modify `sam_ros_service.py` to customize mask generation:

- Adjust `points_per_side` for automatic mask generation
- Modify `pred_iou_thresh` and `stability_score_thresh` for quality filtering
- Change `crop_n_layers` for multi-scale processing

#### Point-based Segmentation

The service also supports point-based segmentation. You can modify the feature tracker to use detected features as prompts for more targeted segmentation.

---

## 8. Acknowledgements

We use [ceres solver](http://ceres-solver.org/) for non-linear optimization and [DBoW2](https://github.com/dorian3d/DBoW2) for loop detection, a generic [camera model](https://github.com/hengli/camodocal) and [GeographicLib](https://geographiclib.sourceforge.io/).

This project integrates [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI Research.

---

## 9. License

The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.

The SAM integration follows the licenses of the original projects:
- VINS Fusion: GPLv3
- Segment Anything: Apache 2.0

We are still working on improving the code reliability. For any technical issues, please contact Tong Qin <qintonguavATgmail.com>.

For commercial inquiries, please contact Shaojie Shen <eeshaojieATust.hk>.
