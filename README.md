# 🚗 Advanced Driver Assistance System

This project is a lightweight simulation of an **Advanced Driver Assistance System** — the kind of technology used in modern cars to improve safety and driving comfort.  
It processes a driving video frame-by-frame, performing:

- **Lane detection** to keep the vehicle centered
- **Object detection** to identify cars, pedestrians, and obstacles
- **Distance estimation** to predict potential collisions
- **Bird’s-eye view transformation** for better visualization
- **Provide real-time driving metrics and status** for safe driving

## 🎥 Demo
<p align="center">
  <img src="Examples/short_test_video_result.gif" alt="ADAS Demo" width="480"/>
</p>

👉 **Watch the full output video here:** [Examples/test_video_result.mp4](Examples/test_video_result.mp4)

**Note**: This project is based on the GitHub repository: [Vehicle-CV-ADAS](https://github.com/jason-li-831202/Vehicle-CV-ADAS)

---

## ✨ Features

### 1. Lane Detection
- Using [Ultra Fast Lane Detection v2 (UFLDv2)](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) model for detecting lane boundaries in each frame
- Outputs lane curvature (radius) and vehicle offset from the center.
- Provides lane status information for driving safety.

### 2. Object Detection
- Using [YOLOv8](https://github.com/ultralytics/ultralytics) for detecting vehicles, pedestrians, and road obstacles.
- Tracks detected objects frame-to-frame with [ByteTrack](https://github.com/FoundationVision/ByteTrack).

### 3. High-Performance Inference
- Both **UFLDv2** and **YOLOv8** models are **converted to TensorRT** format for optimized GPU performance and lower latency.

### 4. Core Functions
- **Distance computation** – Estimates the distance to detected objects and predicts potential collision points.
- **Bird’s-eye view transformation** – Transforms the camera perspective into a top-down view for better lane visualization.
- **Driving metrics computation** – Calculates:
  - Lane curvature (turn radius)
  - Vehicle lateral offset
  - Vehicle movement direction
- **Driving status monitoring** – Provides real-time feedback such as:
  - *Safe*
  - *Warning*
  - *Collision risk*
- **On-screen visualization** – Displays:
  - Lane lines and detection results
  - Bounding boxes for detected objects
  - Bird’s-eye view lane overlay
  - Driving metrics and warnings

---

## 🛠 Implementation & Usage

### 1. Prerequisites
- **Python 3.8+**
- Required Python packages (listed in `requirements.txt`)

Install dependencies:
```
pip install -r requirements.txt
```

### 2. Model Preparation

1. **Download pre-trained models**:

   Use this [link](https://drive.google.com/drive/folders/1vivD7YJ2D-w9BcPidgr-u9xc8zwG1sc0?usp=sharing) to download the prepared models

3. **Convert PyTorch models to ONNX**:  
 - For **object detection** (YOLOv8):
   ```
   python ObjectDetector/convertPytorchToONNX.py
   ```
 - For **lane detection** (UFLDv2):
   ```
   python TrafficLaneDetector/convertPytorchToONNX.py
   ```
3. **Convert ONNX models to TensorRT** *(optional, recommended for speed)*:  
   ```
   python convertOnnxToTensorRT.py --i <path-of-onnx-model>  -o <path-of-trt-model>
   ```
### 3. Run Inference
   ```
   python video_inference.py --video_path <path-of-input-video> --lane_det_model_path <path-of-lane-detection-model> --object_det_model_path <path-of-object-detection-model>
   ```
