# Cam Hand Control 🤖✋

Real-time control of dual **Shadow Hands** in **MuJoCo** using webcam hand tracking with **MediaPipe** and **OpenCV**.

## 📋 Description

This project tracks both hands through a webcam and converts their movements into joint commands for a dual Shadow Hand simulation.

The system detects the **21 MediaPipe landmarks** of each hand and calculates:

* Finger flexion.
* Finger lateral movement.
* Thumb movement and rotation.
* Wrist orientation.
* Forearm rotation.
* Left/right hand orientation.

The calculated movements are normalized to values between `0.0` and `1.0` and mapped to the MuJoCo actuators.

## 📂 Project Structure

```text
cam_hand_control/
│
├── main.py
├── hand_mapping.py
│
├── shadow_hand/
│   ├── scene_dual.xml
│   └── assets/
│
└── README.md
```

### Main files

**`main.py`**

Handles:

* Webcam capture.
* MediaPipe hand tracking.
* MuJoCo simulation.
* Real-time actuator control.
* Movement smoothing.

**`hand_mapping.py`**

Contains the kinematic calculations used to convert MediaPipe landmarks into normalized joint commands.

**`shadow_hand/scene_dual.xml`**

Contains the MuJoCo model of the two Shadow Hands, including joints, actuators, meshes, and simulation configuration.

## ⚙️ Requirements

* Python 3.x
* Webcam
* NumPy
* OpenCV
* MediaPipe
* MuJoCo

## 📦 Installation

Install the required Python packages:

```bash
pip install numpy opencv-python mediapipe mujoco
```

Then enter the project directory:

```bash
cd cam_hand_control
```

## 🚀 Run

Start the application with:

```bash
python main.py
```

Two windows will be displayed:

* **Camera:** shows the webcam image and detected hand landmarks.
* **MuJoCo:** shows the simulated dual Shadow Hands.

Press **`Q`** in the camera window to exit.

## 🖐️ Hand Mapping

MediaPipe provides 21 landmarks per hand:

```text
0       Wrist
1-4     Thumb
5-8     Index
9-12    Middle
13-16   Ring
17-20   Little
```

The landmarks are processed using geometric calculations such as angles, vectors, dot products, and cross products.

The resulting values are mapped to the corresponding MuJoCo joints, including:

```text
WRJ1 / WRJ2    Wrist
FRJ            Forearm rotation
THJ1-THJ5      Thumb
FFJ0-FFJ4      Index
MFJ0-MFJ4      Middle
RFJ0-RFJ4      Ring
LFJ0-LFJ5      Little
```

## 🔄 Smoothing

To reduce sudden movements caused by variations in hand tracking, the actuator commands use a smoothing factor:

```python
SMOOTHING = 0.20
```

This provides smoother motion while maintaining real-time response.

## 🛠️ Technologies

| Technology  | Purpose                   |
| ----------- | ------------------------- |
| Python      | Main programming language |
| MediaPipe   | Hand landmark detection   |
| OpenCV      | Webcam processing         |
| NumPy       | Kinematic calculations    |
| MuJoCo      | Robotic hand simulation   |
| Shadow Hand | Robotic hand model        |

Real-time vision-based control of dual Shadow Hands using MediaPipe and MuJoCo.