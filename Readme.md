# 鱼眼相机标定与去畸变工具集

基于 OpenCV `cv2.fisheye` 模块（等距投影模型）的鱼眼相机标定与去畸变工具，支持单目和双目（左右拼接）相机。

## 文件说明

| 文件 | 作用 |
|---|---|
| `capture_frames.py` | 打开摄像头，按 `s` 键保存当前帧，用于采集标定图片 |
| `fisheye_undistort.py` | 标定鱼眼相机内参 K / 畸变系数 D，并对图像做去畸变处理 |
| `requirements.txt` | 依赖列表 |

## 环境安装

```bash
pip install -r requirements.txt --break-system-packages
```

> 推荐使用 `opencv-contrib-python`（而非 `opencv-python`），以确保 `cv2.fisheye` 模块完整可用：
>
> ```bash
> pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
> pip install opencv-contrib-python --break-system-packages
> ```

## 使用流程

### 第一步：采集标定图片

准备一张棋盘格标定板，用摄像头从不同角度、不同位置拍摄 50 张以上照片。

```bash
python capture_frames.py --out ./calib_images
```

- 预览窗口按 **s** 保存当前帧，按 **q** 或 **ESC** 退出
- 常用参数：`--camera` 指定摄像头编号，`--width/--height` 指定分辨率，`--prefix/--ext` 指定文件名前缀和格式
- 已保存的图片按编号递增命名，重复运行不会覆盖之前的采集结果

### 第二步：标定相机

**单目相机：**

```bash
python fisheye_undistort.py calibrate ./calib_images --rows 6 --cols 9 --square_size 25 --out fisheye_params.npz
```

**双目相机（左右拼接图像）：**

双目的两个镜头畸变参数通常不同，需要分别标定，各自保存一份参数：

```bash
python fisheye_undistort.py calibrate ./calib_images --rows 6 --cols 9 --split left  --out left_params.npz
python fisheye_undistort.py calibrate ./calib_images --rows 6 --cols 9 --split right --out right_params.npz
```

参数说明：

- `--rows / --cols`：棋盘格**内角点**数量（不是格子数，如 10×7 格子对应 9×6 内角点）
- `--square_size`：棋盘格每格实际边长，只影响外参尺度，不影响去畸变效果
- `--split {left,right}`：双目拼接图像先从正中间左右切开，只取一侧参与标定（不传则按单目处理）

标定完成会打印重投影误差 **RMS**（正常应 < 1.0），以及标定得到的 K、D 数值。

### 第三步：图像去畸变

**单目：**

```bash
python fisheye_undistort.py undistort input.jpg --params fisheye_params.npz --balance 0.5 --out undistorted.jpg
```

**双目：**

```bash
python fisheye_undistort.py undistort stereo_frame.jpg --split left  --params left_params.npz  --out left_undistorted.jpg
python fisheye_undistort.py undistort stereo_frame.jpg --split right --params right_params.npz --out right_undistorted.jpg
```

参数说明：

- `--balance`：0~1，控制视场取舍。0 = 尽量裁掉黑边（可能损失部分视野边缘），1 = 保留全部视野（四周有黑边）
- `--split`：需要与去畸变的那一侧对应的 `--params` 文件配套使用

## 常见问题

**报错 `AttributeError: module 'cv2.fisheye' has no attribute 'CALIB_RECOMPUTE_EXTRINSIC'`**

通常是 OpenCV 安装本身有问题（多个 opencv 包冲突，或本地文件名与 `cv2` 冲突），而不是代码问题：

```bash
pip list | findstr opencv   # 检查是否装了多个opencv包
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
pip install opencv-contrib-python --break-system-packages
```

**去畸变结果只显示图像的一角**

最常见的两个原因：

1. **标定分辨率与待处理图像分辨率不一致** —— 脚本已自动检测并按比例缩放 K，会打印提示
2. **标定质量差（RMS 过高）** —— 检查 `--rows/--cols` 是否与实际棋盘格一致，必要时重新采集图片、重新标定

## Python API

也可以在代码里直接调用，不走命令行：

```python
from fisheye_undistort import calibrate_fisheye, undistort_image, split_stereo_image
import cv2

# 标定
K, D, size, rms = calibrate_fisheye("./calib_images", checkerboard=(9, 6), split="left")

# 去畸变
img = cv2.imread("stereo_frame.jpg")
result, new_K = undistort_image(img, K, D, balance=0.5, calib_size=size, split="left")
```
