Create venv environment

```commandline
python -m venv camera
```

Activate

```commandline
.\camera\Scripts\activate
```

Upgrade pip
```commandline
python.exe -m pip install --upgrade pip
```

Install
```commandline
pip install .\requirements.txt 
```

```commandline
python capture_frames.py --out ./calib_images
```

```commandline
# 第一步：标定（准备 10~20 张棋盘格照片放一个文件夹）
python fisheye_undistort.py calibrate ./calib_images --rows 6 --cols 9 --square_size 25

# 第二步：用标定结果去畸变
python fisheye_undistort.py undistort input.jpg --params fisheye_params.npz --balance 0.5
```