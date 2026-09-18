"""
鱼眼相机畸变校正
==================
使用 OpenCV 的 cv2.fisheye 模块完成两件事：
1. calibrate_fisheye(): 用棋盘格标定板拍摄的多张图片，标定出相机内参 K 和畸变系数 D
2. undistort_image(): 用标定得到的 K、D 对图像做去畸变处理

鱼眼镜头视场角很大（常见 >150°），普通的 cv2.calibrateCamera / cv2.undistort
（针孔模型）对畸变很大的鱼眼图像效果很差，必须用专门的 cv2.fisheye 系列函数
（等距投影模型 equidistant model）。

依赖: pip install opencv-python numpy --break-system-packages
"""

import cv2
import numpy as np
import glob
import os


def split_stereo_image(img, side="left"):
    """
    将左右拼接的双目图像从正中间左右分割，取出一侧。

    参数:
        img:  完整的双目拼接图像 (BGR, np.ndarray)，宽度应为单目图像宽度的2倍
        side: "left" 取左半边，"right" 取右半边

    返回:
        分割后的单目图像
    """
    h, w = img.shape[:2]
    half_w = w // 2
    if side == "left":
        return img[:, :half_w]
    elif side == "right":
        return img[:, half_w:]
    else:
        raise ValueError(f"side 必须是 'left' 或 'right'，收到: {side}")


def calibrate_fisheye(image_dir, checkerboard=(9, 6), square_size=1.0,
                       image_ext="jpg", show_corners=False, split=None):
    """
    用一组棋盘格标定图片，标定鱼眼相机的内参矩阵 K 和畸变系数 D。

    参数:
        image_dir:    存放标定图片的文件夹路径
        checkerboard: 棋盘格内角点数量 (宽方向角点数, 高方向角点数)
                      注意是"内角点"，比如 10x7 的格子对应 9x6 个内角点
        square_size:  棋盘格每格的实际边长（单位任意，如 mm/cm，只影响外参/尺度）
        image_ext:    图片后缀，如 'jpg' / 'png'
        show_corners: 是否弹窗显示检测到的角点（调试用）
        split:        双目拼接图像的分割方式，None 表示不分割（单目相机）；
                      "left" / "right" 表示先从图像正中间左右切开，只用其中一侧
                      来标定该侧相机。双目需要分别跑两次（split="left" 和
                      split="right"），得到左右两组独立的 K/D。

    返回:
        K, D, image_size, rms_error
    """
    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    # 用 getattr 容错：极少数损坏/不完整的 opencv 安装会缺失个别 flag，
    # 缺失时该项按 0（不启用）处理，而不是直接 AttributeError 崩溃。
    # 正常安装（推荐 opencv-contrib-python）这三个 flag 都应该存在。
    if not hasattr(cv2, "fisheye") or not hasattr(cv2.fisheye, "calibrate"):
        raise RuntimeError(
            "当前 OpenCV 安装缺少 cv2.fisheye 模块，请重新安装: "
            "pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y && "
            "pip install opencv-contrib-python --break-system-packages"
        )

    calibration_flags = (
        getattr(cv2.fisheye, "CALIB_RECOMPUTE_EXTRINSIC", 0)
        + getattr(cv2.fisheye, "CALIB_CHECK_COND", 0)
        + getattr(cv2.fisheye, "CALIB_FIX_SKEW", 0)
    )

    # 世界坐标系下的角点坐标 (z=0 平面), 形状要求为 (1, N, 3)
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3D 世界坐标点
    imgpoints = []  # 2D 图像坐标点

    images = sorted(glob.glob(os.path.join(image_dir, f"*.{image_ext}")))
    if not images:
        raise FileNotFoundError(f"在 {image_dir} 下没有找到 *.{image_ext} 图片")

    image_size = None
    used = 0
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        if split is not None:
            img = split_stereo_image(img, side=split)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # (w, h)

        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            corners_sub = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            objpoints.append(objp)
            imgpoints.append(corners_sub)
            used += 1
            if show_corners:
                vis = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners_sub, ret)
                cv2.imshow("corners", vis)
                cv2.waitKey(200)
        else:
            print(f"[跳过] 未检测到棋盘格角点: {fname}")

    if show_corners:
        cv2.destroyAllWindows()

    if used < 5:
        raise RuntimeError(f"有效标定图片太少 ({used} 张)，建议至少 10~20 张不同角度/位置的照片")

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, image_size, K, D, rvecs, tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    print(f"标定完成，使用了 {used}/{len(images)} 张图片，重投影误差 RMS = {rms:.4f}")
    print("K =\n", K)
    print("D =\n", D.ravel())
    return K, D, image_size, rms


def undistort_image(img, K, D, balance=0.0, new_size=None, calib_size=None, split=None):
    """
    对单张图像做鱼眼去畸变。

    参数:
        img:        输入图像 (BGR, np.ndarray)
        K, D:       calibrate_fisheye() 得到的内参与畸变系数
        balance:    0~1，控制视场取舍。
                    0 = 尽量裁掉黑边，但可能丢失部分视野边缘
                    1 = 保留全部原始视野，但四周会有黑边
        new_size:   输出图像尺寸 (w, h)，默认与输入相同
        calib_size: 标定时用的图像尺寸 (w, h)。若与当前 img 尺寸不同，
                    会自动按比例缩放 K，否则会出现"只显示图像一角"的错误结果。
        split:      若输入是双目拼接图像，先按 "left"/"right" 从正中间分割
                    出一侧再做去畸变（要求该侧对应标定得到的 K/D）。
                    None 表示 img 本身已经是单目图像，不做分割。

    返回:
        去畸变后的图像, 实际使用的 new_K
    """
    if split is not None:
        img = split_stereo_image(img, side=split)

    h, w = img.shape[:2]
    dim = new_size if new_size else (w, h)

    # 标定分辨率与当前图像分辨率不一致时，K 必须按比例缩放，否则焦距/主点
    # 完全对不上当前图像尺度，remap 会把大部分像素采样到源图外(变黑)，
    # 只剩很小一块碰巧对齐 —— 这就是"结果只显示一角"最常见的原因。
    if calib_size is not None and tuple(calib_size) != (w, h):
        scale_x = w / calib_size[0]
        scale_y = h / calib_size[1]
        K = K.copy()
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy
        print(f"[提示] 标定尺寸 {calib_size} 与当前图像尺寸 {(w, h)} 不一致，已自动缩放 K")

    # 根据 balance 重新估计一个新的相机矩阵，避免图像边缘被裁剪过多或黑边过多
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, dim, cv2.CV_16SC2
    )
    undistorted = cv2.remap(
        img, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    return undistorted, new_K


def main():
    import argparse
    parser = argparse.ArgumentParser(description="鱼眼相机标定与去畸变")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("calibrate", help="用棋盘格图片标定相机，输出 K/D 到 npz 文件")
    p1.add_argument("image_dir", help="棋盘格标定图片所在文件夹")
    p1.add_argument("--rows", type=int, default=6, help="棋盘格内角点行数")
    p1.add_argument("--cols", type=int, default=9, help="棋盘格内角点列数")
    p1.add_argument("--square_size", type=float, default=1.0)
    p1.add_argument("--split", choices=["left", "right"], default=None,
                     help="双目拼接图像先从中间分割，取左/右哪一侧来标定（不传则按单目处理）")
    p1.add_argument("--out", default="fisheye_params.npz")

    p2 = sub.add_parser("undistort", help="用已标定的 K/D 对图像去畸变")
    p2.add_argument("image_path", help="待处理图像路径")
    p2.add_argument("--params", default="fisheye_params.npz", help="标定参数 npz 文件")
    p2.add_argument("--balance", type=float, default=0.5)
    p2.add_argument("--split", choices=["left", "right"], default=None,
                     help="输入是双目拼接图像时，先分割出左/右哪一侧再去畸变（需配合对应侧的params）")
    p2.add_argument("--out", default="undistorted.jpg")

    args = parser.parse_args()

    if args.cmd == "calibrate":
        K, D, size, rms = calibrate_fisheye(
            args.image_dir, checkerboard=(args.cols, args.rows),
            square_size=args.square_size, split=args.split
        )
        np.savez(args.out, K=K, D=D, size=size, rms=rms)
        print(f"参数已保存到 {args.out}")

    elif args.cmd == "undistort":
        data = np.load(args.params)
        K, D = data["K"], data["D"]
        calib_size = tuple(data["size"]) if "size" in data else None
        rms = float(data["rms"]) if "rms" in data else None

        if rms is not None and rms > 1.0:
            print(f"[警告] 标定重投影误差 RMS = {rms:.3f} 偏高（正常应 < 1.0），"
                  f"K/D 可能不准，去畸变结果可能异常，建议重新标定")

        img = cv2.imread(args.image_path)
        if img is None:
            raise FileNotFoundError(args.image_path)
        result, new_K = undistort_image(img, K, D, balance=args.balance,
                                          calib_size=calib_size, split=args.split)
        cv2.imwrite(args.out, result)
        print(f"去畸变结果已保存到 {args.out}")


if __name__ == "__main__":
    main()