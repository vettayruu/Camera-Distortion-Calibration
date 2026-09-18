"""
摄像头图像采集工具
==================
打开摄像头实时预览，按 's' 键保存当前帧到指定文件夹（用于收集鱼眼标定图片等场景），
按 'q' 或 ESC 退出。

依赖: pip install opencv-python --break-system-packages
用法:
    python capture_frames.py                     # 默认摄像头0，保存到 ./captured
    python capture_frames.py --camera 1           # 指定摄像头编号
    python capture_frames.py --out ./calib_images # 指定保存目录
    python capture_frames.py --prefix img --ext png
"""

import cv2
import os
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="按键保存摄像头帧")
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备编号")
    parser.add_argument("--out", default="./captured", help="图片保存目录")
    parser.add_argument("--prefix", default="frame", help="文件名前缀")
    parser.add_argument("--ext", default="jpg", help="保存格式，如 jpg/png")
    parser.add_argument("--width", type=int, default=None, help="摄像头分辨率宽（可选）")
    parser.add_argument("--height", type=int, default=None, help="摄像头分辨率高（可选）")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera}")

    # 已存在的编号继续往后累加，避免覆盖之前采集的图片
    count = 0
    existing = [f for f in os.listdir(args.out) if f.startswith(args.prefix)]
    if existing:
        nums = []
        for f in existing:
            try:
                n = int(os.path.splitext(f)[0].replace(args.prefix + "_", ""))
                nums.append(n)
            except ValueError:
                pass
        if nums:
            count = max(nums) + 1

    print("按 's' 保存当前帧，按 'q' 或 ESC 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头帧失败")
            break

        display = frame.copy()
        text = f"Saved: {count}  (s=save, q=quit)"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Capture - press s to save, q to quit", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            filename = os.path.join(args.out, f"{args.prefix}_{count:04d}.{args.ext}")
            cv2.imwrite(filename, frame)
            print(f"[已保存] {filename}")
            count += 1

            # 保存瞬间做一个简单的闪烁反馈
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), (255, 255, 255), -1)
            cv2.imshow("Capture - press s to save, q to quit", flash)
            cv2.waitKey(60)

        elif key == ord('q') or key == 27:  # 27 = ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"共采集 {count} 张图片，保存在 {args.out}")


if __name__ == "__main__":
    main()