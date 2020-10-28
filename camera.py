import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import datasets
import numpy as np
from yolov4 import yolo

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)


def main():
    # 创建摄像头捕获模块
    cap = cv2.VideoCapture(0)

    # 创建窗口
    window_handle = cv2.namedWindow("USB Camera", cv2.WINDOW_FREERATIO)
    cv2.moveWindow("USB Camera", 130, 50)
    cv2.resizeWindow("USB Camera", 1280, 720)

    # 创建图像检测器
    qrDecoder = cv2.QRCodeDetector()

    # 逐帧显示
    while cv2.getWindowProperty("USB Camera", 0) >= 0:
        ret_val, img = cap.read()
        # print(img.shape)

        # 图像太大需要调整
        height, width = img.shape[0:2]
        if width > 1280:
            new_width = 1280
            new_height = int(new_width / width * height)
            img = cv2.resize(img, (new_width, new_height))

        img = yolo(img)

        # 显示图像
        cv2.imshow("USB Camera", img)

        keyCode = cv2.waitKey(30) & 0xFF
        if keyCode == 27:  # ESC键退出
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    pass


if __name__ == '__main__':
    main()