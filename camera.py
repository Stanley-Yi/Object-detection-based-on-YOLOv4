import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import cv2
from yolov4 import yolo

# Specifies the use of GPUs
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)


def main():
    # Create the camera capture module
    cap = cv2.VideoCapture(0)

    # Create a window
    window_handle = cv2.namedWindow("USB Camera", cv2.WINDOW_FREERATIO)
    cv2.moveWindow("USB Camera", 130, 50)
    cv2.resizeWindow("USB Camera", 1280, 720)

    # Create image detector
    qrDecoder = cv2.QRCodeDetector()

    # display frame by frame
    while cv2.getWindowProperty("USB Camera", 0) >= 0:
        ret_val, img = cap.read()

        # Resize image
        height, width = img.shape[0:2]
        if width > 1280:
            new_width = 1280
            new_height = int(new_width / width * height)
            img = cv2.resize(img, (new_width, new_height))

        img = yolo(img)

        # display image
        cv2.imshow("USB Camera", img)

        keyCode = cv2.waitKey(30) & 0xFF
        if keyCode == 27:  # Press the ESC key to exit
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    main()