import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import time
import cv2

LABELS = open("coco.names").read().strip().split("\n")
np.random.seed(666)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# 导入 YOLO 配置和权重文件并加载网络：
net = cv2.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')
# 获取 YOLO 未连接的输出图层
layer = net.getUnconnectedOutLayersNames()
def yolo(img):
    image = img
    # 获取图片尺寸
    (H, W) = image.shape[:2]
    # 从输入图像构造一个 blob，然后执行 YOLO 对象检测器的前向传递，给我们边界盒和相关概率
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    # 前向传递，获得信息
    layerOutputs = net.forward(layer)
    # 用于得出检测时间
    end = time.time()
    print("YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []

    # 循环提取每个输出层
    for output in layerOutputs:
        # 循环提取每个框
        for detection in output:
            # 提取当前目标的类 ID 和置信度
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # 通过确保检测概率大于最小概率来过滤弱预测
            if confidence > 0.5:
                # 将边界框坐标相对于图像的大小进行缩放，YOLO 返回的是边界框的中心(x, y)坐标，
                # 后面是边界框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # 转换出边框左上角坐标
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # 更新边界框坐标、置信度和类 id 的列表
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # 非最大值抑制，确定唯一边框
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # 确定每个对象至少有一个框存在
    if len(idxs) > 0:
        # 循环画出保存的边框
        for i in idxs.flatten():
            # 提取坐标和宽度
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # 画出边框和标签
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1, lineType=cv2.LINE_AA)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, lineType=cv2.LINE_AA)

    # cv2.imshow("Tag", image)
    # cv2.waitKey(0)
    return image
