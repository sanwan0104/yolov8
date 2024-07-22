# CheckYOLOLabels.py
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def listPathAllfiles(dirname):
    """
    遍历指定目录下的所有文件并返回一个包含这些文件路径的列表。
    """
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result

if __name__ == '__main__':
    # YOLO标签文件的保存路径
    labelspath = "D:\\machine_learning\\yolo8\\ultralytics-main\\datasets\\val\\labels"
    # YOLO图片文件的保存路径
    imagespath = "D:\\machine_learning\\yolo8\\ultralytics-main\\datasets\\val\\images"

    # 获取所有标签文件的路径
    labelsFiles = listPathAllfiles(labelspath)

    # 逆序遍历标签文件，因为通常最新的文件在最后
    for lbf in labelsFiles[::-1]:
        # 读取标签文件的每一行，并将其分割成一个列表
        labels = open(lbf, "r").readlines()
        labels = list(map(lambda x: x.strip().split(" "), labels))
        # 构造对应的图片文件名
        imgfileName = os.path.join(imagespath, os.path.basename(lbf)[:-4] + ".jpg")
        # 从文件中读取图片，cv2.imdecode函数可以将字节流解码为图像
        img = cv2.imdecode(np.fromfile(imgfileName, dtype=np.uint8), 1)

        # 遍历每个标签
        for lbs in labels:
            # 将标签字符串转换为浮点数，并去掉类别索引
            lb = list(map(float, lbs))[1:]
            # 根据标签计算边界框的左上角和右下角坐标
            x1 = int((lb[0] - lb[2] / 2) * img.shape[1])
            y1 = int((lb[1] - lb[3] / 2) * img.shape[0])
            x2 = int((lb[0] + lb[2] / 2) * img.shape[1])
            y2 = int((lb[1] + lb[3] / 2) * img.shape[0])
            # 在图像上绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

        # 调整图像大小，使其最大边长为600像素
        ratio = 600 / min(img.shape[0:2])
        img = cv2.resize(img, dsize=(int(img.shape[1] * ratio), int(img.shape[0] * ratio)))

        # 显示带有边界框的图像
        cv2.imshow("1", img)
        # 等待用户按键，按任意键继续
        cv2.waitKey()
        # 关闭所有OpenCV创建的窗口
        cv2.destroyAllWindows()
