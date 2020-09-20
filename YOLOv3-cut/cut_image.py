
import os
import cv2 as cv
import argparse
import numpy as np
import cv2
from PIL import Image
import pytesseract as tess
from PIL import Image
import glob
import time
from keras.utils import multi_gpu_model


start = time.clock()
weightsPath = "D:/YOLOV3-cut/model_data/yolov3.weights"
configPath = "D:/YOLOV3-cut/yolov3.cfg"
labelsPath = "D:/YOLOV3-cut/model_data/coco_classes.txt"
rootdir = "D:/YOLOV3-cut/images1"   #图像读取地址
savepath = "D:/YOLOV3-cut/images2"  # 图像保存地址
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#初始化一些参数
LABELS = open(labelsPath).read().strip().split("\n")  #物体类别
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")#颜色
filelist = os.listdir(rootdir)  # 打开对应的文件夹
total_num = len(filelist)  # 得到文件夹中图像的个数
print(total_num)
# 如果输出的文件夹不存在，创建即可
if not os.path.isdir(savepath):
    os.makedirs(savepath)
        
for(dirpath,dirnames,filenames) in os.walk(rootdir):
    for filename in filenames:
    # 必须将boxes在遍历新的图片后初始化
        boxes = []
        confidences = []
        classIDs = []
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        path = os.path.join(dirpath,filename)
        image = cv.imread(path)
        (H, W) = image.shape[:2]
    # 得到 YOLO需要的输出层
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        #从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        print(layerOutputs)
        #在每层输出上循环
        for output in layerOutputs:
            # 对每个检测进行循环
            for detection in output:
                scores = detection[14:15]
                classID = np.argmax(scores)
                confidence = scores[classID]
                #过滤掉那些置信度较小的检测结果
                if confidence > 0.5:
                    #框后接框的宽度和高度
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    #边框的左上角
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # 更新检测出来的框
                   # 批量检测图片注意此处的boxes在每一次遍历的时候要初始化，否则检测出来的图像框会叠加
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # 极大值抑制
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.05, 0.05)
        k = -1
        if len(idxs) > 0:
            # for k in range(0,len(boxes)):
            #for i in idxs.flatten() :
            for index, i in enumerate(idxs.flatten()):
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                if 0.3 < h/w < 1.4:
                    # 在原图上绘制边框和类别
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    # image是原图，     左上点坐标， 右下点坐标， 颜色， 画线的宽度
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    # 各参数依次是：图片，添加的文字，左上角坐标(整数)，字体，        字体大小，颜色，字体粗细
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                    # 图像裁剪注意坐标要一一对应
                    # 图片裁剪 裁剪区域【Ly:Ry,Lx:Rx】
                    cut = image[y:(y + h), x:(x + w)]
                    #print(cut)

                    # boxes的长度即为识别出来的车辆个数，利用boxes的长度来定义裁剪后车辆的路径名称
                    #if k < len(boxes):
                       # k = k+1
                   # 从字母a开始每次+1
                    #t = chr(ord("a")+k)
                    # 写入文件夹，这块写入的时候不支持int（我也不知道为啥），所以才用的字母
                    #cv.imwrite(savepath+"/"+filename.split(".")[0]+"_"+t+".jpg",cut)
                    #cv.imwrite(savepath + "/" + filename.split(".")[0] + "_{0}.jpg".format(index), cut)
                    cv.imwrite("D:/YOLOV3-cut/images2/demo.jpg",cut)
                    #cv.imwrite("demo.jpg", cut)
                else:
                    print("正常退出")

'''
def recoginse_text(image):
    """
    步骤：
    1、灰度，二值化处理
    2、形态学操作去噪
    3、识别
    :param image:
    :return:
    """

    # 灰度 二值化
    #inverse_img = cv.bitwise_not(image)  # 图像取反操作2，比较快
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 全局图像二值化
    cv.imshow('binary_image', binary)
    textImage  = cv.bitwise_not(binary)
    #textImage = Image.fromarray(binary)
    #b, g, r = cv.split(inverse_img)  # RGB三通道分离
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #cv.imshow('hsv1', hsv)
    #mask1 = cv.inRange(hsv, (0, 0, 0), (180, 255, 46))
    cv.imshow('textImage', textImage)
    #mask2 = cv.bitwise_not(mask1)
    #cv.imshow('mask2', mask2)
    #timg = cv.bitwise_and(image, image, mask=mask1)
    #timg2 = cv.bitwise_not(timg)
    #mask2 = cv.bitwise_not(mask1)
    #cv.imshow('mask2',  timg2)
    #cv.imshow('timg2', timg2)
    #cv.imwrite('timg.jpg', timg)
    #background = np.zeros(image.shape, image.dtype)
    #background[:, :, 0] = 255
    #cv.imshow('background', background)
    #mask = cv.bitwise_not(mask2)
    #dst = cv.bitwise_or(timg, background, mask=timg2)
    #cv.imshow('dst1', dst)
    #cv.imwrite('dst1.jpg', dst)
    #dst = cv.add(dst, timg)
    #cv.imshow('dst2', dst)
    #cv.imwrite('dst2.jpg', dst)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    #inverse_img = cv.bitwise_not(timg)
    #cv.imshow('timg1', timg)
    #gray = cv.cvtColor(inverse_img, cv.COLOR_BGR2GRAY)
    #b, g, r = cv.split(timg)
    #cv.imshow('b', b)
    #cv.imshow('g', g)
    #cv.imshow('r', r)
    #cv.imshow('gray', gray)
    #mask3 = cv.inRange(gray, 20, 255)
    #mask4 = cv.bitwise_not(mask3)
    #binary = cv.adaptiveThreshold(mask4, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 8)  #局部二值化阈值分割
    #ret, binary = cv.threshold(mask4, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 全局图像二值化
    #mask4 = cv.bitwise_not(gray)
    #ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 全局图像二值化
    #textImage = Image.fromarray(mask4)
    #print("thresh value %s"%ret)
    #cv.imshow('mask3', mask3)
    #cv.imshow('mask4', mask4)
    #cv.imshow('binary', binary)
    #cv.imwrite("D:/YOLOV3-cut/images2/cut.jpg", binary)
    # 黑底白字取非，变为白底黑字（便于pytesseract 识别）
    #cv.bitwise_not(binary, binary)
    #textImage = Image.fromarray(binary)
    #cv.imshow('textImage', textImage)
    # 图片转文字
    #tt = cv.imread("D:/keras-yolo3-master-Experiment-cut/images2/cut.jpg")
    #text = tess.image_to_string(textImage)
    #print("识别结果：%s"%text)


def main():
    # 读取需要识别的数字字母图片，并显示读到的原图
    src = cv.imread("D:/YOLOV3-cut/images2/demo.jpg")
    cv.imshow("src", src)

    # 识别
    recoginse_text(src)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
'''
'''
src = cv.imread("D:/keras-yolo3-master-Experiment-cut/images2/999.PNG")
#inverse_img = 255 - src  #图像取反操作1，比较慢
#inverse_img = cv.bitwise_not(src)  #图像取反操作2，比较快
gary = cv.cvtColor(src,cv.COLOR_BGR2GRAY)  #灰度转换
#b, g, r = cv.split(inverse_img)  #RGB三通道分离
#ret, binary = cv.threshold(gary, 0, 255, cv2.THRESH_TRIANGLE)  #图像二值化
ret, binary = cv.threshold(gary, 0, 255, cv2.THRESH_TRIANGLE)  #图像二值化
text = tess.image_to_string(binary)  #OCR字符识别
print("识别结果：%s"%text)
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",gary)
cv.waitKey(0)
cv.destroyAllWindows()
end = time.clock()
print('time: f% s' %(end-start))
'''
'''
time.sleep(10)
path = "D:/keras-yolo3-master-Experiment-cut/images2"

paths = glob.glob(os.path.join(path, '*.jpg'))

# 输出所有文件和文件夹
for file in paths:
    fp = open(file, 'rb')
    img = Image.open(fp)
    fp.close()
    width = img.size[0]
    height = img.size[1]

    if (width != 50) or (height != 50):
        os.remove(file)
'''



