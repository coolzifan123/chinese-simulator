

photo_path = r'../ICCV2019-LearningToPaint/baseline/home/liuyt/FontRL/data/font/2'
path = r'../ICCV2019-LearningToPaint/baseline/image/3/GB5_R.bmp'
import cv2,time
import numpy as np
import glob
from skimage import morphology


def Three_element_add(array):
    array0 = array[:]
    array1 = np.append(array[1:],np.array([0]))
    array2 = np.append(array[2:],np.array([0, 0]))
    arr_sum = array0 + array1 + array2
    return arr_sum[:-2]


def VThin(image, array):
    NEXT = 1
    height, width = image.shape[:2]
    for i in range(1,height):
        M_all = Three_element_add(image[i])
        for j in range(1,width):
            if NEXT == 0:
                NEXT = 1
            else:
                M = M_all[j-1] if j<width-1 else 1
                if image[i, j] == 0 and M != 0:
                    a = np.zeros(9)
                    if height-1 > i and width-1 > j:
                        kernel = image[i - 1:i + 2, j - 1:j + 2]
                        a = np.where(kernel == 255, 1, 0)
                        a = a.reshape(1, -1)[0]
                    NUM = np.array([1,2,4,8,0,16,32,64,128])
                    sumArr = np.sum(a*NUM)
                    image[i, j] = array[int(sumArr)] * 255
                    if array[int(sumArr)] == 1:
                        NEXT = 0
    return image


def HThin(image, array):
    height, width = image.shape[:2]
    NEXT = 1
    for j in range(1,width):
        M_all = Three_element_add(image[:,j])
        for i in range(1,height):
            if NEXT == 0:
                NEXT = 1
            else:
                M = M_all[i-1] if i < height - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = np.zeros(9)
                    if height - 1 > i and width - 1 > j:
                        kernel = image[i - 1:i + 2, j - 1:j + 2]
                        a = np.where(kernel == 255, 1, 0)
                        a = a.reshape(1, -1)[0]
                    NUM = np.array([1, 2, 4, 8, 0, 16, 32, 64, 128])
                    sumArr = np.sum(a * NUM)
                    image[i, j] = array[int(sumArr)] * 255
                    if array[int(sumArr)] == 1:
                        NEXT = 0
    return image


def Xihua(binary, array, num=10):
    binary_image = binary.copy()
    image = cv2.copyMakeBorder(binary_image, 1, 0, 1, 0, cv2.BORDER_CONSTANT, value=0)
    for i in range(num):
        VThin(image, array)
        HThin(image, array)
    return image


array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]


if __name__ == '__main__':
    #img = list(sorted(glob.glob(photo_path + '/generated*.*')))

    image = cv2.imread(path, 0)
    ret, binary = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)

    binary[binary == 255] = 1
    binary=1-binary
    skeleton0 = morphology.skeletonize(binary)
    skeleton0 = 1 - skeleton0
    skeleton = skeleton0.astype(np.uint8) * 255
    cv2.imwrite("images/3ai.png", skeleton)

    # cv2.imshow('image', image)
    # cv2.imshow('binary', binary)
    # cv2.waitKey(0)
    #
    # t1 = time.time()
    #
    # iThin = Xihua(binary, array)
    # t2 = time.time()
    # print('cost time:',t2-t1)
    # cv2.imshow('iThin', iThin)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('images/chu.png',iThin)


