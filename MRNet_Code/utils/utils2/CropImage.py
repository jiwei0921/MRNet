import cv2
import numpy as np


def cropImage(Image):
    height, width = Image.shape[:2]

    TempImg, TempMask = creatMask(Image, threshold=10)

    rowsMask0, colsMask0 = np.where(TempMask > 0)
    minColIndex0, maxColIndex0 = np.argmin(colsMask0), np.argmax(colsMask0)
    minCol, maxCol = colsMask0[minColIndex0], colsMask0[maxColIndex0]

    minRowIndex0, maxRowIndex0 = np.argmin(rowsMask0), np.argmax(rowsMask0)
    minRow, maxRow = rowsMask0[minRowIndex0], rowsMask0[maxRowIndex0]

    upperLimit = minRow
    lowerLimit = maxRow+1
    leftLimit =  minCol
    rightLimit = maxCol+1

    # upperLimit = np.maximum(0, minRow - 20)   #20
    # lowerLimit = np.minimum(maxRow + 20, height)   #20
    # leftLimit = np.maximum(0, minCol - 20)   #lowerLimit = np.minimum(maxCol + 50, width)   #20
    # rightLimit = np.minimum(maxCol + 20, width)

    if len(Image.shape) == 3:
        ImgCropped = Image[upperLimit:lowerLimit, leftLimit:rightLimit, :]
        MaskCropped = TempMask[upperLimit:lowerLimit, leftLimit:rightLimit]
        # ImgCropped[:20, :, :] = 0
        # ImgCropped[-20:, :, :] = 0
        # ImgCropped[:, :20, :] = 0
        # ImgCropped[:, -20:, :] = 0
    elif len(Image.shape) == 2:
        ImgCropped = Image[upperLimit:lowerLimit, leftLimit:rightLimit]
        MaskCropped = TempMask[upperLimit:lowerLimit, leftLimit:rightLimit]
        # ImgCropped[:20, :] = 0
        # ImgCropped[-20:, :] = 0
        # ImgCropped[:, :20] = 0
        # ImgCropped[:, -20:] = 0
    else:
        ImgCropped = Image.copy()
        MaskCropped = TempMask

    # ImgCropped = Image.copy()
    # MaskCropped = TempMask

    # ImgCropped = cv2.bitwise_and(ImgCropped,ImgCropped,mask=MaskCropped)
    return ImgCropped



def creatMask(Image, threshold = 10):
    ##This program try to creat the mask for the filed-of-view
    ##Input original image (RGB or green channel), threshold (user set parameter, default 10)
    ##Output: the filed-of-view mask

    if len(Image.shape) == 3: ##RGB image
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        Mask0 = gray >= threshold

    else:  #for green channel image
        Mask0 = Image >= threshold


    # ######get the largest blob, this takes 0.18s
    cvVersion = int(cv2.__version__.split('.')[0])

    Mask0 = np.uint8(Mask0)
    if cvVersion == 2:
        contours, hierarchy = cv2.findContours(Mask0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(Mask0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    Mask = np.zeros(Image.shape[:2], dtype=np.uint8)
    cv2.drawContours(Mask, contours, max_index, 1, -1)

    ResultImg = Image.copy()
    if len(Image.shape) == 3:
        ResultImg[Mask ==0] = (255,255,255)
    else:
        ResultImg[Mask==0] = 255

    return ResultImg, Mask

