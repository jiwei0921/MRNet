import cv2

def BGR2RGB(Image):
    #the input image is BGR image from OpenCV
    RGBImage = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
    return RGBImage


def RGB2BGR(Image):
    #the input image is RGB image
    BGRImage = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
    return BGRImage