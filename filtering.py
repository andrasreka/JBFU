import cv2
from datetime import datetime
import numpy as np
from bilateral import jbmf

if __name__ == "__main__":
    
    input_img = cv2.imread("../data/flash_no_flash/potsWB_00_flash.jpg", 0)
    #g = cv2.imread("testa.jpg", 0)

    #only for speedy testing
    if input_img.shape[0] > 300:
        scale_percent = 10
        width = int(input_img.shape[1] * scale_percent / 100)
        height = int(input_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        src = cv2.resize(input_img, dim, interpolation = cv2.INTER_AREA)
    else:
        src = input_img
    
    print(src.shape)
    cv2.imwrite("original.png",src)

    filtered_image_OpenCV = cv2.bilateralFilter(src, 7, 12.0, 16.0)
    cv2.imwrite("filtered_OpenCV.png",filtered_image_OpenCV)

    start=datetime.now()
    filtered_image_own = jbmf(src, src, 7, 12.0, 16.0, np.ones((7,7)))
    print(datetime.now()-start)

    cv2.imshow("filtered_my.png", filtered_image_own)
    cv2.imwrite("jbf.png",filtered_image_own)

    cv2.waitKey(0)
    cv2.destroyAllWindows()







