import cv2
from datetime import datetime
import numpy as np
from bilateral import jbmf, jbf

if __name__ == "__main__":
    
    f = cv2.imread("../data/flash_no_flash/potsWB_00_flash.jpg")
    g = cv2.imread("../data/flash_no_flash/potsWB_01_noflash.jpg")

    filtered_image_own = np.zeros(g.shape)

    start=datetime.now()
    weights = np.ones((5,5))

    weights[1:4,1:4] = 2
    print(weights)

    for i in range(3):
        print("Channel:", i)
        #filtered_image_own[:,:,i] = jbf(f[:,:,i], g[:,:,i], 7, 5.0, 5.0)
        filtered_image_own[:,:,i] = jbmf(f[:,:,i], g[:,:,i], 5, 6.0, 6.0, weights)

    print(datetime.now()-start)

    cv2.imwrite("flash_no_flash_fusion_jbmf_5_6_6.png",filtered_image_own)





