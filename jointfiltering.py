import cv2
from datetime import datetime
import numpy as np
from bilateral import jbmf, jbf

if __name__ == "__main__":
    
    f = cv2.imread("../data/flash_no_flash/potsWB_00_flash.jpg")
    g = cv2.imread("../data/flash_no_flash/potsWB_01_noflash.jpg")

    filtered_image_own = np.zeros(g.shape)

    start=datetime.now()

    
    for i in range(3):
        print("Channel:", i)
        #filtered_image_own[:,:,i] = jbf(f[:,:,i], g[:,:,i], 7, 5.0, 5.0)
        filtered_image_own[:,:,i] = jbmf(f[:,:,i], g[:,:,i], 7, 5.0, 5.0, np.ones((7,7)))

    print(datetime.now()-start)

    cv2.imwrite("flash_no_flash_fusion_jbmf.png",filtered_image_own)





