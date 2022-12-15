import cv2
from bilateral import upsampling_iterative, upsampling
import numpy as np
from datetime import datetime,timedelta
if __name__ == "__main__":

    kernel = 7
    sigma_spatial = 4
    sigma_range = 7
    weights = np.ones((kernel, kernel))
    images = ['Art']#['Books', 'Dolls', 'Laundry', 'Moebius', 'Reindeer']
    sums =  [timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0),timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0),timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0),timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)]
    base = "data/scenes_2005/"
    for image in images:
        print("Processing {}".format(image))

        rgb = cv2.imread(base + image + "/view1_large.png", 0)
        d = cv2.imread(base + image + "/disp1.png", 0)

        start=datetime.now()
        jb_sf = upsampling(rgb, d, kernel, sigma_spatial, sigma_range)
        sums[0] += datetime.now()-start
        cv2.imwrite(base + image + "/jb_sf_large.png",jb_sf)

        start=datetime.now()
        jb_it = upsampling_iterative(rgb, d, kernel, sigma_spatial, sigma_range)
        sums[1] += datetime.now()-start
        cv2.imwrite(base + image + "/jb_it_large.png",jb_it)
        
        start=datetime.now()
        jbm_sf = upsampling(rgb, d, kernel, sigma_spatial, sigma_range, weights)
        sums[2] += datetime.now()-start
        cv2.imwrite(base + image + "/jbm_sf_large.png",jbm_sf)
        
        start=datetime.now()
        jbm_it = upsampling_iterative(rgb, d, kernel, sigma_spatial, sigma_range, weights)
        sums[3] += datetime.now()-start
        cv2.imwrite(base + image + "/jbm_it_large.png",jbm_it)

    secs = [s.seconds for s in sums]
    print(secs) 



