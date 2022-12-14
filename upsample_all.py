import cv2
from bilateral import upsampling_iterative, upsampling
import numpy as np
if __name__ == "__main__":

    kernel = 7
    sigma_spatial = 4
    sigma_range = 7
    weights = np.ones((kernel, kernel))
    images = ['Art', 'Books', 'Dolls', 'Laundry', 'Moebius', 'Reindeer']
    base = "data/scenes_2005/"
    for image in images:
        print("Processing {}".format(image))

        rgb = cv2.imread(base + image + "/view1.png", 0)
        d = cv2.imread(base + image + "/disp1.png", 0)

        jb_sf = upsampling(rgb, d, kernel, sigma_spatial, sigma_range)
        cv2.imwrite(base + image + "/jb_sf.png",jb_sf)

        jb_it = upsampling_iterative(rgb, d, kernel, sigma_spatial, sigma_range)
        cv2.imwrite(base + image + "/jb_it.png",jb_it)

        jbm_sf = upsampling(rgb, d, kernel, sigma_spatial, sigma_range, weights)
        cv2.imwrite(base + image + "/jbm_sf.png",jbm_sf)

        jbm_it = upsampling_iterative(rgb, d, kernel, sigma_spatial, sigma_range, weights)
        cv2.imwrite(base + image + "/jbm_it.png",jbm_it)





