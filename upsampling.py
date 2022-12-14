import cv2
from datetime import datetime
from bilateral import upsampling, upsampling_iterative

TEST_MODE = True

if __name__ == "__main__":

    SIGMA_SPATIAL = 4
    SIGNA_RANGE = 4
    KERNEL = 7

    rgb = cv2.imread("im2.ppm", 0)
    depth = cv2.imread("disp2.png", 0)
    print(rgb.shape)

    if TEST_MODE:
        scale_percent = 30
        width = int(rgb.shape[1] * scale_percent / 100)
        height = int(rgb.shape[0] * scale_percent / 100)
        dim = (width, height)
        rgb = cv2.resize(rgb, dim)
        print(rgb.shape)
    
    start=datetime.now()
    upsampled = upsampling(rgb, depth, KERNEL, SIGMA_SPATIAL, SIGNA_RANGE)
    print(datetime.now()-start)

    cv2.imwrite("upsampled_sf.png",upsampled)

    start=datetime.now()
    upsampled = upsampling_iterative(rgb, depth, KERNEL, SIGMA_SPATIAL, SIGNA_RANGE)
    print(datetime.now()-start)

    cv2.imwrite("upsampled_it.png",upsampled)

