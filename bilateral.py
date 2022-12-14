import numpy as np
import cv2
import sys
from joblib import Parallel, delayed
from datetime import datetime


def distance(x, y, i, j): return np.sqrt((x-i)**2 + (y-j)**2)

def gaussian(x, sigma): return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))


def weighted_median(vals, w):
    
    pairs = [[vals[index], w[index]] for index in range(len(vals))]
    pairs.sort(key = lambda p: p[0])
    sums = 0

    for element, weight in pairs:
        sums += weight
        if sums > 0.5: return element


def bilateral_filter(f, g, x, y, kernel, sigma_spatial, sigma_range):
    
    #f usid for weighting
    #g is filtered
    
    hl = int(kernel/2)
    
    f = cv2.copyMakeBorder(f,hl,hl,hl,hl,cv2.BORDER_REPLICATE)
    g = cv2.copyMakeBorder(g,hl,hl,hl,hl,cv2.BORDER_REPLICATE)


    i_filtered = 0
    Wp = 0
    for i in range(kernel - 1):
        for j in range(kernel - 1):
            
            window_x = x - hl + i
            window_y = y - hl + j

            mean_i = int(f[int(window_x)][int(window_y)]) - int(f[x][y])
            mean_s = distance(window_x, window_y, x, y)

            prod = gaussian(mean_i, sigma_range) * gaussian(mean_s, sigma_spatial)
            i_filtered += g[int(window_x)][int(window_y)] * prod
            Wp += prod

    i_filtered = i_filtered / Wp
    return int(round(i_filtered))

def weighted_median_bilateral_filter(f, g, x, y, kernel, sigma_spatial, sigma_range, weights):
    
    #f usid for weighting
    #g is filtered
    
    hl = int(kernel/2)
    
    f = cv2.copyMakeBorder(f,hl,hl,hl,hl,cv2.BORDER_REPLICATE)
    g = cv2.copyMakeBorder(g,hl,hl,hl,hl,cv2.BORDER_REPLICATE)

    min_sum = np.inf
    min_y = None

    print(x,y)
    for y in range (256):
        i_filtered = 0
        for i in range(kernel - 1):
            for j in range(kernel - 1):
                
                window_x = x - hl + i
                window_y = y - hl + j

                mean_i = int(f[int(window_x)][int(window_y)]) - int(f[x][y])
                mean_s = distance(window_x, window_y, x, y)


                gi = gaussian(mean_i, sigma_range)
                gs = gaussian(mean_s, sigma_spatial)
                i_filtered += 1 * abs(y - g[int(window_x)][int(window_y)]) * gi * gs
        if i_filtered<min_sum:
            min_sum = i_filtered
            min_y = y

    return min_y
             


def jbf(f, g, filter_diameter, sigma_i, sigma_s):

    #f usid for weighting
    #g is filtered
    
    filtered_image = np.zeros(g.shape)
    filtered_image =  Parallel(n_jobs=12)(delayed(bilateral_filter)(f, g, i, j, filter_diameter, sigma_i, sigma_s) for i in range (len(g)) for j in range (len(g[0])))
    return np.array(filtered_image, dtype = np.uint8).reshape(g.shape)

def jbmf(f, g, filter_diameter, sigma_i, sigma_s, w):

    #f usid for weighting
    #g is filtered
    
    filtered_image = np.zeros(g.shape)
    filtered_image =  Parallel(n_jobs=32)(delayed(weighted_median_bilateral_filter)(f, g, i, j, filter_diameter, sigma_i, sigma_s, w) for i in range (len(g)) for j in range (len(g[0])))
    return np.array(filtered_image, dtype = np.uint8).reshape(g.shape)


def upsampling(rgb, depth, kernel, s1, s2):

    print("Upsample running: {}, {}, {}".format(kernel, s1, s2))
    
    rgb_ori = rgb.copy()
    #upsampling fector based on rows
    uf = int(np.log2(rgb.shape[1]/depth.shape[1])) 

    print(uf,rgb.shape[1],depth.shape[1])

    for i in range(uf):

        print("{} iter from {}".format(i, uf))
        
        #D *= 2 (size)
        width = int(depth.shape[1] * 2)
        height = int(depth.shape[0] * 2)
        dim = (width, height)        
        
        rgb = cv2.resize(rgb, dim) 
        depth = cv2.resize(depth, dim) 

        depth = jbf(rgb, depth, kernel, s1, s2)

    
    depth = cv2.resize(depth, (rgb_ori.shape[1], rgb_ori.shape[0]))
    return jbf(rgb_ori, depth, kernel, s1, s2)


# if __name__ == "__main__":

#     SIGMA_SPATIAL = 4
#     SIGNA_RANGE = 4
#     KERNEL = 7

#     rgb = cv2.imread("im2.ppm", 0)
#     depth = cv2.imread("disp2.png", 0)
#     print(rgb.shape)

#     # scale_percent = 30
#     # width = int(rgb.shape[1] * scale_percent / 100)
#     # height = int(rgb.shape[0] * scale_percent / 100)
#     # dim = (width, height)
#     # rgb = cv2.resize(rgb, dim)
#     # print(rgb.shape)
    
#     start=datetime.now()
#     upsampled = upsampling(rgb, depth, KERNEL, SIGMA_SPATIAL, SIGNA_RANGE)
#     print(datetime.now()-start)

#     cv2.imwrite("upsampled.png",upsampled)


if __name__ == "__main__":
    

    f = cv2.imread("lena.png", 0)
    #g = cv2.imread("testa.jpg", 0)


    # #only for speedy testing
    # if input_img.shape[0] > 512:
    #     scale_percent = 50
    #     width = int(input_img.shape[1] * scale_percent / 100)
    #     height = int(input_img.shape[0] * scale_percent / 100)
    #     dim = (width, height)
    #     src = cv2.resize(input_img, dim, interpolation = cv2.INTER_AREA)
    # else:
    #     src = input_img
    
    cv2.imwrite("ori.png",f)

    filtered_image_OpenCV = cv2.bilateralFilter(f, 7, 12.0, 16.0)
    # cv2.imshow("filtered_image_OpenCV.png", filtered_image_OpenCV)
    cv2.imwrite("filtered_image_OpenCV.png",filtered_image_OpenCV)

    start=datetime.now()
    filtered_image_own = jbmf(f, f, 7, 12.0, 16.0, None)
    print(datetime.now()-start)

    cv2.imshow("filtered_image_own.png", filtered_image_own)
    cv2.imwrite("jbf.png",filtered_image_own)

    cv2.waitKey(0)
    cv2.destroyAllWindows()







