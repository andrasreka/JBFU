import numpy as np
import cv2
from joblib import Parallel, delayed

def distance(x, y, i, j): return np.sqrt((x-i)**2 + (y-j)**2)

def gaussian(x, sigma): return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))

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

def weighted_median_bilateral_filter(f, g, x, y, kernel_size, sigma_spatial, sigma_range, weights):
    #f usid for weighting
    #g is filtered

    min_sum = np.inf
    min_y = None
    hl = int(kernel_size/2)

    medians = set(g[x:x+2*hl, y:y+2*hl].flatten())
    for y_intensity in medians:
        i_filtered = 0
        for i in range(kernel_size - 1):
            for j in range(kernel_size - 1):
                
                window_x = int(x - hl + i)
                window_y = int(y - hl + j)

                mean_i = int(f[window_x,window_y]) - int(f[x][y])
                mean_s = distance(window_x, window_y, x, y)

                gi = gaussian(mean_i, sigma_range)
                gs = gaussian(mean_s, sigma_spatial)
                i_filtered += weights[i,j] * abs(y_intensity - int(g[window_x,window_y])) * gi * gs
        
        if i_filtered<min_sum:
            min_sum = i_filtered
            min_y = y_intensity

    return min_y
            

def jbf(f, g, kernel_size, sigma_i, sigma_s):
    #f usid for weighting
    #g is filtered

    filtered_image = np.zeros(g.shape)
    filtered_image =  Parallel(n_jobs=12)(delayed(bilateral_filter)(f, g, i, j, kernel_size, sigma_i, sigma_s) for i in range (len(g)) for j in range (len(g[0])))
    return np.array(filtered_image, dtype = np.uint8).reshape(g.shape)


def jbmf(f, g, kernel_size, sigma_i, sigma_s, w):
    #f usid for weighting
    #g is filtered

    hl = int(kernel_size/2)
    height,width = g.shape

    filtered_image = np.zeros(g.shape)
    filtered_image =  Parallel(n_jobs=12)(delayed(weighted_median_bilateral_filter) (
        cv2.copyMakeBorder(f,hl,hl,hl,hl,cv2.BORDER_REPLICATE), 
        cv2.copyMakeBorder(g,hl,hl,hl,hl,cv2.BORDER_REPLICATE),
        i, 
        j, 
        kernel_size, 
        sigma_i, 
        sigma_s, 
        w) for i in range(width) for j in range(height))
 
    #nonparallel version for debugging
    # filtered_image = [(weighted_median_bilateral_filter)(
    #     cv2.copyMakeBorder(f,hl,hl,hl,hl,cv2.BORDER_REPLICATE), 
    #     cv2.copyMakeBorder(g,hl,hl,hl,hl,cv2.BORDER_REPLICATE),
    #     i, 
    #     j, 
    #     kernel_size, 
    #     sigma_i, 
    #     sigma_s, 
    #     w) for i in range(width) for j in range(height)]

    return np.array(filtered_image, dtype = np.uint8).reshape(g.shape)


def upsampling_iterative(rgb, depth, kernel, s1, s2):

    print("Upsample iterative running: {}, {}, {}".format(kernel, s1, s2))
    
    rgb_ori = rgb.copy()
    #upsampling factor
    uf = int(np.log2(rgb.shape[1]/depth.shape[1])) 

    for i in range(uf):

        print("{} iter from {}".format(i, uf))
        
        dim = (depth.shape[1] * 2, depth.shape[0] * 2)        
        rgb = cv2.resize(rgb, dim) 
        depth = cv2.resize(depth, dim) 

        depth = jbf(rgb, depth, kernel, s1, s2)

    
    depth = cv2.resize(depth, (rgb_ori.shape[1], rgb_ori.shape[0]))
    return jbf(rgb_ori, depth, kernel, s1, s2)


def upsampling(rgb, depth, kernel, s1, s2):

    print("Upsample running: {}, {}, {}".format(kernel, s1, s2))
    return jbf(rgb, cv2.resize(depth, (rgb.shape[1], rgb.shape[0])), kernel, s1, s2)

