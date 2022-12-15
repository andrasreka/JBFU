import numpy as np
import cv2
from joblib import Parallel, delayed

def distance(x, y, i, j): return np.sqrt((x-i)**2 + (y-j)**2)

def gaussian(x, sigma): return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))

def bilateral_filter(f, g, x, y, kernel_size, sigma_spatial, sigma_range):    
    #f used for weighting
    #g is filtered
    
    hl = int(kernel_size/2)
    center_x = x + hl
    center_y = y + hl
    
    i_filtered = 0
    Wp = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            
            window_x = x + i
            window_y = y + j

            mean_i = int(f[window_x,window_y]) - int(f[center_x][center_y])
            mean_s = distance(window_x, window_y, center_x, center_y)

            prod = gaussian(mean_i, sigma_range) * gaussian(mean_s, sigma_spatial)
            i_filtered += g[window_x][window_y] * prod
            Wp += prod

    i_filtered = i_filtered / Wp
    return int(round(i_filtered))

def weighted_median_bilateral_filter(f, g, x, y, kernel_size, sigma_spatial, sigma_range, weights):
    #f used for weighting
    #g is filtered
    #x,y the kernel right top corner

    hl = int(kernel_size/2)
    center_x = x + hl
    center_y = y + hl
    medians = list(set(g[x:x+kernel_size, y:y+kernel_size].flatten()))
    bilateral_filter = np.zeros((kernel_size,kernel_size))

    for i in range(kernel_size):
            for j in range(kernel_size):
                
                window_x = x + i
                window_y = y + j

                #print(window_x, window_y, x, y)
                mean_i = int(f[window_x,window_y]) - int(f[center_x][center_y])
                mean_s = distance(window_x, window_y, center_x, center_y)

                gi = gaussian(mean_i, sigma_range)
                gs = gaussian(mean_s, sigma_spatial)
         
                bilateral_filter[i,j] = gi*gs

    costs = [np.sum(weights * abs(intensity - g[x:x+kernel_size, y:y+kernel_size]) * bilateral_filter)  for intensity in medians]
    return medians[np.array(costs).argmin()]
            

def jbf(f, g, kernel_size, sigma_i, sigma_s):
    #f used for weighting
    #g is filtered

    hl = int(kernel_size/2)
    height,width = g.shape

    # filtered_image =  Parallel(n_jobs=-1)(delayed(bilateral_filter)(
    # cv2.copyMakeBorder(f,hl,hl,hl,hl,cv2.BORDER_REPLICATE),
    # cv2.copyMakeBorder(g,hl,hl,hl,hl,cv2.BORDER_REPLICATE), 
    # i, 
    # j, 
    # kernel_size, 
    # sigma_i, 
    # sigma_s) for i in range(height) for j in range(width))

  

    filtered_image =  [(bilateral_filter)(
    cv2.copyMakeBorder(f,hl,hl,hl,hl,cv2.BORDER_REPLICATE),
    cv2.copyMakeBorder(g,hl,hl,hl,hl,cv2.BORDER_REPLICATE), 
    i, 
    j, 
    kernel_size, 
    sigma_i, 
    sigma_s) for i in range(height) for j in range(width)]
    
    return np.array(filtered_image, dtype = np.uint8).reshape(g.shape)

def jbmf(f, g, kernel_size, sigma_i, sigma_s, w):
    #f used for weighting
    #g is filtered

    hl = int(kernel_size/2)
    height,width = g.shape

    # filtered_image =  Parallel(n_jobs=-1)(delayed(weighted_median_bilateral_filter) (
    #     cv2.copyMakeBorder(f,hl,hl,hl,hl,cv2.BORDER_REPLICATE), 
    #     cv2.copyMakeBorder(g,hl,hl,hl,hl,cv2.BORDER_REPLICATE),
    #     i, 
    #     j, 
    #     kernel_size, 
    #     sigma_i, 
    #     sigma_s, 
    #     w) for i in range(height) for j in range(width))
 
  
    #nonparallel version for debugging
    filtered_image = [(weighted_median_bilateral_filter)(
        cv2.copyMakeBorder(f,hl,hl,hl,hl,cv2.BORDER_REPLICATE), 
        cv2.copyMakeBorder(g,hl,hl,hl,hl,cv2.BORDER_REPLICATE),
        i, 
        j, 
        kernel_size, 
        sigma_i, 
        sigma_s, 
        w) for i in range(height) for j in range(width)]

    return np.array(filtered_image, dtype = np.uint8).reshape(g.shape)


def upsampling_iterative(rgb, depth, kernel, s1, s2, weights=None):

    print("Upsample iterative running: {}, {}, {}".format(kernel, s1, s2))
    
    rgb_ori = rgb.copy()
    uf = int(np.log2(rgb.shape[1]/depth.shape[1])) 

    for i in range(uf):

        print("{} iter from {}".format(i, uf))
        
        dim = (depth.shape[1] * 2, depth.shape[0] * 2)        
        rgb = cv2.resize(rgb, dim) 
        depth = cv2.resize(depth, dim) 

        if weights is None:
            depth = jbf(rgb, depth, kernel, s1, s2)
        else:
            depth = jbmf(rgb, depth, kernel, s1, s2, weights)
    
    depth = cv2.resize(depth, (rgb_ori.shape[1], rgb_ori.shape[0]))

    if weights is None:
        return jbf(rgb, depth, kernel, s1, s2)
    
    return jbmf(rgb, depth, kernel, s1, s2, weights)


def upsampling(rgb, depth, kernel, s1, s2, weights=None):

    print("Upsample running: {}, {}, {}".format(kernel, s1, s2))

    if weights is None:
        return jbf(rgb, cv2.resize(depth, (rgb.shape[1], rgb.shape[0])), kernel, s1, s2)

    return jbmf(rgb, cv2.resize(depth, (rgb.shape[1], rgb.shape[0])), kernel, s1, s2, weights)

