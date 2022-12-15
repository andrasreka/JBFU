import open3d as o3d
import numpy as np
from PIL import Image

def replace_zeroes(data):
  min_nonzero = np.min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data

def RGBD_to_PC(rgb, d, focal_length, baseline, scale = 9.88):

    disparity_matrix = (focal_length/scale) * (baseline/scale) / d#replace_zeroes(d)

    img = o3d.geometry.Image(rgb.astype('uint8'))
    depth = o3d.geometry.Image(disparity_matrix.astype(np.uint16))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)
    o3d_pinhole = o3d.camera.PinholeCameraIntrinsic()

    o3d_pinhole.set_intrinsics(
        rgb.shape[1], rgb.shape[0], focal_length, focal_length, 0.5 * rgb.shape[0], 0.5 * rgb.shape[1]
    )

    pcd_from_depth_map = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_pinhole)

    pcd_from_depth_map.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd_from_depth_map])


if __name__ == '__main__':

    rgb_source = 'cones/conesF/im2.ppm'
    d_source = 'cones/conesF/disp2.pgm' #JBFU/upsampled.png'
    rgb =  np.asarray(Image.open(rgb_source).convert('L'))
    disparity = np.asarray(Image.open(d_source).convert('L'))
    RGBD_to_PC(rgb, disparity, 3740, 160, scale = 9.88)


