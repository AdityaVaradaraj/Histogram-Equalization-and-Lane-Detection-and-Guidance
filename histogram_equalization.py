#!/usr/env python
import numpy as np
import cv2
import glob

# Histogram Equalization
def histogram_equalization(y):
    N_pix_hist = y.shape[0]*y.shape[1]
    h = np.zeros((256,1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            h[y[i][j]] += 1
    cfd = np.zeros((256,1))
    for k in range(256):
        cfd[k] = np.sum(h[0:k+1])/N_pix_hist
    eq = np.zeros(y.shape, dtype=np.uint8)
    for l in range(y.shape[0]):
        for m in range(y.shape[1]):
            eq[l][m] = cfd[y[l][m]]*255

    return eq.astype(np.uint8)

# Adaptive Histogram Equalization
def AHE(y):
    # Dividing image into 8 x 8 = 64 tiles
    tile_height = int(y.shape[0]/8)
    tile_width = int(y.shape[1]/8)
    adapt_y = y.copy()
    for i in range(0, y.shape[0], tile_height):
        for j in range(0, y.shape[1], tile_width):
            if i+tile_height <= y.shape[0] and j+tile_width <= y.shape[1]:
                adapt_y[i:i+tile_height, j:j+tile_width] = histogram_equalization(y[i:i+tile_height,j:j+tile_width])
    return adapt_y

if __name__ == '__main__':
    cv_img = []
    for img in sorted(glob.glob("./adaptive_hist_data/*.png")):
        n = cv2.imread(img)
        cv_img.append(n)
    frameSize = (cv_img[0][:,:,0].shape[1], cv_img[0][:,:,0].shape[0])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_eq = cv2.VideoWriter('Histogram_Equalization.mp4',fourcc, 2, frameSize)
    out_adapt = cv2.VideoWriter('Adaptive_Histogram_Equalization.mp4',fourcc, 2, frameSize)
    for img in cv_img:
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_ch = ycrcb_img[:, :, 0].copy()
        
        # Standard Histogram Equalization
        ycrcb_img[:,:,0] = histogram_equalization(y_ch)
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        
        # Adaptive Histogram Equalization
        new_ycrcb = ycrcb_img.copy()
        new_ycrcb[:,:,0] = AHE(y_ch)
        adapt_eq_img = cv2.cvtColor(new_ycrcb, cv2.COLOR_YCrCb2BGR)

        out_eq.write((equalized_img).astype(np.uint8))
        out_adapt.write((adapt_eq_img).astype(np.uint8))
    out_eq.release()
    out_adapt.release()