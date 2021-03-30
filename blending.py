import cv2
import numpy as np
import argparse
from pynput.keyboard import Key, Controller, Listener

ref_point = []
crop = False

def gaussian_pyramid(img, num_levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr

def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1
    
    laplacian_pyr = [laplacian_top]
    for i in range(num_levels,0,-1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr
 
def blend(laplacian_A,laplacian_B,mask_pyr):
    LS = []
    for la,lb,mask in zip(laplacian_A,laplacian_B,mask_pyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS

def reconstruct(laplacian_pyr):

    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i+1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst

def shape_selection(event, x, y, flags, param):
	global ref_point, crop

	if event == cv2.EVENT_LBUTTONDOWN:
		ref_point = [(x, y)]

	elif event == cv2.EVENT_LBUTTONUP:
		ref_point.append((x, y))

		cv2.rectangle(img2, ref_point[0], ref_point[1], (0, 255, 0), 2)
		cv2.imshow("img2", img2)


# Construct ArgumentParser
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--background",type=str, required=True,help="path to background img2")
ap.add_argument("-f", "--foreground", type=str, required=True,help="path to foreground img2")
args = vars(ap.parse_args())

# Resizing images to same the size 
img1 = cv2.imread((args["background"]))
img1 = cv2.resize(img1, (2048, 1024))
img2 = cv2.imread((args["foreground"]))
img2 = cv2.resize(img2, (2048, 1024))

# Create the mask
clone = img2.copy()
cv2.namedWindow("img2")
cv2.setMouseCallback("img2", shape_selection)
while True:
	# display the img2 and wait for a keypress
	cv2.imshow("img2", img2)
	key = cv2.waitKey(1) & 0xFF

	# press 'r' to reset the window
	if key == ord("r"):
		img2 = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

# Creating mask image by user selected coordinates
if len(ref_point) == 2:
    mask = np.zeros((1000,1800,3), dtype='float32')
    mask[ref_point[0][1]+3:ref_point[1][1]-3, ref_point[0][0]+3:ref_point[1][0]-3,:] = (1,1,1)

# Defining number of levels of the laplacian pyramid
num_levels = 7

# Genrating the pyramids
gaussian_pyr_1 = gaussian_pyramid(img1, num_levels)
laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
gaussian_pyr_2 = gaussian_pyramid(img2, num_levels)
laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)

mask_pyr_final = gaussian_pyramid(mask, num_levels)
mask_pyr_final.reverse()

# Merging the pyramids
add_laplace = blend(laplacian_pyr_1,laplacian_pyr_2,mask_pyr_final)
final  = reconstruct(add_laplace)

# Save and displaying the final image
cv2.imwrite('final.jpg',final[num_levels])
cv2.imshow('Result', cv2.imread('final.jpg'))
cv2.waitKey(0)