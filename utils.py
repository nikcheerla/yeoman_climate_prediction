
#Misc utils, including nuclei detection and image processing

import numpy as np

from scipy import ndimage as nd
import scipy.misc
import skimage as ski
import skimage.io as skio
from skimage import filters
from skimage.exposure import rescale_intensity
from skimage import morphology
import scipy.ndimage.morphology as smorphology
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy.signal import convolve
from scipy.linalg import norm

from skimage.morphology import binary_dilation, binary_erosion

import matplotlib.pyplot as plt
import glob, sys, os, traceback, contextlib, resource
import IPython

from guppy import hpy


sys.setrecursionlimit(20000)


MIN_DIST = 20.0
MIN_SIZE = 50





class Suppressor(object):

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout

    def write(self, x): pass
    def flush(self, *args, **kwargs): pass









def nuclei_detect_pipeline(img, MinPixel = 200, MaxPixel=2500):
    return nuclei_detection(img, MinPixel, MaxPixel)


def nuclei_detection(img, MinPixel, MaxPixel):
    img_f = ski.img_as_float(img)
    adjustRed = rescale_intensity(img_f[:,:,0])
    roiGamma = rescale_intensity(adjustRed, in_range=(0, 0.5));
    roiMaskThresh = roiGamma < (250 / 255.0) ;

    roiMaskFill = morphology.remove_small_objects(~roiMaskThresh, MinPixel);
    roiMaskNoiseRem = morphology.remove_small_objects(~roiMaskFill,150);
    roiMaskDilat = morphology.dilation(roiMaskNoiseRem, morphology.disk(3));
    roiMask = smorphology.binary_fill_holes(roiMaskDilat)

    hsv = ski.color.rgb2hsv(img);
    hsv[:,:,2] = 0.8;
    img2 = ski.color.hsv2rgb(hsv)
    diffRGB = img2-img_f
    adjRGB = np.zeros(diffRGB.shape)
    adjRGB[:,:,0] = rescale_intensity(diffRGB[:,:,0],in_range=(0, 0.4))
    adjRGB[:,:,1] = rescale_intensity(diffRGB[:,:,1],in_range=(0, 0.4))
    adjRGB[:,:,2] = rescale_intensity(diffRGB[:,:,2],in_range=(0, 0.4))

    gauss = gaussian_filter(adjRGB[:,:,2], sigma=3, truncate=5.0);

    bw1 = gauss>(100/255.0);
    bw1 = bw1 * roiMask;
    bw1_bwareaopen = morphology.remove_small_objects(bw1, MinPixel)
    bw2 = smorphology.binary_fill_holes(bw1_bwareaopen);

    bwDist = nd.distance_transform_edt(bw2);
    filtDist = gaussian_filter(bwDist,sigma=5, truncate=5.0);

    L = label(bw2)
    R = regionprops(L)
    coutn = 0
    for idx, R_i in enumerate(R):
        if R_i.area < MaxPixel and R_i.area > MinPixel:
            r, l = R_i.centroid
            #print(idx, filtDist[r,l])
        else:
            L[L==(idx+1)] = 0
    BW = L > 0
    return BW



def bound(tuple, low=0, high=2000):
    xs, ys = tuple
    return min(max(xs, low), high), min(max(ys, low), high)




#floodfill on image
def floodfill(bin_map, x, y, visited, grid_markoff):
    if len(visited) > 1000:
        return
    if x >= grid_markoff.shape[0] or y >= grid_markoff.shape[1] or x < 0 or y < 0:
        return
    if grid_markoff[x, y]:
        return
    if not bin_map[x, y]:
        return

    grid_markoff[x, y] = True
    visited.append((x, y))

    floodfill(bin_map, x+1, y, visited, grid_markoff)
    floodfill(bin_map, x, y+1, visited, grid_markoff)
    floodfill(bin_map, x-1, y, visited, grid_markoff)
    floodfill(bin_map, x, y-1, visited, grid_markoff)


#gets nuclei number, avg nuclei statistics
def centroid_coords(heatmap):

    #IPython.embed()
    grid_markoff = np.zeros(shape=heatmap.shape)
    
    coords = []
    num_nuclei = 0
    avg_nuclei = 0
    for x in range(0, heatmap.shape[0]):
        for y in range(0, heatmap.shape[1]):
            visited = []
            floodfill(heatmap, x, y, visited, grid_markoff)
            if len(visited) > 0:
                #print ("Found nuclei!", x, y)
                num_nuclei += 1
                avg_nuclei += len(visited)
                centroid_nuclei = np.array(visited).mean(axis=0)
                
                if len(visited) < MIN_SIZE:
                    continue

                #print (len(visited), centroid_nuclei)
                coords.append((centroid_nuclei[0], centroid_nuclei[1]))

    return coords

def f1_score(heatmaps, true_maps):

    tp, num_pred, num_true = 0, 0, 0
    for i in range(0, len(heatmaps)):
        heatmap, true_map = heatmaps[i], true_maps[i]

        centroids_pred = centroid_coords(heatmap)
        centroids_true = centroid_coords(true_map)
        num_pred += len(centroids_pred)
        num_true += len(centroids_true)

        #print (num_pred)
        #print (num_true)

        
        for x2, y2 in centroids_true:
            for x1, y1 in centroids_pred:
                #print ((x1, y1), (x2, y2), int(norm(((x1 - y1), (x2 - y2)))))
                if (norm(((x1 - x2), (y1 - y2))) < MIN_DIST):
                    tp +=1
                    break

        #print (tp)

    print (tp, num_pred, num_true)
    num_pred = max(num_pred, 1)
    num_true = max(num_true, 1)
    precision = tp*1.0/num_pred
    recall = tp*1.0/num_true

    return (2*precision*recall)/(precision + recall)


def best_f1_score(pred_heatmaps, true_maps):

    best = (None, None)
    for thresh in np.linspace(0.3, 0.99, 10):
        score = f1_score(pred_heatmaps > thresh, true_maps)
        print ("Threshold: ", thresh, "F1-Score: ", score)
        if best[1] > score:
            best = (thresh, score)

    for thresh in np.linspace(best[0] -0.05, best[0]+0.1, 10):
        score = f1_score(pred_heatmaps > thresh, true_maps)
        if best[1] > score:
            best = (thresh, score)

    return best






def evaluate_model_on_directory(fcn_model, directory, rotations=[0, 1], window_size=[800], prefix="_image.jpg",
            overlap=0, suffix="_inter.jpg", visualize_thresh=True):

    for img_file in sorted(glob.glob(directory + "/*" + prefix)):
        pred_file = img_file[:-(len(prefix))] + suffix

        print (pred_file)

        if prefix[-4:] == '.npy':
            img = np.load(img_file)
        else:
            img = scipy.misc.imread(img_file)/255.0

        preds = []
        for k in rotations:
            for wsize in window_size:
                img2 = np.rot90(img, k=k)
                preds_tmp = fcn_model.evaluate_tiled(img2, window_size=wsize, overlap=overlap)
                #preds_tmp = fcn_model.evaluate(img2)
                preds_tmp = np.rot90(preds_tmp, k=-k)
                preds.append(preds_tmp)
        preds = np.mean(np.array(preds), axis=0)
        preds[0, 0] = 1

        
        radius = 3
        kernel = np.zeros((2*radius+1, 2*radius+1))
        y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        kernel[mask] = 1
        preds = convolve(preds, kernel, mode='same')
        preds = convolve(preds, kernel, mode='same')
        preds = convolve(preds, kernel, mode='same')
        preds = convolve(preds, kernel, mode='same')
        
        scipy.misc.imsave(pred_file, preds)

        if visualize_thresh:
            preds_thresh = preds < filters.threshold_otsu(preds)
            thresh_file = img_file[:-10] + "_thresh.jpg"
            scipy.misc.imsave(thresh_file, preds_thresh)









if __name__ == "__main__":
    from models import FCNModel
    import sys
    
    if sys.argv[1] == 'evaluate':

        model = FCNModel.load("results/checkpoint.h5")
        evaluate_model_on_directory(model, sys.argv[2], suffix="_inter.jpg")

    elif sys.argv[1] == 'score':
        pred_heatmaps, true_maps = [], []
        for img_file in sorted(glob.glob(sys.argv[2] + "/*_image.jpg")):
            
            pred_file = img_file[:-10] + "_inter.jpg"
            hmap_file = img_file[:-10] + "_heatmap.jpg"

            if not os.path.exists(pred_file):
                continue
            
            pred_heatmaps.append(scipy.misc.imread(pred_file)/255.0)
            true_maps.append(scipy.misc.imread(hmap_file)/255.0)

        pred_heatmaps, true_maps = np.array(pred_heatmaps), np.array(true_maps)

        thresh, f1score = best_f1_score(pred_heatmaps, true_maps)
        print ("Best F1 score: ", f1score, "at threshold ", thresh)
