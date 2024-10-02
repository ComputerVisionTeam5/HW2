import matplotlib.pyplot as plt
import cv2 as cv
import time
from IPython.display import Image
import numpy as np
import random
from sklearn.cluster import KMeans
import skfuzzy as fuzz


def plot_img(n, figsize,titles,imgs, n_row=1):
    """
    Plots multiple images in a single row with specified titles.

    Parameters:
    - n (int): Number of images to plot.
    - figsize (tuple): Size of the figure (width, height).
    - titles (list of str): List of titles for the images.
    - imgs (list of numpy arrays): List of images to be plotted. Images should be in BGR format.
    """
    x, y = figsize
    fig, axes = plt.subplots(n_row, n // n_row, figsize=(x, y))
    axes = axes.ravel()
    for i in range(n):
        axes[i].imshow(cv.cvtColor(imgs[i], cv.COLOR_BGR2RGB))
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def euclidean_distance(point1, point2):
    """
    Calculates the euclidean distance given two points

    Parameters:
    - point1 & point2 (np.array): coordinate points e.g. BGR space

    Returns:
    - distance (float): euclidean distance.
    """
    if np.isscalar(point1) and np.isscalar(point2):
        distance = abs(point1 - point2)
    else:
        distance = np.linalg.norm(point1 - point2)

    return distance 

def assign_colors(num_classes):
    """
    Assign a unique color to each class.
    """
    color_map = {}
    for class_idx in range(num_classes + 1):
        # Assign random colors (or you can define specific ones)
        color_map[class_idx] = [random.randint(0, 255) for _ in range(3)]
    return color_map

def naive_region_growing(img, mode=0, tiling_size=2, factor = 1, maxpool_factor = 4):
    """
    Region growing algorithm for segmentation

    Parameters:
    - img (np.array): image to be segmented (BGR format)
    - mode (int): 0 for BGR, 1 for gray, 2 for HSV
    """
    
    if mode == 1:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif mode == 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    
    small_img = cv.resize(img, (img.shape[1] // maxpool_factor, img.shape[0] // maxpool_factor), interpolation=cv.INTER_AREA)
    pooled_img = cv.dilate(small_img, np.ones((2,2), np.uint8))

    height, width = pooled_img.shape[:2]
    img_classes = np.empty((height, width, 1), dtype=np.int32)
    number_classes = 1
    std_dev = np.std(pooled_img) * factor

    for i in range(height):
        for j in range(width):
            current_point = pooled_img[i, j] if pooled_img.ndim == 3 else pooled_img[i, j]
            left_neighbor = None
            upper_neighbor = None

            tile_start_i = max(i - tiling_size, 0)
            tile_start_j = max(j - tiling_size, 0)
            tile_end_i = i
            tile_end_j = j

            if j > 0:
                left_neighbor = pooled_img[i, j - 1] if pooled_img.ndim == 3 else [pooled_img[i, j - 1]]
            if i > 0:
                upper_neighbor = pooled_img[i - 1, j] if pooled_img.ndim == 3 else [pooled_img[i - 1, j]]

            
            std_dev = np.std(pooled_img[tile_start_i:tile_end_i, tile_start_j:tile_end_j]) * factor
                    
            if (i == 0): # First row logic (no upper point)
                if (j == 0): # Seed point
                    img_classes[i,j] = 0
                else:
                    if euclidean_distance(current_point, left_neighbor) > std_dev:
                        img_classes[i,j] = number_classes
                        number_classes += 1
                    else:
                        img_classes[i,j] = img_classes[i,j-1] # Assigning left neighbor class
            elif (j == 0): # First column logic (no left point)
                if euclidean_distance(current_point, upper_neighbor) > std_dev:
                    img_classes[i,j] = number_classes
                    number_classes += 1
                else:
                    img_classes[i,j] = img_classes[i-1,j] # Assigning upper neighbor class
            else: # No first row, no first column
                if (euclidean_distance(current_point, upper_neighbor) > std_dev) and (euclidean_distance(current_point, left_neighbor) > std_dev):
                    img_classes[i,j] = number_classes
                    number_classes += 1
                else:
                    if euclidean_distance(current_point, upper_neighbor) < euclidean_distance(current_point, left_neighbor):
                        img_classes[i,j] = img_classes[i-1,j] # Assigning upper neighbor class
                    else:
                        img_classes[i,j] = img_classes[i,j-1] # Assigning left neighbor class

    color_map = assign_colors(number_classes)
    result_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            class_idx = int(img_classes[i, j])
            result_image[i, j] = color_map[class_idx]

    plot_img(3,(9,4.5),["Original", "Preprocessed", "Naive Segmented"],[img,pooled_img,result_image])
    print(number_classes)

def t_statistic(region,p):
    """
    Returns the t-statistic, works for one channel or three channels (recursively)

    Parameters:
    - Region (np.array): set of points that are part from the region
    - p (np.array or ): new point (e.g., BGR, HSV or grayscale)
    """
    if p.ndim == 1 and len(p) == 3:
        b,g,r = p[0], p[1], p[2]
        B,G,R = region[:, :, 0], region[:, :, 1], region[:, :, 2]
        t_b = t_statistic(B,b)
        t_g = t_statistic(G,g)
        t_r = t_statistic(R,r)

        return np.sqrt(t_b**2 + t_g**2 + t_r**2) 
    else:
        N = region.size
        mean_region = np.mean(region)
        var_region = np.var(region)

        if var_region == 0:
            return 0
        
        return np.sqrt((N - 1) * N / (N + 1) * ((p - mean_region) ** 2) / var_region)

def preprocess_image(img, maxpool_factor, cycles_pre, resize_factor=4):
    """
    Preprocess image for segmentation algorithm

    Parameters:
    - img (np.array): image to be segmented (BGR format)
    - maxpool_factor (int): resizing factor to downscale the image.
    - cycles_pre (int): number of cycles of erode and dilute to remove noise.
    - cycles_pos (int): number of cycles of opening and closing to remove noise.
    - resize_factor (int): decrease the image size by that factor for faster convergence

    Returns:
    - pre_img (np.array): 
    """
    height, width = img.shape[:2]
    resized_image = cv.resize(img, (width // resize_factor, height // resize_factor), interpolation=cv.INTER_LINEAR)
    small_img = cv.resize(resized_image, (resized_image.shape[1] // maxpool_factor, resized_image.shape[0] // maxpool_factor), interpolation=cv.INTER_AREA)
    pooled_img = cv.dilate(small_img, np.ones((2,2), np.uint8))

    kernel = np.ones((5,5),np.uint8)
    kernel_mor = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    
    pre_img = cv.erode(pooled_img, kernel, iterations=cycles_pre)
    pre_img = cv.dilate(pre_img, kernel, iterations=cycles_pre)

    pre_img = cv.morphologyEx(pre_img, cv.MORPH_OPEN, kernel_mor)
    pre_img = cv.morphologyEx(pre_img, cv.MORPH_CLOSE, kernel_mor)

    return pre_img

def region_growing_2(img, mode=0, resize_factor = 2, maxpool_factor = 4, T_threshold = 2, merge_threshold = 4, cycles_pre=1, cycles_pos=2, init="random"):
    """
    Region growing algorithm for segmentation using T-statistic

    Parameters:
    - img (np.array): image to be segmented (BGR format)
    - mode (int): 0 for BGR, 1 for gray, 2 for HSV
    - resize_factor (int): resizing factor before max pooling.
    - maxpool_factor (int): resizing factor to downscale the image.
    - T_threshold (float): threshold for T-statistic to decide region inclusion.
    - merge_threshold (float): merge threshold (should be softer than T_threshold)
    - cycles_pre (int): number of cycles of erode and dilute to remove noise.
    - cycles_pos (int): number of cycles of opening and closing to remove noise.
    - init (str): 'random' for random initialization, 'center' for center, anything else for upper left corner (0,0). 
    """
    
    if mode == 1:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif mode == 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    pre_img = preprocess_image(img,maxpool_factor=maxpool_factor,cycles_pre=cycles_pre, resize_factor=resize_factor)

    height, width = pre_img.shape[:2]
    img_classes = np.full((height, width), -1, dtype=np.int32)
    number_classes = 0

    if init == "random":
        seed_i = random.randint(0, height - 1)
        seed_j = random.randint(0, width - 1)
    elif init == "center":
        seed_i, seed_j = height // 2, width // 2
    else:
        seed_i, seed_j = 0,0

    img_classes[seed_i, seed_j] = number_classes  # First region (seed)
    region_pixels = [(seed_i, seed_j)]

    while region_pixels:
        current_i, current_j = region_pixels.pop(0)
        current_region = pre_img[max(0, current_i - 1):min(height, current_i + 1),
                                   max(0, current_j - 1):min(width, current_j + 1)]

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # Explore neighbors
            ni, nj = current_i + di, current_j + dj
            if 0 <= ni < height and 0 <= nj < width and img_classes[ni, nj] == -1:  # Image constraints
                test_pixel = pre_img[ni, nj]
                t_value = t_statistic(current_region, test_pixel)

                if t_value < T_threshold:
                    img_classes[ni, nj] = number_classes
                    region_pixels.append((ni, nj))

        if not region_pixels: # If no more pixels in the region, looks for unclassified
            unclassified = np.argwhere(img_classes == -1)
            if len(unclassified) > 0:
                seed_i, seed_j = unclassified[0]
                number_classes += 1
                img_classes[seed_i, seed_j] = number_classes
                region_pixels.append((seed_i, seed_j))  # Start a new region

    color_map_pre_merge = assign_colors(number_classes)
    pre_merge_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            class_idx = int(img_classes[i, j])
            pre_merge_image[i, j] = color_map_pre_merge[class_idx]
    
    mean_colors = [] # Merging similar regions
    for class_id in range(number_classes + 1):
        region_mask = (img_classes == class_id)
        mean_color = pre_img[region_mask].mean(axis=0)
        mean_colors.append(mean_color)

    for i in range(number_classes):
        for j in range(i + 1, number_classes + 1):
            if np.linalg.norm(mean_colors[i] - mean_colors[j]) < merge_threshold:
                img_classes[img_classes == j] = i

    unique_classes = np.unique(img_classes)
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
    
    img_classes = np.vectorize(class_mapping.get)(img_classes)
    
    number_classes = len(unique_classes)
    color_map = assign_colors(number_classes)
    result_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            class_idx = int(img_classes[i, j])
            result_image[i, j] = color_map[class_idx]

    pos_image = result_image
    
    kernel_mor = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    for i in range(cycles_pos):
        pos_image = cv.morphologyEx(pos_image, cv.MORPH_OPEN, kernel_mor)
        pos_image = cv.morphologyEx(pos_image, cv.MORPH_CLOSE, kernel_mor)

    plot_img(5, (15, 7.5), ["Original", "Preprocessed", "Segmented (Pre-Merge)", "Segmented (Post-Merge)", "Postprocessed"], [img, pre_img, pre_merge_image, result_image, pos_image])
    print(number_classes)


def kmeans_segmentation(img, k=3, cycles_pos=1):
    """
    K-means segmentation of an image with k clusters. Each pixel will be assigned the mean color of its cluster.

    Parameters:
    - img (np.array): Input image in BGR format (from OpenCV).
    - k (int): Number of clusters for K-means.
    
    Returns:
    - segmented_img (np.array): Image where each pixel has the mean color of its cluster.
    """
    pixel_values = img.reshape((-1, 3))  # flatten image
    pixel_values = np.float32(pixel_values)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    _, labels, centers = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    
    segmented_img = centers[labels.flatten()]
    
    segmented_img = segmented_img.reshape(img.shape) # Reshape back to the original shape
    
    kernel_mor = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) # For opening and closing (cleaning segmentation)
    for i in range(cycles_pos):
        pos_image = cv.morphologyEx(segmented_img, cv.MORPH_OPEN, kernel_mor)
        pos_image = cv.morphologyEx(pos_image, cv.MORPH_CLOSE, kernel_mor)

    plot_img(3, (9, 4.5), ["Original", "Segmented", "Segmented (cleaned)"], [img, segmented_img, pos_image])

    return segmented_img

def kmeans_segmentation(img, k=3, cycles_pos=1):
    """
    K-means segmentation of an image with k clusters. Each pixel will be assigned the mean color of its cluster.

    Parameters:
    - img (np.array): Input image in BGR format (from OpenCV).
    - k (int): Number of clusters for K-means.
    
    Returns:
    - segmented_img (np.array): Image where each pixel has the mean color of its cluster.
    """
    pixel_values = img.reshape((-1, 3))  # flatten image
    pixel_values = np.float32(pixel_values)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    _, labels, centers = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    
    segmented_img = centers[labels.flatten()]
    
    segmented_img = segmented_img.reshape(img.shape) # Reshape back to the original shape
    
    kernel_mor = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) # For opening and closing (cleaning segmentation)
    for i in range(cycles_pos):
        pos_image = cv.morphologyEx(segmented_img, cv.MORPH_OPEN, kernel_mor)
        pos_image = cv.morphologyEx(pos_image, cv.MORPH_CLOSE, kernel_mor)

    plot_img(3, (9, 4.5), ["Original", "Segmented", "Segmented (cleaned)"], [img, segmented_img, pos_image])

    return segmented_img

def Fuzzy_kmeans_segmentation(img, k=3, cycles_pos=1):
    """
    K-means segmentation of an image with k clusters. Each pixel will be assigned the mean color of its cluster.

    Parameters:
    - img (np.array): Input image in BGR format (from OpenCV).
    - k (int): Number of clusters for K-means.
    
    Returns:
    - segmented_img (np.array): Image where each pixel has the mean color of its cluster.
    """
    pixel_values = img.reshape((-1, 3))  # flatten image
    pixel_values = np.float32(pixel_values)
    
    pixel_values = pixel_values.T

    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(pixel_values, k, 2, error=0.005, maxiter=1000)    
    labels = np.argmax(u, axis=0)
    
    centers = np.uint8(cntr)
    
    segmented_img = centers[labels]
        
    segmented_img = segmented_img.reshape(img.shape)
    
    kernel_mor = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) # For opening and closing (cleaning segmentation)
    for i in range(cycles_pos):
        pos_image = cv.morphologyEx(segmented_img, cv.MORPH_OPEN, kernel_mor)
        pos_image = cv.morphologyEx(pos_image, cv.MORPH_CLOSE, kernel_mor)

    plot_img(3, (9, 4.5), ["Original", "Segmented", "Segmented (cleaned)"], [img, segmented_img, pos_image])

    return segmented_img

def outsu_segmentation(img, cycles_pos, print_img=True):

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    pixel_prob = hist / hist.sum()    
    omega = np.cumsum(pixel_prob)
    mean_cumulative = np.cumsum(np.arange(256) * pixel_prob)
    global_mean = mean_cumulative[-1]
    
    omega[omega == 0] = 1e-6
    
    omega_inv = 1 - omega
    
    sigma_b_squared = (global_mean * omega - mean_cumulative) ** 2 / (omega * omega_inv)
    
    optimal_threshold = np.argmax(sigma_b_squared)
    
    _, segmented_image = cv.threshold(img, optimal_threshold, 255, cv.THRESH_BINARY)

    segmented_img = -segmented_image+255

    kernel_mor = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) # For opening and closing (cleaning segmentation)
    pos_image = segmented_img
    
    for i in range(cycles_pos):
        pos_image = cv.morphologyEx(pos_image, cv.MORPH_OPEN, kernel_mor)
        pos_image = cv.morphologyEx(pos_image, cv.MORPH_CLOSE, kernel_mor)
    edges = cv.Canny(pos_image, 100, 200)
    if print_img:
        plot_img(4, (12, 6), ["Original", "Segmented", "Segmented (cleaned)", "edges"], [img, segmented_img, pos_image, edges])   
    return pos_image

def map_distance(img):
    ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY) 
    dist = cv.distanceTransform(thresh, cv.DIST_L2, 5) 
    dist_output = cv.normalize(dist, None, 0, 1.0, cv.NORM_MINMAX)
    kernel = np.ones((5, 5), np.uint8) 
    plot_img(2, (8, 4), ["Original", "dist_output"], [img, dist_output])   

def skeletonize_manual(binary_img):
    skeleton = np.zeros(binary_img.shape, np.uint8)
    
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    
    img = binary_img.copy()
    
    while True:
        eroded = cv.erode(img, kernel)
        
        opened = cv.dilate(eroded, kernel)
        
        temp = cv.subtract(img, opened)
        
        skeleton = cv.bitwise_or(skeleton, temp)
        
        img = eroded.copy()
        
        if cv.countNonZero(img) == 0:
            break

    plot_img(2, (8, 4), ["Original", "dist_output"], [binary_img, skeleton])   
    return skeleton

def pipeline_features(color_img):
    copy_img = color_img.copy()
    binary_img = outsu_segmentation(color_img,1,False)
    _, binary_img = cv.threshold(binary_img, 127, 255, cv.THRESH_BINARY)
    num_labels, labels = cv.connectedComponents(binary_img)

    label_hue = np.uint8(179 * labels / np.max(labels)) 
    blank_ch = 255 * np.ones_like(label_hue)

    labeled_image = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)  
    labeled_image[label_hue == 0] = 0
    
    for i in range(1, num_labels): 
        component_mask = np.uint8(labels == i) * 255  # Binary mask for the current component

        contours, _ = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        perimeter = cv.arcLength(contours[0], True)
        area = cv.contourArea(contours[0])
        moments = cv.moments(contours[0])
        hu_moments = cv.HuMoments(moments).flatten()

        print(f"Component {i}:")
        print(f"  Perimeter: {perimeter:.2f} pixels")
        print(f"  Area: {area:.2f} pixels^2")
        print(f"  Hu Moments: {hu_moments}")

        masked_color_image = cv.bitwise_and(color_img, color_img, mask=component_mask)

        for j, color_name in enumerate(['Blue', 'Green', 'Red']):
            # Extract the color channel
            color_channel = masked_color_image[:, :, j]
            
            # Only consider the pixels belonging to the component (non-zero in mask)
            component_pixels = color_channel[component_mask > 0]
            
            # Compute mean and variance
            mean_color = np.mean(component_pixels)
            variance_color = np.var(component_pixels)

            # Display the results for this color channel
            print(f"  {color_name} Channel - Mean: {mean_color:.2f}, Variance: {variance_color:.2f}")

        cv.drawContours(copy_img, contours, -1, (0, 255, 0), 2)

    plot_img(3, (9, 4.5), ["Color image (contoured)", "Binary", "Connected"], [copy_img, binary_img, labeled_image])
