try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse
    import cv2
    import math
    from imutils import contours
    from skimage import measure
    import imutils
    from skimage.exposure import rescale_intensity
    import scipy.stats as st
    from skimage.feature import peak_local_max

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def get_dist(point1, point2):
    dx = math.pow(point1[0] - point2[0], 2)
    dy = math.pow(point1[1] - point2[1], 2)
    return math.sqrt(dx + dy)


def compare_red_points(image, point1, point2):
    first_rgb = image[point1[0], point1[1]]
    second_rgb = image[point2[0], point2[1]]

    first_score = first_rgb[0] * 1 + first_rgb[1] * -0.5 + first_rgb[2] * -0.2
    second_score = second_rgb[0] * 1 + second_rgb[1] * -0.5 + second_rgb[2] * -0.2
    if first_score > second_score:
        return point1
    return point2


def compare_green_points(image, point1, point2):
    first_rgb = image[point1[0], point1[1]]
    second_rgb = image[point2[0], point2[1]]

    first_score = first_rgb[0] * -0.2 + first_rgb[1] * 1 + first_rgb[2] * 0.3
    second_score = second_rgb[0] * -0.2 + second_rgb[1] * 1 + second_rgb[2] * 0.3
    if first_score > second_score:
        return point1
    return point2


def get_cluster(points, threshold, comp_func, image):
    processed_points = {}
    for pair in points:
        pair = tuple(pair)
        for point in processed_points:
            dist = get_dist(point, pair)
            if dist < threshold:
                del processed_points[point]
                new_point = comp_func(image, point, pair)
                processed_points[new_point] = new_point
                break
        else:
            processed_points[pair] = pair
    return processed_points


def find_tfl_lights(image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    kernel = np.array((
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 3, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]))

    kernel = kernel - kernel.mean()

    # find red points
    image_copy = image.copy()
    red_image = image_copy[:, :, 0]
    ret, red_image = cv2.threshold(red_image, 200, 255, cv2.THRESH_BINARY)
    out = cv2.filter2D(red_image, -1, kernel)
    out = ndimage.maximum_filter(out, size=1, mode='constant')

    red_points = peak_local_max(out, min_distance=20)

    red_points = get_cluster(red_points, 20, compare_red_points, image)

    red_x, red_y = [], []
    for point in red_points:
        curr_RGB = image[point[0], point[1]]
        if (curr_RGB[1] < 170 or curr_RGB[2] < 120) and curr_RGB[0] >= 200:
            red_x.append(point[1])
            red_y.append(point[0])

    # find green points
    image_copy = image.copy()
    green_image = image_copy[:, :, 1]
    ret, green_image = cv2.threshold(green_image, 220, 255, cv2.THRESH_BINARY)
    out = cv2.filter2D(green_image, -1, kernel)
    out = ndimage.maximum_filter(out, size=1, mode='constant')

    green = peak_local_max(out, min_distance=20)
    green_points = get_cluster(green, 20, compare_green_points, image)

    green_x, green_y = [], []
    for point in green_points:
        curr_RGB = image[point[0], point[1]]
        if curr_RGB[0] <= 180 and curr_RGB[1] >= 220 and curr_RGB[2] >= 160:
            green_x.append(point[1])
            green_y.append(point[0])

    return red_x, red_y, green_x, green_y


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights2(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)

    plt.plot(red_x, red_y, '+', color='r', markersize=10)
    plt.plot(green_x, green_y, '.', color='g', markersize=10)



def tfl_detect(image_path):
    red_x, red_y, green_x, green_y = find_tfl_lights(image_path, some_threshold=42)
    res=[]
    for i in range(len(red_x)):
        point = [red_x[i],red_y[i]]
        res.append(point)
    for i in range(len(green_x)):
        point=[green_x[i],green_y[i]]
        res.append(point)

    return res