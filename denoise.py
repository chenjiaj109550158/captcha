import cv2
import numpy as np
import yaml


def denoised_task1(img, show=False):
    with open("parameters.yaml") as f:
        param = yaml.safe_load(f)

    if show:
        cv2.imshow('original', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize_times = 2
    # (h, w) = gry.shape[:2]
    # gry = cv2.resize(gry, (w*resize_times, h*resize_times))
    # cv2.imshow('2', gry)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = np.ones((param['denoise']['kernel_size_t1'],
                     param['denoise']['kernel_size_t1']), np.uint8)
    # denoised = cv2.dilate(gry,kernel,iterations = 1)
    denoised_lil = cv2.morphologyEx(img_grey, cv2.MORPH_CLOSE, kernel)

    ret, denoised_lil = cv2.threshold(denoised_lil, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # ret, denoised_lil = cv2.threshold(denoised_lil, 230, 255, cv2.THRESH_BINARY_INV)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        denoised_lil, None, None, None, 4, cv2.CV_32S)

    areas = stats[1:, cv2.CC_STAT_AREA]
    denoised_big = np.zeros((labels.shape), np.uint8)

    area_threshold = param['denoise']['area_removed_size_t1']
    for i in range(0, nlabels - 1):
        if areas[i] >= area_threshold:  # keep
            denoised_big[labels == i + 1] = 255
    if show:
        cv2.imshow("after denoised", denoised_big)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return denoised_big


def denoised_task2(img, show=False):
    with open("parameters.yaml") as f:
        param = yaml.safe_load(f)

    if show:
        cv2.imshow('original', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize_times = 2
    # (h, w) = gry.shape[:2]
    # gry = cv2.resize(gry, (w*resize_times, h*resize_times))
    # cv2.imshow('2', gry)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = np.ones((param['denoise']['kernel_size_t2'],
                     param['denoise']['kernel_size_t2']), np.uint8)
    # denoised = cv2.dilate(gry,kernel,iterations = 1)
    denoised_lil = cv2.morphologyEx(img_grey, cv2.MORPH_CLOSE, kernel)

    # ret, denoised_lil = cv2.threshold(denoised_lil, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ret, denoised_lil = cv2.threshold(denoised_lil, 230, 255, cv2.THRESH_BINARY_INV)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        denoised_lil, None, None, None, 4, cv2.CV_32S)

    areas = stats[1:, cv2.CC_STAT_AREA]
    denoised_big = np.zeros((labels.shape), np.uint8)

    area_threshold = param['denoise']['area_removed_size_t2']
    for i in range(0, nlabels - 1):
        if areas[i] >= area_threshold:  # keep
            denoised_big[labels == i + 1] = 255
    if show:
        cv2.imshow("after denoised", denoised_big)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return denoised_big

def denoised_task3(img, show=False):
    with open("parameters.yaml") as f:
        param = yaml.safe_load(f)

    if show:
        cv2.imshow('original', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize_times = 2
    # (h, w) = gry.shape[:2]
    # gry = cv2.resize(gry, (w*resize_times, h*resize_times))
    # cv2.imshow('2', gry)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = np.ones((param['denoise']['kernel_size_t3'],
                     param['denoise']['kernel_size_t3']), np.uint8)
    # denoised = cv2.dilate(gry,kernel,iterations = 1)
    denoised_lil = cv2.morphologyEx(img_grey, cv2.MORPH_CLOSE, kernel)

    # ret, denoised_lil = cv2.threshold(denoised_lil, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ret, denoised_lil = cv2.threshold(denoised_lil, 230, 255, cv2.THRESH_BINARY_INV)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        denoised_lil, None, None, None, 4, cv2.CV_32S)

    areas = stats[1:, cv2.CC_STAT_AREA]
    denoised_big = np.zeros((labels.shape), np.uint8)

    area_threshold = param['denoise']['area_removed_size_t3']
    for i in range(0, nlabels - 1):
        if areas[i] >= area_threshold:  # keep
            denoised_big[labels == i + 1] = 255
    if show:
        cv2.imshow("after denoised", denoised_big)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return denoised_big
