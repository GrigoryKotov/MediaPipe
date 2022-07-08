# roi extraction from palm print/vein image
import numpy as np
import cv2


def get_roi(img_original, x2, x1, y2, y1, label):
    h, w, _ = img_original.shape
    img = np.zeros((h + 20, w + 20, 3), np.uint8)
    img[10:-10, 10:-10, :] = img_original
    print(label)
    if label == "Right":
        v1 = np.array([x2 * w, y2 * h])
        v2 = np.array([x1 * w, y1 * h])
    else:
        v2 = np.array([x2 * w, y2 * h])
        v1 = np.array([x1 * w, y1 * h])

    theta = np.arctan2((v2 - v1)[1], (v2 - v1)[0]) * 180 / np.pi
    R = cv2.getRotationMatrix2D(tuple([int(v2[0]), int(v2[1])]), theta, 1)
    v1 = (R[:, :2] @ v1 + R[:, -1]).astype(np.int)
    v2 = (R[:, :2] @ v2 + R[:, -1]).astype(np.int)
    img_r = cv2.warpAffine(img, R, (w, h))

    # ux = int(v1[0] - (v2 - v1)[0] * 0.1)
    # uy = int(v1[1] + (v2 - v1)[0] * 0.1)
    # lx = int(v2[0] + (v2 - v1)[0] * 0.1)
    # ly = int(v2[1] + (v2 - v1)[0] * 1.2)
    ux = int(v1[0] - (v2 - v1)[0] * 0.05)
    uy = int(v1[1] + (v2 - v1)[0] * 0.05)
    lx = int(v2[0] + (v2 - v1)[0] * 0.05)
    ly = int(v2[1] + (v2 - v1)[0] * 1)

    # delta_y is movement value in y ward
    delta_y = (ly - uy) * 0.15
    # delta_y = (ly - uy) * 0.2
    ly = int(ly - delta_y)
    uy = int(uy - delta_y)

    delta_x = (lx - ux) * 0.01
    lx = int(lx + delta_x)
    ux = int(ux + delta_x)

    if label == "Right":
        delta_x = (lx - ux) * 0.05
        lx = int(lx + delta_x)
        ux = int(ux + delta_x)
    # roi = img_r
    roi = img_r[uy:ly, ux:lx]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        print("roi image size is close to zero")
        return None
    if abs(roi.shape[0] - roi.shape[1]) > 70:
        print("ROI image raito is too high!")
        return None
    if roi.shape[0] < 240:
        print("The palm is too far from the camera!")
        return None

    n_zeros = np.count_nonzero(roi == 0)
    if n_zeros > 500:
        print("palm position is invalid")
        return None
    return roi
