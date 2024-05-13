import numpy as np
import cv2
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    largest_set = []

    for _ in range(10):
        i, j = random.choice(matched_pairs) # 랜덤으로 하나의 match 선택
        consistent_set = [(i, j)] # 초기 match를 포함

        # 랜덤으로 선택된 match의 feature 정보 추출
        row_i, col_i, scale_i, orientation_i = keypoints1[i]
        row_j, col_j, scale_j, orientation_j = keypoints2[j]
        
        # orientation 차이, scale 차이 계산
        base_orientation_difference = (orientation_j - orientation_i) % (2 * np.pi)
        base_scale_difference = scale_j / scale_i

        for k, l in matched_pairs:
            if (k, l) == (i, j):
                continue
            # match의 feature 정보 추출
            row_k, col_k, scale_k, orientation_k = keypoints1[k]
            row_l, col_l, scale_l, orientation_l = keypoints2[l]
            
            # orientation 차이, scale 차이 계산
            orientation_difference = (orientation_l - orientation_k) % (2 * np.pi)
            scale_difference = scale_l / scale_k

            # 하나의 match가 다른 match와 consistent한지 확인
            orientation_difference = (np.abs(orientation_difference - base_orientation_difference) < np.deg2rad(orient_agreement)) or (np.abs(orientation_difference - base_orientation_difference) > (2 * np.pi - np.deg2rad(orient_agreement)))
            scale_difference = base_scale_difference * (1 - scale_agreement) < scale_difference < base_scale_difference * (1 + scale_agreement)

            # 조건 만족하면 consistent_set에 추가
            if orientation_difference and scale_difference:
                consistent_set.append((k, l))
        
        # 이전과 비교하여 가장 큰 집합 찾기
        if len(consistent_set) > len(largest_set):
            largest_set = consistent_set

    ## END
    assert isinstance(largest_set, list)
    return largest_set



def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    y1 = descriptors1.shape[0]
    y2 = descriptors2.shape[0]
    temp = np.zeros(y2)
    matched_pairs = []

    for i in range(y1):
        for j in range(y2):
            temp[j] = math.acos(np.dot(descriptors1[i], descriptors2[j]))
        
        compare = sorted(range(len(temp)), key= lambda k : temp[k])
        if (temp[compare[0]] / temp[compare[1]]) < threshold:
            matched_pairs.append([i, compare[0]])
    ## the following is just a placeholder to show you the output format
    # num = 5
    # matched_pairs = [[i, i] for i in range(num)]
    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    hc_xys = np.pad(xy_points, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=1)
    xys_p = h @ hc_xys.T
    z_cor = np.where(xys_p[-1, :] == 0, 0.0000001, xys_p[-1, :])
    hc_xys_p = xys_p / z_cor
    xys_p = hc_xys_p[:-1, :]
    # END
    return xys_p.T

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START
    max_inliers = 0
    h = None
    N = xy_src.shape[0]

    for i in range(num_iter):
        sample_indices = np.random.choice(N, 4, replace=False)
        src_sample = xy_src[sample_indices]
        ref_sample = xy_ref[sample_indices]

        H = cv2.findHomography(src_sample, ref_sample)
        xy_res = KeypointProjection(xy_src, H[0])
        res_dis = np.sqrt(np.sum((xy_res - xy_ref)**2, axis=1))
        inlier = np.sum(res_dis < tol)

        if inlier > max_inliers:
            h = H[0]
            max_inliers = inlier
    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
