from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""
def gauss1d(sigma):
    # sigma에 6을 곱한 후 올림 계산
    len = int(np.ceil(sigma * 6))
    if len % 2 == 0: 
        # len이 짝수라면 1을 더해서 홀수로 만들어주기
        len += 1 

    # 시작점이 -(len // 2), 끝점이 len // 2, 구간 내 숫자의 개수가 len인 1D array 생성
    # ex) sigma=1.6일 때 x=[-5, -4, -3, -2, -1 ,0, 1, 2, 3, 4, 5]
    x = np.linspace( -(len // 2), len // 2, len)
    
    # 각각의 원소에 Gaussian function 적용
    gauss1d_filter = np.exp(-x**2 / (2 * sigma**2))

    # 합이 1이 되도록 normalization
    gauss1d_filter /= gauss1d_filter.sum()

    return gauss1d_filter

def gauss2d(sigma):
    # 1D gaussian filter끼리 외적 연산
    gauss2d_filter = np.outer(gauss1d(sigma), gauss1d(sigma)) # n*n size

    # 합이 1이 되도록 normalization
    gauss2d_filter /= gauss2d_filter.sum()

    return gauss2d_filter

def convolve2d(array,filter):
    # input을 'np.float32' type으로 변환
    array = np.asarray(array, dtype=np.float32)
    filter = np.asarray(filter, dtype=np.float32)

    # filter 상하좌우로 뒤집기
    filter_flipped = np.flip(filter)

    # zero padding array 만들기
    # zero padding 없이 convolution을 수행하면, 출력 이미지의 크기는 입력 이미지보다 작아진다.
    
    # filter의 크기
    f = filter.shape[0] 
    # zero padding 크기 m 계산
    m = (f - 1) // 2

    # array의 상하좌우 모든 방향으로 m개의 행과 열을 0으로 채우기
    array_padded = np.pad(array, ((m, m), (m, m)), 'constant', constant_values=0)

    # 입력 이미지와 동일한 크기로 출력 이미지 초기화 
    output = np.zeros_like(array)
    
    # 이중 반복문으로 convolution 수행
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # zero padding이 적용된 이미지에서 convolution을 수행할 작은 영역 선택
            # 현재 pixel을 중심으로 filter 크기만큼의 영역을 선택 
            # ex) i=0, j=0, 3*3 kernel => array_padded[0:3, 0:3]
            area = array_padded[i : i + f, j : j + f]
            
            # 상하좌우로 뒤집은 filter로 convolution 연산 수행
            output[i, j] = np.sum(area * filter_flipped)

    return output

def gaussconvolve2d(array,sigma):
    # 'gauss2d' 함수로 2D gaussian filter 생성
    filter = gauss2d(sigma)

    # 'convolve2d' 함수로 convolution 연산 수행
    output = convolve2d(array, filter)

    return output

def reduce_noise(img):
    """ Return the gray scale gaussian filtered image with sigma=1.6
    Args:
        img: RGB image. Numpy array of shape (H, W, 3).
    Returns:
        res: gray scale gaussian filtered image (H, W).
    """
    # greyscale 이미지로 변환
    im_grey = img.convert('L')
    im_array = np.asarray(im_grey, dtype=np.float32)

    # 'gaussconvolve2d' 함수로 filter 생성 후 convolution 연산 수행
    im_after_convolution = gaussconvolve2d(im_array, 1.6)

    # float를 uint8로 다시 변환 후 res에 저장
    res = im_after_convolution.astype(np.uint8)  

    # 결과를 다시 PIL 이미지로 변환한 후 저장
    im_filtered = Image.fromarray(res)

    # 원본 이미지와 블러링된 이미지 보여주기
    img.show()
    im_filtered.show()
    
    return res

def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    # X, Y Sobel filter
    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # x, y 방향으로 Sobel filter 적용
    # intensity x, y 값 얻기 위해 convolve2d 함수 사용
    Ix = convolve2d(img, x_filter) 
    Iy = convolve2d(img, y_filter)
    
    # x-axis gradient image
    Image.fromarray(np.abs(Ix).astype(np.uint8)).save('./x_axis_gradient.bmp')
    # y-axis gradient image
    Image.fromarray(np.abs(Iy).astype(np.uint8)).save('./y_axis_gradient.bmp')

    # Magnitude of gradient 구하기
    # np.hypot(x1, x2)는 sqrt(x1^2 + x2^2)와 동일
    G = np.hypot(Ix, Iy)
    # G를 0에서 255 사이 값으로 매핑
    G = G / G.max() * 255

    # Direction of gradient 구하기
    theta = np.arctan2(Iy, Ix)

    return (G, theta)

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    # 라디안을 도(degree)로 바꾸기
    angle = theta * (180 / np.pi)
    # sobel filter에서 arctan2 사용했기 때문에 angle은 -180도에서 180도 사이의 값을 가짐
    # angle이 음수라면 180 더해주기 
    angle = np.where(angle < 0, angle + 180, angle)

    # G의 행의 개수, 열의 개수 확인
    m, n = G.shape
    # non-maximum suppression을 적용한 이미지인 res 초기화 
    res = np.zeros((m, n), dtype=np.float32)

    direction = np.array([0, 45, 90, 135])

    # 제일 끝(edge)은 이웃 픽셀이 충분하지 않으므로 NMS 연산 적용하지 않음
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            # angle이 0, 45, 90, 135 중 어디에 가장 가까운지 찾기
            # 이때 argmin()은 direction의 특정 index 반환
            closest_direction =  direction[np.abs(direction - angle[i, j]).argmin()]

            # direction이 0도라면 (왼쪽 픽셀) vs (현재 픽셀), (오른쪽 픽셀) vs (현재 픽셀)의 G(gradient magnitude) 비교
            if (closest_direction == 0):
                pixel1 = G[i, j - 1]
                pixel2 = G[i, j + 1]
            # direction이 45도라면
            elif (closest_direction == 45):
                pixel1 = G[i - 1, j + 1]
                pixel2 = G[i + 1, j - 1]
            # direction이 90도라면
            elif (closest_direction == 90):
                pixel1 = G[i - 1, j]
                pixel2 = G[i + 1, j]
            # direction이 135도라면
            elif (closest_direction == 135): 
                pixel1 = G[i - 1, j - 1]
                pixel2 = G[i + 1, j + 1] 
            
            # 현재 pixel의 G가 두 이웃 픽셀의 G보다 크거나 같다면 값 유지
            if (pixel1 <= G[i, j]) and (pixel2 <= G[i, j]):
                res[i, j] = G[i, j]
            # 현재 pixel의 G가 두 이웃 픽셀의 G보다 작다면 값을 0으로 만들어 중요하지 않은 edge 삭제
            else:
                res[i, j] = 0

    return res

# NMS 연산으로 나온 픽셀들을 strong, weak, non-relevant 총 세 가지 type의 edge로 구분
def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    diff = np.max(img) - np.min(img)
    T_high = np.min(img) + diff * 0.15
    T_low = np.min(img) + diff * 0.03

    # res 초기화
    res = np.zeros(img.shape, dtype=np.uint8)
    # T_high보다 크면 strong edge
    res = np.where(img > T_high, 255, res)
    # T_low와 T_high 사이면 weak edge
    res = np.where((T_low <= img) & (img <= T_high), 80, res)
    # T_low보다 작으면 non-relevant edge
    res = np.where(img < T_low, 0, res)

    return res

def dfs(img, res, i, j, visited=[]):
    # calling dfs on (i, j) coordinate imply that
    #   1. the (i, j) is strong edge
    #   2. the (i, j) is weak edge connected to a strong edge
    # In case 2, it meets the condition to be a strong edge
    # therefore, change the value of the (i, j) which is weak edge to 255 which is strong edge
    res[i, j] = 255

    # mark the visitation
    visited.append((i, j))

    # examine (i, j)'s 8 neighbors
    # call dfs recursively if there is a weak edge
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

# 픽셀을 tracking하여 실제 edge 라인 형성
def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    m, n = img.shape
    res = np.copy(img)
    visited = [] # 방문한 픽셀의 좌표 저장

    for i in range(m):
        for j in range(n):
            # strong edge라면
            if img[i, j] == 255 and (i, j) not in visited:
                # 방문 표시
                visited.append((i, j))
                # 이웃 픽셀 중에 값이 80인 픽셀이 있다면 255로 바꾸고, 재귀적으로 같은 과정 반복
                dfs(img, res, i, j, visited)
    
    # weak edge 제거
    res = np.where(res == 80, 0, res)

    return res

def main():
    RGB_img = Image.open('./iguana.bmp')

    noise_reduced_img = reduce_noise(RGB_img)
    Image.fromarray(noise_reduced_img.astype('uint8')).save('./iguana_blurred.bmp', 'BMP')
    
    g, theta = sobel_filters(noise_reduced_img)
    Image.fromarray(g.astype('uint8')).save('./iguana_sobel_gradient.bmp', 'BMP')
    Image.fromarray(theta.astype('uint8')).save('./iguana_sobel_theta.bmp', 'BMP')

    non_max_suppression_img = non_max_suppression(g, theta)
    Image.fromarray(non_max_suppression_img.astype('uint8')).save('./iguana_non_max_suppression.bmp', 'BMP')

    double_threshold_img = double_thresholding(non_max_suppression_img)
    Image.fromarray(double_threshold_img.astype('uint8')).save('./iguana_double_thresholding.bmp', 'BMP')

    hysteresis_img = hysteresis(double_threshold_img)
    Image.fromarray(hysteresis_img.astype('uint8')).save('./iguana_hysteresis.bmp', 'BMP')

if __name__ == "__main__":
    main()