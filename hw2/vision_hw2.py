from PIL import Image
import numpy as np
import math

# part 1-1
def boxfilter(n):
    assert n % 2 == 1, "Dimension must be odd"
    arr = np.full((n, n), 1 / (n**2)) # 모든 요소들이 지정된 값으로 초기화된 N차원 배열 생성
    return arr

# part 1-2
def gauss1d(sigma):
    # 6 times sigma rounded up to the next odd integer
    len = int(np.ceil(sigma * 6))
    if len % 2 == 0: # 짝수라면 홀수로 만들어주기
        len += 1

    x = np.linspace( -(len // 2), len // 2, len) # [-5, -4, -3, -2, -1 ,0, 1, 2, 3, 4, 5]
    
    # apply the array x to the given density function
    gauss1d_filter = np.exp(-x**2 / (2 * sigma**2))
    # normalization
    gauss1d_filter /= gauss1d_filter.sum()

    return gauss1d_filter

# part 1-3
def gauss2d(sigma):
    # use 'np.outer' with the 1D array from the function gauss1d(sigma)
    gauss2d_filter = np.outer(gauss1d(sigma), gauss1d(sigma)) # n*n 크기

    # normalization
    gauss2d_filter /= gauss2d_filter.sum()

    return gauss2d_filter

# part 1-4 (a)
# 위에서 만들었던 filter(=kernel)를 convolution 연산을 통해 array(=img)에 적용
def convolve2d(array, filter):
    # input variables are in type 'np.float32'
    array = np.asarray(array, dtype=np.float32)
    filter = np.asarray(filter, dtype=np.float32)

    # filter 뒤집기
    filter_flipped = np.flip(filter)

    # make padding array
    # 컨볼루션 연산을 수행할 때, 출력 이미지의 크기는 필터의 크기와 스트라이드(필터가 이동하는 거리)에 따라 달라진다. 제로 패딩 없이 컨볼루션을 수행하면, 출력 이미지의 크기는 입력 이미지보다 작아진다.
    
    # 패딩 크기 m 계산
    f = filter.shape[0] # 필터의 크기
    m = (f - 1) // 2

    array_padded = np.pad(array, ((m, m), (m, m)), 'constant', constant_values=0)

    # 입력 이미지와 동일한 크기로 출력 이미지 초기화 
    output = np.zeros_like(array)
    
    # 컨볼루션 수행
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # 패딩된 이미지에서 컨볼루션을 수행할 작은 영역 선택
            # 현재 픽셀을 중심으로 필터의 크기만큼의 영역을 선택 ex) 3*3 kernel => array_padded[0:3, 0:3]
            area = array_padded[i : i + f, j : j + f]
            
            # 컨볼루션 연산 수행
            output[i, j] = np.sum(area * filter_flipped)

    return output

# part 1-4 (b)
def gaussconvolve2d(array, sigma):
    # generating a filter with 'gauss2d'
    filter = gauss2d(sigma)

    # apply it to the array with 'convolve2d(array, filter)'
    output = convolve2d(array, filter)

    return output

# part 1-4 (c), (d)
def part1_4_c_d():
    im = Image.open('hw2/3b_tiger.bmp')
    im_grey = im.convert('L')
    im_array = np.asarray(im_grey, dtype=np.float32)

    im_after_convolution = gaussconvolve2d(im_array, 3)
    im_after_convolution = im_after_convolution.astype(np.uint8) # OSError: cannot write mode F as BMP

    im_filtered = Image.fromarray(im_after_convolution)
    im_filtered.save('hw2/part1_4_result.png','PNG')

    im.show()
    im_filtered.show()

# part 2-1
def low_pass_filter():
    # 이미지를 불러오고 RGB로 변환
    im = Image.open('hw2/3a_lion.bmp')
    im = im.convert('RGB')

    # 각 채널(R, G, B)을 분리
    # 배열에서 모든 가로와 세로 픽셀을 포함하지만 채널은 i로 지정된 특정 채널만을 선택
    # 이렇게 하여 i가 0일 때는 r, 1일 때는 g, 2일 때는 b 채널을 가져온다.
    im_array = np.asarray(im, dtype=np.float32)
    channels = [im_array[:, :, i] for i in range(3)]

    # 각 채널에 대해 filter 적용
    blurred_channels = [gaussconvolve2d(channel, 5) for channel in channels]

    # filter(=blur) 처리된 채널을 다시 합쳐서 이미지 생성
    blurred_array = np.stack(blurred_channels, axis=-1)
    
    # 이미지의 데이터 타입을 uint8로 변환하여 저장 범위 조정
    blurred_array = np.clip(blurred_array, 0, 255).astype(np.uint8)
    
    blurred_im = Image.fromarray(blurred_array, 'RGB')
    blurred_im.save('hw2/part2_1_result.png', 'PNG')
    blurred_im.show()

# part 2-2
def high_pass_filter():
    im = Image.open('hw2/3a_lion.bmp')
    im = im.convert('RGB')

    im_array = np.asarray(im, dtype=np.float32)
    channels = [im_array[:, :, i] for i in range(3)]

    # 각 채널에 대해 filter 적용
    low_freq_channels = [gaussconvolve2d(channel, 5) for channel in channels]
    
    # original - low frequency = high frequency
    # c - lf / for c, lf / in zip
    high_freq_channels = [channel - low for channel, low in zip(channels, low_freq_channels)]

    # 음수 값 가지지 않게 하기 위해 각각 128을 더하고 RGB 이미지 재구성
    # 이미지는 (높이, 너비, 색상 채널)로 구성되는데 axis=-1로 설정하여 3차원 배열의 마지막 차원에 오도록 함
    high_freq_array = np.stack([high + 128 for high in high_freq_channels], axis=-1)
    
    # 이미지의 데이터 타입을 uint8로 변환하여 저장 범위 조정
    # numpy.clip(array, min, max) => array 내의 element들에 대해서 min 값 보다 작은 값들을 min으로, max 값 보다 큰 값들을 max로 바꿔주는 함수
    high_freq_array = np.clip(high_freq_array, 0, 255).astype(np.uint8)
    
    high_freq_im = Image.fromarray(high_freq_array, 'RGB')
    high_freq_im.save('hw2/part2_2_result.png', 'PNG')
    high_freq_im.show()

# part 2-3
def hybrid_image():
    im = Image.open('hw2/3a_lion.bmp')
    im = im.convert('RGB')
    
    im_array = np.asarray(im, dtype=np.float32)
    channels = [im_array[:, :, i] for i in range(3)]

    # low frequency
    low_freq_channels = [gaussconvolve2d(channel, 5) for channel in channels]
    
    # high frequency
    high_freq_channels = [channel - low for channel, low in zip(channels, low_freq_channels)]
    
    # 픽셀 값의 범위를 0과 255 사이로 조정하여 artifact 제거
    low_freq_channels_clamped = [np.clip(channel, 0, 255) for channel in low_freq_channels]
    high_freq_channels_clamped = [np.clip(channel, 0, 255) for channel in high_freq_channels]

    # low frequency + high frequency = hybrid image
    # high frequency는 part2_2처럼 128을 더하지 않고 원래 값 사용
    hybrid_channels = [low + high for low, high in zip(low_freq_channels_clamped, high_freq_channels_clamped)]

    # 
    hybrid_array = np.stack(hybrid_channels, axis=-1)
    hybrid_array = hybrid_array.astype(np.uint8)
    #hybrid_array = np.clip(hybrid_array, 0, 255).astype(np.uint8)
    
    hybrid_im = Image.fromarray(hybrid_array, 'RGB')
    hybrid_im.save('hw2/part2_3_result.png', 'PNG')
    hybrid_im.show()


# test
# part 1-1
# print(boxfilter(3))
# print(boxfilter(4))
# print(boxfilter(7))

# part 1-2
# print(gauss1d(0.3))
# print(gauss1d(0.5))
# print(gauss1d(1))
# print(gauss1d(2))

# part 1-3
# print(gauss2d(0.5))
# print(gauss2d(1))

# part 1-4 (c), (d)
# part1_4_c_d()

# part 2-1
# low_pass_filter()

# part 2-2
# high_pass_filter()

# part 2-3
# hybrid_image()
