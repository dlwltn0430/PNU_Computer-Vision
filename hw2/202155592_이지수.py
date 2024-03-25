from PIL import Image
import numpy as np
import math

# part 1-1
def boxfilter(n):
    # 짝수 예외 처리
    assert n % 2 == 1, "Dimension must be odd" 

    # n*n 크기의 배열의 모든 원소 값을 1 / (n**2)로 초기화
    arr = np.full((n, n), 1 / (n**2)) 

    return arr

# part 1-2
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

# part 1-3
def gauss2d(sigma):
    # 1D gaussian filter끼리 외적 연산
    gauss2d_filter = np.outer(gauss1d(sigma), gauss1d(sigma)) # n*n size

    # 합이 1이 되도록 normalization
    gauss2d_filter /= gauss2d_filter.sum()

    return gauss2d_filter

# part 1-4 (a)
# 위에서 만들었던 filter(=kernel)를 convolution 연산을 통해 array(=image)에 적용
def convolve2d(array, filter):
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

# part 1-4 (b)
def gaussconvolve2d(array, sigma):
    # 'gauss2d' 함수로 2D gaussian filter 생성
    filter = gauss2d(sigma)

    # 'convolve2d' 함수로 convolution 연산 수행
    output = convolve2d(array, filter)

    return output

# part 1-4 (c), (d)
def part1_4():
    # 이미지 불러오기
    im = Image.open('hw2/3b_tiger.bmp')
    # greyscale 이미지로 변환
    im_grey = im.convert('L')
    im_array = np.asarray(im_grey, dtype=np.float32)

    # 'gaussconvolve2d' 함수로 filter 생성 후 convolution 연산 수행
    im_after_convolution = gaussconvolve2d(im_array, 3)
    # float를 uint8로 다시 변환 (해당 코드 적지 않으면 "OSError: cannot write mode F as BMP"라고 뜸)
    im_after_convolution = im_after_convolution.astype(np.uint8) 

    # 결과를 다시 PIL 이미지로 변환한 후 저장
    im_filtered = Image.fromarray(im_after_convolution)
    im_filtered.save('hw2/part1_4_result.png','PNG')

    # 원본 이미지와 블러링된 이미지 보여주기
    im.show()
    im_filtered.show()

# part 2-1
def part2_1():
    # 이미지를 불러오기 
    im = Image.open('hw2/3a_lion.bmp')
    # 이미지 모드를 RGB로 변환
    im = im.convert('RGB')

    # 이미지를 numpy array로 변환
    im_array = np.asarray(im, dtype=np.float32)

    # 각 채널(R, G, B)을 분리
    # 배열에서 모든 가로와 세로 픽셀을 포함하지만 채널은 i로 지정된 특정 채널만을 선택
    # 이렇게 하여 i가 0일 때는 R, 1일 때는 G, 2일 때는 B 채널을 가져온다.
    channels = [im_array[:, :, i] for i in range(3)]

    # 각 채널에 대해 filter 적용
    blurred_channels = [gaussconvolve2d(channel, 5) for channel in channels]

    # 각각 분리하여 filter 처리된 채널을 다시 합치기
    blurred_array = np.stack(blurred_channels, axis=-1)
    
    # 이미지의 데이터 타입을 uint8로 변환하여 저장 범위를 0에서 255 사이로 조정
    blurred_array = np.clip(blurred_array, 0, 255).astype(np.uint8)
    
    # 결과를 다시 PIL 이미지로 변환한 후 저장하고 보여주기
    blurred_im = Image.fromarray(blurred_array, 'RGB')
    blurred_im.save('hw2/part2_1_result.png', 'PNG')
    blurred_im.show()

# part 2-2
def part2_2():
    # 이미지를 불러오기 
    im = Image.open('hw2/3b_tiger.bmp')
    # 이미지 모드를 RGB로 변환
    im = im.convert('RGB')

    # 이미지를 numpy array로 변환
    im_array = np.asarray(im, dtype=np.float32)

    # 각 채널(R, G, B)을 분리
    # 배열에서 모든 가로와 세로 픽셀을 포함하지만 채널은 i로 지정된 특정 채널만을 선택
    # 이렇게 하여 i가 0일 때는 R, 1일 때는 G, 2일 때는 B 채널을 가져온다.
    channels = [im_array[:, :, i] for i in range(3)]

    # 각 채널에 대해 filter 적용
    low_freq_channels = [gaussconvolve2d(channel, 5) for channel in channels]
    
    # (original) - (low frequency) = (high frequency)
    # channels과 low_freq_channels의 각 원소들을 튜플 형태로 묶고, 이 튜플들을 순회하면서 원본 채널 데이터(channel)에서 해당 low frequency 채널 데이터(low)를 뺀 값을 계산
    # c - lf / for c, lf / in zip
    high_freq_channels = [channel - low for channel, low in zip(channels, low_freq_channels)]

    # 분리한 채널을 다시 합치기
    # 음수 값 가지지 않게 하기 위해 각각 margin 값인 128을 더하기
    high_freq_array = np.stack([high + 128 for high in high_freq_channels], axis=-1)
    
    # 이미지의 데이터 타입을 uint8로 변환하여 저장 범위를 0에서 255 사이로 조정
    high_freq_array = np.clip(high_freq_array, 0, 255).astype(np.uint8)
    
    # 결과를 다시 PIL 이미지로 변환한 후 저장하고 보여주기
    high_freq_im = Image.fromarray(high_freq_array, 'RGB')
    high_freq_im.save('hw2/part2_2_result.png', 'PNG')
    high_freq_im.show()

# part 2-3
def part2_3():
    # 이미지를 불러오기 
    im = Image.open('hw2/3a_lion.bmp')
    # 이미지 모드를 RGB로 변환
    im = im.convert('RGB')
    
    # 이미지를 numpy array로 변환
    im_array = np.asarray(im, dtype=np.float32)

    # 각 채널(R, G, B)을 분리
    # 배열에서 모든 가로와 세로 픽셀을 포함하지만 채널은 i로 지정된 특정 채널만을 선택
    # 이렇게 하여 i가 0일 때는 R, 1일 때는 G, 2일 때는 B 채널을 가져온다.
    channels = [im_array[:, :, i] for i in range(3)]

    # low frequency
    low_freq_channels = [gaussconvolve2d(channel, 5) for channel in channels]

    #**************************#

    # 이미지를 불러오기 
    im2 = Image.open('hw2/3b_tiger.bmp')
    # 이미지 모드를 RGB로 변환
    im2 = im2.convert('RGB')
    
    # 이미지를 numpy array로 변환
    im_array2 = np.asarray(im2, dtype=np.float32)

    # 각 채널(R, G, B)을 분리
    # 배열에서 모든 가로와 세로 픽셀을 포함하지만 채널은 i로 지정된 특정 채널만을 선택
    # 이렇게 하여 i가 0일 때는 R, 1일 때는 G, 2일 때는 B 채널을 가져온다.
    channels2 = [im_array2[:, :, i] for i in range(3)]

    # low frequency
    low_freq_channels2 = [gaussconvolve2d(channel, 5) for channel in channels2]
    
    # high frequency
    high_freq_channels = [channel - low for channel, low in zip(channels2, low_freq_channels2)]
    
    # pixel 값의 범위를 0과 255 사이로 조정하여 artifact 제거
    low_freq_channels_clamped = [np.clip(channel, 0, 255) for channel in low_freq_channels]
    high_freq_channels_clamped = [np.clip(channel, 0, 255) for channel in high_freq_channels]

    # (low frequency) + (high frequency) = (hybrid image)
    # high frequency는 part2_2처럼 128을 더하지 않고 원래 값 사용
    hybrid_channels = [low + high for low, high in zip(low_freq_channels_clamped, high_freq_channels_clamped)]

    # 분리한 채널 다시 합치기
    hybrid_array = np.stack(hybrid_channels, axis=-1)
    # 이미지의 데이터 타입을 uint8로 변환
    hybrid_array = hybrid_array.astype(np.uint8)
    
    # 결과를 다시 PIL 이미지로 변환한 후 저장하고 보여주기
    hybrid_im = Image.fromarray(hybrid_array, 'RGB')
    hybrid_im.save('hw2/part2_3_result.png', 'PNG')
    hybrid_im.show()


# # test
# # part 1-1
# print(boxfilter(3))
# print(boxfilter(4))
# print(boxfilter(7))

# # part 1-2
# print(gauss1d(0.3))
# print(gauss1d(0.5))
# print(gauss1d(1))
# print(gauss1d(2))

# # part 1-3
# print(gauss2d(0.5))
# print(gauss2d(1))

# # part 1-4 (c), (d)
# part1_4()

# # part 2-1
# part2_1()
    
# # part 2-2
# part2_2()

# # part 2-3
# part2_3()
