# -*- coding: utf-8 -*-
# @Time : 2022/10/1 23:08
# @Author : shuoshuo
# @File : main.py
# @Project : 去雾
import cv2
import numpy as np
import math

# SSIM评估
from skimage.metrics import structural_similarity as sk_cpt_ssim

# 导向滤波（待研究）
def guided_filter(I,p,win_size,eps):

    mean_I = cv2.blur(I,(win_size,win_size))
    mean_p = cv2.blur(p,(win_size,win_size))

    corr_I = cv2.blur(I*I,(win_size,win_size))
    corr_Ip = cv2.blur(I*p,(win_size,win_size))

    var_I = corr_I-mean_I*mean_I
    cov_Ip = corr_Ip - mean_I*mean_p

    a = cov_Ip/(var_I+eps)
    b = mean_p-a*mean_I

    mean_a = cv2.blur(a,(win_size,win_size))
    mean_b = cv2.blur(b,(win_size,win_size))

    q = mean_a*I + mean_b
    return q
def get_min_channel(img):
    return np.min(img,axis=2)
def min_filter(img,r):
    kernel = np.ones((2*r-1,2*r-1))
    return cv2.erode(img,kernel)    #最小值滤波器，可用腐蚀替代
def get_A(img_haze,dark_channel,bins_l):
    hist,bins = np.histogram(dark_channel,bins=bins_l)  #得到直方图
    d = np.cumsum(hist)/float(dark_channel.size)    #累加
    # print(bins)
    threshold=0
    for i in range(bins_l-1,0,-1):
        if d[i]<=0.999:
            threshold=i
            break
    A = img_haze[dark_channel>=bins[threshold]].max()
    #候选区域可视化
    show  = np.copy(img_haze)
    show[dark_channel>=bins[threshold]] = 0,0,255
    # cv2.imwrite('./most_haze_opaque_region.jpg',show*255)
    return A
def get_t(img_haze,A,t0=0.1,w=0.95):
    out = get_min_channel(img_haze)
    out = min_filter(out,r=7)
    t = 1-w*out/A #需要乘上一系数w，为远处的物体保留少量的雾
    t = np.clip(t,t0,1)#论文4.4所提到t(x)趋于0容易产生噪声，所以设置一最小值0.1
    return t

# PSNR评估
# PSNR高于40dB说明图像质量极好（即非常接近原始图像），
# 在30—40dB通常表示图像质量是好的（即失真可以察觉但可以接受），
# 在20—30dB说明图像质量差；
# 最后，PSNR低于20dB图像不可接受
def PSNR(target,ref):
    #必须归一化
    target=target/255.0
    ref=ref/255.0
    MSE = np.mean((target-ref)**2)
    if MSE<1e-10:
        return 100
    MAXI=1
    PSNR = 20*math.log10(MAXI/math.sqrt(MSE))
    return PSNR

def deFogging(input_img, output_path_, index):
    I = input_img/255.0
    dark_channel = get_min_channel(I)
    dark_channel_1 = min_filter(dark_channel,r=7)
    # cv2.imwrite("./dark_channel.jpg", dark_channel_1*255)

    A = get_A(I,dark_channel_1,bins_l=2000)

    t = get_t(I,A)
    t = guided_filter(dark_channel,t,81,0.001)
    t = t[:,:,np.newaxis].repeat(3,axis=2)#升维至(r,w,3)

    J = (I-A)/t +A

    J = np.clip(J,0,1)
    J = J*255
    J =np.uint8(J)

    cv2.imwrite(output_path_ + "/" + "result_" + str(index) + ".jpg",J)

    psnr = PSNR(J,I*255)
    print(f"PSNR:{psnr}")

    # ssim评估，值越接近1图像相似度越高
    ssim = sk_cpt_ssim(J,I*255, win_size=11, data_range=255, multichannel=True)
    print(f"ssim:{ssim}")


