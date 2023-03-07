import Dehazed.defog_v2 as defog
import Derain.Derain_platform.PreNet_rtest as derain
import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
import utiles

# 输入图片路径
input_path = './testdata'
# 模型加载路径
model_path = './result_Version6/model_best.ckpt'
# 图片输出路径
output_path = './results'
# 缓存路径
temp_path = './temp/temp_img.jpg'
# 拉普拉斯方差阈值
THRESHOLD = 100

# 主函数
if __name__ == '__main__':
    # 加载模型
    dataloader, net = derain.prepareModel(input_path, model_path)

    # 用于 def transform_invert(img_, transform_train)
    # test_tfm = transforms.Compose([
    #     # transforms.CenterCrop([128, 128]),    # 这行没有必要，用原始图片进行测试即可
    #     transforms.ToTensor(),
    # ])

    # 用于打印日志
    cnt = 0
    total = 0

    # 获取测试用例总数
    for input, label in dataloader:
        total += 1

    for input, label in dataloader:
        cnt += 1
        input = input.to('cuda')        # 用cuda加速测试，也可以不用，不用cuda加速测试速度会很慢
        print("Epoch: " + str(cnt), )

        with torch.no_grad():
            output_image, _ = net(input) # 输出的是张量
            # output_image = tensor_to_np(output_image)
            # output_image = transform_invert(output_image, test_tfm)

            # 将Tensor保存为jpg图像，方便后续去雾读取正确格式的图片
            save_image(output_image, temp_path)

            # print(output_image)   # 用于测试

        # print(output_image)   # 用于测试

        # 读取缓存中的图片
        output_image = cv2.imread(temp_path)

        '''
        此处添加模糊识别算法
        '''
        utiles.LaplacianValue(output_image, THRESHOLD=100)

        # 去雾
        output_image = defog.deFogging(output_image)

        '''
        此处添加超分辨率重建代码
        '''

        utiles.save_img(output_image, output_path, cnt)
        
        print('finished:{:.2f}%'.format(cnt*100/total))
        print("")
