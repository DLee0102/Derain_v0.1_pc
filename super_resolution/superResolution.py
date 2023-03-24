from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from super_resolution.srmodel import SRResNet


def prepareData(input_path_):
    test_tfm = transforms.Compose([
        # transforms.CenterCrop([128, 128]),    # 这行没有必要，用原始图片进行测试即可
        transforms.ToTensor(),
    ])

    # 测试图像的路径，这里不用自己写dataset，直接用现成的ImageFolder即可，因为测试时不需要获取标签
    test_set = ImageFolder(input_path_, transform=test_tfm)

    return test_set

def prepareModel(input_path_, model_path_):
    # net = PReNet_r(use_GPU=True).to('cuda')     # 用cuda加速测试，也可以不用，不用cuda加速测试速度会很慢
    net = SRResNet().to('cuda')     # 用cuda加速测试，也可以不用，不用cuda加速测试速度会很慢
    net.load_state_dict(torch.load(model_path_, map_location=torch.device('cuda'))['model'])      # 加载训练好的模型参数
    net.eval()      # 将模型切换到测试模式

    test_set = prepareData(input_path_)
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    return dataloader, net