# --------------------------------------------#
#   该部分代码用于看网络结构
# --------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary
from ptflops import get_model_complexity_info
from backbone import resnet50_fpn_backbone
from network_files import FasterRCNN

if __name__ == "__main__":
    # input_shape = [816, 612]  # 长和宽要换一下
    # x= torch.randn([1, 3, 816, 612])
    input_shape = [1088, 800]  # 长和宽要换一下
    x = torch.randn([1, 3, 1088, 800])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    # model = backbone(x)
    # print(model)
    # model = FasterRCNN(num_classes, backbone='vgg').to(device)
    model = FasterRCNN(backbone=backbone, num_classes=2).to(device)
    macs, params = get_model_complexity_info(model, (3, 1088, 800), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # model.eval()
    # print(model(x))
    # print(summary(backbone, (3, input_shape[0], input_shape[1])))
    # #
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input,), verbose=False)
    # --------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    # --------------------------------------------------------#
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
