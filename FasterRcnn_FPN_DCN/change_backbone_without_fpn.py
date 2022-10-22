import os
import datetime

import torch

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils


def create_model(num_classes):
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor

    # # vgg16
    # 'pretrained=True代表加载预训练权值'
    # backbone = torchvision.models.vgg16_bn(pretrained=True)
    # # print(backbone)
    # backbone = create_feature_extractor(backbone, return_nodes={"features.42": "0"})
    # # out = backbone(torch.rand(1, 3, 224, 224))
    # # print(out["0"].shape)
    # backbone.out_channels = 512

    # resnet50 backbone
    backbone = torchvision.models.resnet50(pretrained=True)
    # print(backbone)
    backbone = create_feature_extractor(backbone, return_nodes={"layer3": "0"})
    # out = backbone(torch.rand(1, 3, 224, 224))
    # print(out["0"].shape)
    backbone.out_channels = 1024

    # EfficientNetB0
    # backbone = torchvision.models.efficientnet_b0(pretrained=True)
    # # print(backbone)
    # backbone = create_feature_extractor(backbone, return_nodes={"features.5": "0"})
    # # out = backbone(torch.rand(1, 3, 224, 224))
    # # print(out["0"].shape)
    # backbone.out_channels = 112

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = args.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt")
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=args.num_classes + 1)
    # print(model)

    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # for param in model.backbone.parameters():
    #     print(param)
    # for name, parameter in model.backbone.named_parameters():
    #     split_name = name.split(".")[0]
    #     if split_name in ['layer4', 'layer3', 'layer2', 'layer1', 'conv1']:
    #         print(split_name)
    # # define optimizer
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params,
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    #
    #
    #
    #
    # # learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.33)
    #
    # # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    # if args.resume != "":
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #     args.start_epoch = checkpoint['epoch'] + 1
    #     if args.amp and "scaler" in checkpoint:
    #         scaler.load_state_dict(checkpoint["scaler"])
    #     print("the training process from epoch{}...".format(args.start_epoch))
    #
    # train_loss = []
    # learning_rate = []
    # val_map = []
    #
    # for epoch in range(args.start_epoch, args.epochs):
    #     # train for one epoch, printing every 10 iterations
    #     mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
    #                                           device=device, epoch=epoch,
    #                                           print_freq=50, warmup=True,
    #                                           scaler=scaler)
    #     train_loss.append(mean_loss.item())
    #     learning_rate.append(lr)
    #
    #     # update the learning rate
    #     lr_scheduler.step()
    #
    #     # evaluate on the test dataset
    #     coco_info = utils.evaluate(model, val_data_set_loader, device=device)
    #
    #     # write into txt
    #     with open(results_file, "a") as f:
    #         # 写入的数据包括coco指标还有loss和learning rate
    #         result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
    #         txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
    #         f.write(txt + "\n")
    #
    #     val_map.append(coco_info[1])  # pascal mAP
    #
    #     # save weights
    #     save_files = {
    #         'model': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'lr_scheduler': lr_scheduler.state_dict(),
    #         'epoch': epoch}
    #     if args.amp:
    #         save_files["scaler"] = scaler.state_dict()
    #     torch.save(save_files, "./save_weights/resNetFpn-model-{}.pth".format(epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  first frozen backbone and train 5 epochs                   #
    #  首先冻结前置特征提取网络权重（backbone），训练rpn以及最终预测网络部分 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    '冻结backbone'
    for param in model.backbone.parameters():
        param.requires_grad = False

    '模型训练阶段的所有参数，因为冻结了backbone的参数所以其不参与训练'
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    '''通过5个epoch对其进行微调'''
    init_epochs = 50
    for epoch in range(init_epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # 用coco来评价
        coco_info = utils.evaluate(model, val_data_set_loader, device=device)

        # write into txt
        '写入txt中'
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP
    '''对5个epoch进行保存'''
    torch.save(model.state_dict(), "./save_weights/pretrain.pth")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  second unfrozen backbone and train all network     #
    #  解冻前置特征提取网络权重（backbone），接着训练整个网络权重  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    '网络断了的话可以将权值载入上面的权值里，将微调阶段去掉，设置想迭代的次数即可，这个想法错误看resnetfpn里的思想'
    # 冻结backbone部分底层权重,冻结提取特征前的网络（较低层参数不变）
    '''因为voc数据集本身并不大整个训练效果没这个好'''
    for name, parameter in model.backbone.named_parameters():
        split_name = name.split(".")[0]
        if split_name in ['layer4', 'layer3', 'layer2', 'layer1', 'conv1']:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]  # 找到需要训练的所有参数
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)  # 优化器使用的是SGD
    # learning rate scheduler
    'StepLR每隔一定步数学习率下降，每隔三步下降乘0.33的学习率'
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,  # 每隔一定步数降低学习率，每隔5步将学习率*0.33
                                                   step_size=3,
                                                   gamma=0.33)
    '''迭代了20个epoch'''
    num_epochs = 100
    for epoch in range(init_epochs, num_epochs+init_epochs, 1):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        '每隔三步学习率降低并更新学习率'
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_set_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        '''从第10个epoch开始保存'''
        # if epoch > 10:
        #     save_files = {
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch}
        #     torch.save(save_files, "./save_weights/mobile-model-{}.pth".format(epoch))

        # 仅保存最后5个epoch的权重
        if epoch in range(init_epochs, num_epochs+init_epochs, 5):
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, "./save_weights/mobile-model-{}.pth".format(epoch))



    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='./', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
