import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from model_cl import *
from datasets import Dataset3D
from tensorboardX import SummaryWriter

def train_step(train_data, weight_model, robust_estimator, data_type, prob_type=0, dev='cuda'):
    weight_model.to(data_type)
    # fetch the points, ground truth extrinsic and intrinsic matrices
    correspondences, gt_pose = train_data['correspondences'].to(dev, data_type), \
    train_data['gt_pose'].to(dev, data_type)
    # print("############# correspondences shape ", correspondences.shape) # [batchsize, matched_pointnum, 7]
    # 1. importance score prediction
    # 使用CLNet网络去处理npz中获取到的self.num数量的匹配特征点，其中correspond包含特征点三维左边和匹配置信度
    # 下面的None在Tensor中是用来增加维度的，在对应位置（此处为correspond矩阵最后一维）上增加维度1。
    # 所以下面correspond在经过转置和增维后的维度是：[batchsize, 7, matched_pointnum, 1]
    weights = weight_model(correspondences.transpose(-1, -2)[:, :, :, None]) # 因为没看过CLNet，所以这里的输出就直接当作网络判断的匹配特征点属于内点的概率得了
    # print("########### CLNet output weight shape ", weights.shape) # [batchsize, matched_pointnum]

    # print("########### correspondences矩阵维度 ", correspondences.shape) # 例子[batchsize, point_num, 7]
    # print("########### weights矩阵维度 ", weights.shape) # [batchsize, point_num]

    # import pdb; pdb.set_trace()
    # 2. ransac
    loss_back = 0
    # 这里的for循环是为了对batchsize中的每一组点云分别进行一次RANSAC，然后求出平均误差损失。
    for i, pair in enumerate(correspondences[:, :, :6]):# 这里的for相当于将correspond按照batchsize分割，对各个batchsize的数据进行单独Ransac处理
        # print("############ pair 矩阵维度", pair.shape) # pair = [4000,6]
    # 上面[:,:,6]的原因是，后面的RANSAC不使用npz文件中的score，即不用有源文件提供的匹配得分，而是使用CLNet估计出的weights矩阵。
        Es, loss, avg_loss, _ = robust_estimator(
            pair,
            weights[i],
            gt_pose[i]
        )

        loss_back += avg_loss

    return loss_back/correspondences.shape[0]

def train(
        model, # 这个model就是用来提取特征并完成特征匹配以及置信度计算的。
        estimator, #estimator就是可微分的RANSAC
        train_loader,
        valid_loader,
        opt
):
    # the name of the folder we save models, logs
    saved_file = create_session_string(
        "train",
        opt.sampler,
        opt.epochs,
        opt.fmat,
        opt.nfeatures,
        opt.snn,
        opt.session,
        opt.w0,
        opt.w1,
        opt.w2,
        opt.threshold
    )
    writer = SummaryWriter('results/point/' + saved_file + '/vision', comment="model_vis")
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    valid_loader_iter = iter(valid_loader)

    # save the losses to npy file
    train_losses = []
    valid_losses = []

    if opt.precision == 2:
        data_type = torch.float64
    elif opt.precision == 0:
        data_type = torch.float16
    else:
        data_type = torch.float32

    # start epoch
    for epoch in range(opt.epochs):
        # each step
        # 感觉好像是enumerate在每次for循环的时候，才会去构建一组batchsize数据去供下面的代码训练
        for idx, train_data in enumerate(tqdm(train_loader)):  # 这里提取出来的train_data是一个dict字典，不是tensor张量
# trainloader属于torch中的DataLoad类，他会根据创建时给定的batchsize参数去将一个个npz文件组合成【32，x，x，x】的维度，即满足batchsize大小

            # print("############### 字典 train_data的内容",train_data.keys())
            model.train()
            # train_data是字典，只有correspond和gtpose两个关键字。
            # one step
            optimizer.zero_grad()
    # 这里的traindata维度是 [batchsize, correspond_num, 7] 这里的7表示两个匹配特征的三维点+匹配特征的匹配置信度
            train_loss = train_step(train_data, model, estimator, data_type, prob_type=opt.prob, dev=opt.device)
            train_loss.retain_grad()
            # gradient calculation, ready for back propagation
            if torch.isnan(train_loss):
                print("pls check, there is nan value in loss!", train_loss)
                continue

            try:
                train_loss.backward()
                print("successfully back-propagation", train_loss)

            except Exception as e:
                print("we have trouble with back-propagation, pls check!", e)
                continue

            if torch.isnan(train_loss.grad):
                print("pls check, there is nan value in the gradient of loss!", train_loss.grad)
                continue

            train_losses.append(train_loss.cpu().detach().numpy())
            # for vision
            writer.add_scalar('train_loss', train_loss, global_step=epoch*len(train_loader)+idx)

            # add gradient clipping after backward to avoid gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            # check if the gradients of the training parameters contain nan values
            nans = sum([torch.isnan(param.grad).any() for param in list(model.parameters()) if param.grad is not None])
            if nans != 0:
                print("parameter gradients includes {} nan values".format(nans))
                continue

            optimizer.step()
            # check check if the training parameters contain nan values
            nan_num = sum([torch.isnan(param).any() for param in optimizer.param_groups[0]['params']])
            if nan_num != 0:
                print("parameters includes {} nan values".format(nan_num))
                continue

        torch.save(model.state_dict(), 'results/point/' + saved_file + '/model' + str(epoch) + '.net')
        print("_______________________________________________________")

        # validation 模型验证
        with torch.no_grad():
            model.eval()
            try:
                valid_data = next(valid_loader_iter)
            except StopIteration:
                pass

            valid_loss = train_step(valid_data, model, estimator, data_type, prob_type=opt.prob, dev=opt.device)
            valid_losses.append(valid_loss)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch * len(train_loader) + idx)
            writer.flush()
            print('Step: {:02d}| Train loss: {:.4f}| Validation loss: {:.4f}'.format(
                    epoch*len(train_loader)+idx,
                    train_loss,
                    valid_loss
                ), '\n')

    np.save('results/point/' + saved_file + '/' + 'loss_record.npy', (train_losses, valid_losses))

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':

    # Parse the parameters
    parser = create_parser(
        description="Generalized Differentiable RANSAC.")

    config = parser.parse_args()

    # check if gpu device is available
    config.device = torch.device('cuda:0' if torch.cuda.is_available() and config.device != 'cpu' else 'cpu')
    print(f"Running on {config.device}")

    train_model = CLNet().to(config.device)
    robust_estimator = RANSACLayer3D(config) # 可以看出，可微分的RANSAC和提特征的网络都是分开的，所以提取和匹配特征的网络与可微分RANSAC是相互独立可学的
    # use the pretrained model to initialize the weights if provided.
    if config.model is not None:
        train_model.load_state_dict(torch.load(config.model))
    else:
        train_model.apply(init_weights)
    train_model.train()

    # collect dataset list 官方使用指南里面的 pth参数就是这个datapath参数
    train_scenes = os.listdir(config.data_path)

    train_folders = [config.data_path + '/' + i  + '/' for i in train_scenes]
    # print(train_folders) # trainfolder就是train/3DMatch数据集下每个场景文件夹的名称

# 这里的Dataset3D创建时会在类对象属性中保存每个npz文件的路径
    train_dataset = Dataset3D(train_folders)
    print("@#@####################### ",train_dataset.__len__())
    v_folders = [config.data_path.replace('train', 'val')  + '/' + i + '/'  for i in os.listdir(config.data_path.replace('train', 'val'))]

    v_dataset = Dataset3D(v_folders)
    print("@#@####################### ",v_dataset.__len__())

    print("\n=== BATCH MODE: Training and validation on", len(train_scenes), len(v_folders), "datasets. =================")

    # 下面的 torch.utils.data.DataLoader会根据traindataset类对象中保存的路径，并调用Dataset3D类对象中的__getitem__属性，以及根据batchsize大小来一组数据集
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True
    )
    print(f'Loading training data: {len(train_dataset)} image pairs.')
    valid_data_loader = torch.utils.data.DataLoader(
        v_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True
    )

    print(f'Loading validation data: {len(v_dataset)} image pairs.')

    train(train_model, robust_estimator, train_data_loader, valid_data_loader, config)
