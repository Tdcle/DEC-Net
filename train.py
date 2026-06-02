from thop import clever_format
from loader import get_datasets
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from models.DEC_Net import DEC_Net

from engine.engine import *

from ALRA import ALRA
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'best.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset_1, test_dataset_1 = get_datasets(path_Data = config.data_path_1, mode='binary', need_val=False)
    if config.datasets == "ISIC2017_REFUGE2_TN3K_merge":
        train_dataset_2, test_dataset_2 = get_datasets(path_Data = config.data_path_2, mode='multiclass', need_val=False)
    else:
        train_dataset_2, test_dataset_2 = get_datasets(path_Data = config.data_path_2, mode='binary', need_val=False)
    train_dataset_3, test_dataset_3 = get_datasets(path_Data = config.data_path_3, mode='binary', need_val=False)

    def repeat_dataset(dataset, target_length):
        """重复较短数据集到目标长度，较长数据集保持不变"""
        if len(dataset) >= target_length:
            return dataset  # 直接返回原始数据集
        times = (target_length + len(dataset) - 1) // len(dataset)
        return torch.utils.data.Subset(
            torch.utils.data.ConcatDataset([dataset] * times),
            indices=range(target_length)
        )

    # 对齐数据集长度
    max_length = max(len(train_dataset_1), len(train_dataset_2), len(train_dataset_3))
    train_dataset_1 = repeat_dataset(train_dataset_1, max_length)
    train_dataset_2 = repeat_dataset(train_dataset_2, max_length)
    train_dataset_3 = repeat_dataset(train_dataset_3, max_length)

    train_loader_1 = DataLoader(train_dataset_1,
                                batch_size=config.batch_size,
                                shuffle=True,
                                pin_memory=False,
                                num_workers=config.num_workers)
    test_loader_1 = DataLoader(test_dataset_1,
                                batch_size=8,
                                shuffle=False,
                                pin_memory=False,
                                num_workers=config.num_workers,
                                drop_last=False)

    train_loader_2 = DataLoader(train_dataset_2,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   pin_memory=False,
                                   num_workers=config.num_workers)
    test_loader_2 = DataLoader(test_dataset_2,
                                  batch_size=8,
                                  shuffle=False,
                                  pin_memory=False,
                                  num_workers=config.num_workers,
                                  drop_last=False)

    train_loader_3 = DataLoader(train_dataset_3,
                                batch_size=config.batch_size,
                                shuffle=True,
                                pin_memory=False,
                                num_workers=config.num_workers)
    test_loader_3 = DataLoader(test_dataset_3,
                               batch_size=8,
                               shuffle=False,
                               pin_memory=False,
                               num_workers=config.num_workers,
                               drop_last=False)

    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    def count_parameters(model):
        """统计模型总参数量"""
        return sum(p.numel() for p in model.parameters())

    # 使用
    analysis_model = DEC_Net(num_classes=model_cfg['num_classes'],
                                input_channels=model_cfg['input_channels'],
                                c_list=model_cfg['c_list'],
                                bridge=model_cfg['bridge'], ).cuda()

    total_params = count_parameters(analysis_model)

    analysis_model.eval()
    input = torch.randn(1, 3, 256, 256).cuda()
    task_id = 0
    epoch = 0
    MACs, params = profile(analysis_model, inputs=(input,task_id,epoch))
    # 将结果转换为更易于阅读的格式
    MACs, params = clever_format([MACs, params], '%.4f')
    print(f"运算量：{MACs}, 参数量：{total_params / 1e6:.4f}M")
    del analysis_model
    torch.cuda.empty_cache()

    model = DEC_Net(num_classes=model_cfg['num_classes'],
                       input_channels=model_cfg['input_channels'],
                       c_list=model_cfg['c_list'],
                       bridge=model_cfg['bridge'], ).cuda()

    model.train()


    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()


    print('#----------Set other params----------#')
    max_dsc = 0.0
    start_epoch = 1
    min_epoch = 1


    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)



    print('#----------Training----------#')
    alra_optimizer = ALRA(
        n_tasks=3,
        device=torch.device("cuda"),
        w_lr=0.1,
        gamma=0.01,
        warmup_epochs=10,
        loss_smooth=0.8,
        floor_value=0.2,
        w_grad_clip=0.1
    )
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        train_one_epoch(
            train_loader_1,
            train_loader_2,
            train_loader_3,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            alra=alra_optimizer,
            scaler=scaler
        )

        dsc = val_one_epoch(
                test_loader_1,
                test_loader_2,
                test_loader_3,
                model,
                criterion,
                epoch,
                logger,
                config
            )


        if dsc > max_dsc:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            max_dsc = dsc
            min_epoch = epoch


    # if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
    print('#----------Testing----------#')
    best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
    model.load_state_dict(best_weight)
    dsc = test_one_epoch(
            test_loader_1,
            test_loader_2,
            test_loader_3,
            model,
            criterion,
            logger,
            config,
        )
    os.rename(
        os.path.join(checkpoint_dir, 'best.pth'),
        os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-dsc{max_dsc:.4f}.pth')
    )


if __name__ == '__main__':
    config = setting_config
    main(config)