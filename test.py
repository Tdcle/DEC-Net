from loader import *
from models.DEC_Net import DEC_Net
from engine.engine import *
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings

warnings.filterwarnings("ignore")


def main(config):
    test_model_index = 0
    print('#----------Creating logger----------#')
    work_dir = ["/home/jnu/wlc/HME-Net/results/0.9146_",]
    pth_name = ['best-epoch196-dsc0.9146.pth',]
    sys.path.append(work_dir[test_model_index] + '/')
    log_dir = os.path.join(work_dir[test_model_index], 'log')
    checkpoint_dir = os.path.join(work_dir[test_model_index], 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, pth_name[test_model_index])
    outputs = os.path.join(work_dir[test_model_index], 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)

    config.outputs = outputs
    config.save_visualization = True

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]  # [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    model = DEC_Net(num_classes=model_cfg['num_classes'],
                               input_channels=model_cfg['input_channels'],
                               c_list=model_cfg['c_list'],
                               bridge=model_cfg['bridge'], ).cuda()

    print('#----------Preparing dataset----------#')

    test_dataset_1 = get_test_datasets(path_Data=config.data_path_1, task_id=0, mode='binary')
    if config.datasets == "ISIC2017_REFUGE2_TN3K_merge":
        test_dataset_2 = get_test_datasets(path_Data=config.data_path_2, task_id=1, mode='multiclass')
    else:
        test_dataset_2 = get_test_datasets(path_Data=config.data_path_2, task_id=1, mode='binary')
    test_dataset_3 = get_test_datasets(path_Data=config.data_path_3, task_id=2, mode='binary')


    test_loader_1 = DataLoader(test_dataset_1,
                               batch_size=8,
                               shuffle=False,
                               pin_memory=True,
                               num_workers=config.num_workers,
                               drop_last=False)

    test_loader_2 = DataLoader(test_dataset_2,
                               batch_size=8,
                               shuffle=False,
                               pin_memory=True,
                               num_workers=config.num_workers,
                               drop_last=False)

    test_loader_3 = DataLoader(test_dataset_3,
                               batch_size=8,
                               shuffle=False,
                               pin_memory=True,
                               num_workers=config.num_workers,
                               drop_last=False)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion

    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
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


if __name__ == '__main__':
    config = setting_config
    main(config)