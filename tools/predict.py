import torch
import os
import numpy as np
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
import datetime

# 1. 加载配置文件和模型
def load_model(config_file, model_pth):
    # 加载配置文件
    cfg_from_yaml_file(config_file, cfg)

    # 构建模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)

    # 加载训练好的模型
    model.load_params_from_file(filename=model_pth, logger=None)

    # 将模型设置为评估模式
    model.eval()

    # 将模型移动到GPU
    model.cuda()
    return model

# 2. 加载数据集
def load_dataloader(config_file, batch_size=1, workers=4, dist=False):
    # 加载配置文件
    cfg_from_yaml_file(config_file, cfg)

    # 构建数据加载器
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=dist, workers=workers, logger=None, training=False
    )
    return test_loader

# 3. 进行推理并保存结果
def inference_and_save(model, test_loader, output_file='predictions.txt'):
    with torch.no_grad():
        # 打开文件准备写入预测结果
        with open(output_file, 'a') as f:
            for i, data_dict in enumerate(test_loader):
                # 将数据移到GPU
                for key in data_dict:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].cuda()

                # 获取模型的输出
                pred_dicts, _, _ = model.forward(data_dict)

                # 提取预测结果
                result = pred_dicts[0]

                # 将每个检测框的结果写入文件
                for i in range(len(result['boxes_lidar'])):
                    box = result['boxes_lidar'][i].cpu().numpy()  # 检测框的坐标
                    score = result['pred_scores'][i].cpu().numpy()  # 置信度
                    label = result['pred_labels'][i].cpu().numpy()  # 类别
                    f.write(f"{label} {score} {' '.join(map(str, box))}\n")
                print(f"Processed batch {i + 1}")

# 4. 主函数
def main(config_file, model_pth, output_file='predictions.txt', batch_size=1, workers=4):
    # 加载模型
    model = load_model(config_file, model_pth)

    # 加载数据集
    test_loader = load_dataloader(config_file, batch_size=batch_size, workers=workers)

    # 执行推理并保存结果
    inference_and_save(model, test_loader, output_file)

if __name__ == '__main__':
    # 你需要替换以下路径为实际的文件路径
    config_file = '/home/luxiaodong/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml'  # 配置文件路径
    model_pth = '/home/luxiaodong/OpenPCDet/output/cfgs/kitti_models/pv_rcnn/default/ckpt/latest_model.pth'  # 模型的 .pth 文件路径
    output_file = 'predictions.txt'  # 输出结果文件名

    # 设置batch size和工作线程数
    batch_size = 1
    workers = 4

    # 调用主函数进行推理
    main(config_file, model_pth, output_file, batch_size, workers)

