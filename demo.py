import argparse
import glob
from pathlib import Path

# 尝试导入 open3d 进行3D数据可视化，如果失败则使用 mayavi 作为备用
try:
    import open3d
    from visual_utils import open3d_vis_utils as V  # 可视化工具
    OPEN3D_FLAG = True  # 设置标志，表示使用 open3d
except:
    import mayavi.mlab as mlab  # 导入 mayavi 作为备用
    from visual_utils import visualize_utils as V  # 可视化工具
    OPEN3D_FLAG = False  # 如果使用 mayavi，则标志为 False

import numpy as np
import torch

# 导入 OpenPCDet 库相关模块
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# 自定义数据集类 DemoDataset 继承自 DatasetTemplate
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        初始化 DemoDataset 数据集类
        Args:
            root_path: 数据集根目录
            dataset_cfg: 数据集配置
            class_names: 类别名称
            training: 是否用于训练
            logger: 日志记录器
            ext: 文件扩展名，默认为 .bin 格式
        """
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.root_path = root_path
        self.ext = ext
        # 获取指定路径下的所有点云数据文件
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()  # 对数据文件进行排序
        self.sample_file_list = data_file_list  # 存储点云文件列表

    def __len__(self):
        # 返回数据集大小
        return len(self.sample_file_list)

    def __getitem__(self, index):
        # 获取某个点云文件的数据并加载
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)  # 读取 bin 文件
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])  # 读取 npy 文件
        else:
            raise NotImplementedError  # 未实现其他格式

        # 创建数据字典
        input_dict = {
            'points': points,
            'frame_id': index,
        }

        # 准备数据
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


# 配置文件解析函数
def parse_config():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')  # 配置文件路径
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')  # 点云数据路径
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')  # 预训练模型路径
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')  # 点云文件格式

    # 解析命令行参数
    args = parser.parse_args()

    # 从配置文件加载设置
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


# 主函数
def main():
    # 解析命令行参数
    args, cfg = parse_config()
    # 创建日志记录器
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    # 初始化数据集
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')  # 输出数据集大小

    # 构建模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    # 加载预训练模型参数
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()  # 将模型移动到GPU
    model.eval()  # 设置为评估模式

    # 不需要梯度计算，进行推理
    with torch.no_grad():
        # 遍历数据集中的每个点云样本
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')  # 输出当前处理的样本索引
            # 将样本数据集合并
            data_dict = demo_dataset.collate_batch([data_dict])
            # 将数据加载到GPU
            load_data_to_gpu(data_dict)
            # 进行前向推理
            pred_dicts, other_dicts = model.forward(data_dict)

            # 打印预测结果
            print(f"Prediction for sample {idx + 1}:")
            print(pred_dicts)
            print(f"other_dicts for sample {idx + 1}:")
            print(other_dicts)

            # 可视化预测结果
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            # 如果使用 mayavi 进行可视化
            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')  # 完成

# 程序入口
if __name__ == '__main__':
    main()

