# Trout Dataset Tutorial
For the trout dataset template, we only consider the basic scenario: raw point clouds and 
their corresponding annotations. Point clouds are supposed to be stored in `.bin` format.

## Label format
We only consider the most basic information -- category and bounding box in the label template.
Annotations are stored in the `.txt`. Each line represents a box in a given scene as below:
```
# format: [x y z dx dy dz heading category_name]
34.5235 21.5415 1.9735 13.103 8.107 3.967 1.571 Building
71.8115 68.089 1.911 55.964 17.807 3.722 1.571 Tree

```
The box should in the unified 3D box definition (see [README(PCDet)](../README(PCDet).md))

## Files structure
Files should be placed as the following folder structure:
```
PCDet_trout
├── data
│   ├── TROUT
│   │   │── ImageSets
│   │   │   │── test.txt
│   │   │   │── train.txt
│   │   │   │── val.txt
│   │   │── points
│   │   │   │── 000000.bin
│   │   │   │── 999999.bin
│   │   │── points_test
│   │   │   │── 000000.bin
│   │   │   │── 999999.bin
│   │   │── labels
│   │   │   │── 000000.txt
│   │   │   │── 999999.txt
│   │   │── labels_test
│   │   │   │── 000000.txt
│   │   │   │── 999999.txt
├── pcdet
├── tools
```
Dataset splits need to be pre-defined and placed in `ImageSets`

## Hyper-parameters Configurations

### Point cloud features
Modify following configurations in `trout_dataset.yaml` to 
suit your own point clouds.
```yaml
POINT_FEATURE_ENCODING: {
    encoding_type: absolute_coordinates_encoding,
    used_feature_list: ['x', 'y', 'z', 'intensity'],
    src_feature_list: ['x', 'y', 'z', 'intensity'],
}
...
# In gt_sampling data augmentation
NUM_POINT_FEATURES: 4

```

#### Point cloud range and voxel sizes
For voxel based detectors such as SECOND, PV-RCNN and CenterPoint, the point cloud range and voxel size should follow:
1. Point cloud range along z-axis / voxel_size is 40
2. Point cloud range along x&y-axis / voxel_size is the multiple of 16.

Notice that the second rule also suit pillar based detectors such as PointPillar and CenterPoint-Pillar.

### Category names and anchor sizes
Category names and anchor size are need to be adapted to trout datasets.
```yaml
CLASS_NAMES: ['Building', …… , 'Tree'] 
...
MAP_CLASS_TO_KITTI: {
    'Building': 'Building',
     ……,
    'Tree': 'Tree',
}

# In gt sampling data augmentation
PREPARE: {
 filter_by_min_points: ['Building:5', …… , 'Tree:5'],
 filter_by_difficulty: [-1],
}
SAMPLE_GROUPS: ['Building:15', …… , 'Tree:15']
...
```
In addition, please also modify the code for creating infos in `__init__.py` and `trout_dataset.py`

```
PCDet_trout
├── pcdet
│   ├── datasets
│   │   │── __init__.py
```

```python
from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoDataset
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset
from .once.once_dataset import ONCEDataset
from .argo2.argo2_dataset import Argo2Dataset
from .custom.custom_dataset import CustomDataset
from .trout.trout_dataset import TROUTDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset,
    'ONCEDataset': ONCEDataset,
    'CustomDataset': CustomDataset,
    'TROUTDataset': TROUTDataset,
    'Argo2Dataset': Argo2Dataset
}


```

```
PCDet_trout
├── pcdet
│   ├── datasets
│   │   │── trout
│   │   │   │── trout_dataset.py

```

```python
    def get_label(self, idx):
        if self.split == 'test':  # Check whether it is in test mode
            label_file = self.root_path / 'labels_test' / ('%s.txt' % idx)  # Use the point_test folde
        else:
            label_file = self.root_path / 'labels' / ('%s.txt' % idx)  # Other modes use the points folder
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[:-1])
            gt_names.append(line_list[-1])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)


       
    def get_lidar(self, idx):
        if self.split == 'test':  # Check whether it is in test mode
            lidar_file = self.root_path / 'points_test' / ('%s.bin' % idx)  # Use the point_test folder
        else:
            lidar_file = self.root_path / 'points' / ('%s.bin' % idx)  # Other modes use the points folder
            
        assert lidar_file.exists() # Check whether the file exists
        
        point_features = np.fromfile(lidar_file, dtype=np.float32) # Read point cloud data using np.fromfile(), each point contains 4 
        
        point_features = point_features.reshape(-1, 4) # Reshape to (num_points, 4), where each point contains (x, y, z, intensity)
        
        return point_features
```

```python
def create_TROUT_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = TROUTDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split, test_split = 'train', 'val', 'test'  # Add test
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('TROUT_infos_%s.pkl' % train_split)
    val_filename = save_path / ('TROUT_infos_%s.pkl' % val_split)
    test_filename = save_path / ('TROUT_infos_%s.pkl' % test_split)  # Add the file name of test

    print('------------------------Start to generate data infos------------------------')

    # Process the train dataset
    dataset.set_split(train_split)
    TROUT_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(TROUT_infos_train, f)
    print('TROUT info train file is saved to %s' % train_filename)

    # Handle val datasets
    dataset.set_split(val_split)
    TROUT_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(TROUT_infos_val, f)
    print('TROUT info val file is saved to %s' % val_filename)

    # Handle test datasets
    dataset.set_split(test_split)
    TROUT_infos_test = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features  
    )
    with open(test_filename, 'wb') as f:
        pickle.dump(TROUT_infos_test, f)
    print('TROUT info test file is saved to %s' % test_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    
    # Create the groundtruth database just for the train set
    
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')

```

```python
if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_TROUT_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        #Modified based on trout's data set
        create_TROUT_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Building', 'Fully_loaded_cargo_ship', 'Fully_loaded_container_ship', 'Lock_gate', 'Tree', 'Unladen_cargo_ship'],
            data_path=ROOT_DIR / 'data' / 'TROUT',
            save_path=ROOT_DIR / 'data' / 'TROUT',
        )
)
```

The above code is just a few major modifications, if you want to generate your own dataset, the best way is to look for TROUT and trout in the 'trout_dataset.py' code and modify it according to your dataset.



## Create data info
Generate the data infos by running the following command:
```shell
python -m pcdet.datasets.trout.trout_dataset create_TROUT_infos tools/cfgs/dataset_configs/trout_dataset.yaml
```


