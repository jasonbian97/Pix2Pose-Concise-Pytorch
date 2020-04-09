# Pix2Pose Concise Version (Pytorch)

## How to run
1). create python virtual environment with python>3.5 (test pass on python 3.6)
```shell script
pip install -r requirement.txt
```
2). setup config file: cfg_bop2019.json

3). check your file structure, it should be like following:
```
├── Pix2Pose
│   ├── bop_toolkit_lib
│   ├── ...
│   ├── pix2pose_model
├── maskrcnn
│   ├── maskrcnn_config.json
│   ├── maskrcnn_utils.py
│   ├── models
│   └── weights
└── unet
    ├── unet_config.json
    ├── unet_utils.py
    └── weights
```

4). run 
```shell script 
python mainPipeline.py <gpu_id> <path_cfg_json> <dataset_name>
e.g. python mainPipeline.py 0 ./cfg_bop2019.json ycbv
```

