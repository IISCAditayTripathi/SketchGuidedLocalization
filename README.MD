# The novel contribution of the paper are in the following files:
1. Atttention Model in attention class - lib/model/faster_rcnn/faster_rcnn_oneshot.py
2. Fusion strategies in faster_rcnn_early_fusion.py


# To setup repository, please follow these instructions:

1. Setup Faster R-CNN using instructions given in https://github.com/jwyang/faster-rcnn.pytorch
2. Download pre-trained weights for resnet-50 for image feature extractor and follow instructions given in [image_model](https://github.com/timy90022/One-Shot-Object-Detection)
3. Download pretrained weights for resnet-50 for sketch feature extractor and metadata: [sketch_model](https://drive.google.com/file/d/1W8Y-njfE09aP13nKs_yrQr8XL1R0Tdq0/view?usp=sharing)
4. Store the downloaded data in ../data folder

# Download and Process QuickDraw

Download quickdraw dataset [here](https://quickdraw.withgoogle.com/#) and create the pickle file [processed_quick_draw_paths_common_classes.pkl] by running following commands:
```
python read_store_quick_draw.py
python get_train_test_quick_draw.py
python get_common_classes.py
```

# Train and test the localization model
- Enter default `--sketch-path` and `--sketch_class_2_label paths` in train and test files
- To train - Run: `python train_val_sketch_oneshot.py --dataset coco --net res50 --cuda --seen 7 --nw 10 --bs 10`
- To test - Run: `python test_net_oneshot.py --dataset coco --net res50 --cuda --seen 7 --s <session> --checkepoch <epoch> --p <checkpoint>`
