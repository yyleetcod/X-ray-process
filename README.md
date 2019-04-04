# X-ray-process
## Landmark Detection
- generate_tfrecord.py: encode input images and generate tfrecord file
- decode_tfrecord.py: decode tfrecord file
- landmark.py: use an 8-layer CNN to regress landmarks, 4 points(8 values) for each image
## Localization
## Classification
- xray: input images
- file_name.py: generating the name list of files in xray
- train.py: resnet18/30 used to classify the size of joints
- train_label.py: the label denoting the size of joints

