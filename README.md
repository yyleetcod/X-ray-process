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
## Combine_Training
- combin_train.py: combine front image, side image and some numerical information to train model
- process_excel.py: the raw data was in a excel table, process the data into a csv format file
- process_csv.py: process the csv file to tfrecord format
