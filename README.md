# X-ray-process
## Landmark Detection
- generate_tfrecord.py: encode input images and generate tfrecord file
- decode_tfrecord.py: decode tfrecord file
- landmark.py: use 8-layer CNN to regress landmark, 4 points(8 values) for each
