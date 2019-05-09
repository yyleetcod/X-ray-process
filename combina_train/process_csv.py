import pandas as pd
import random
import os
from PIL import Image
import tensorflow as tf

IMG_SIZE = 300
def crop_and_resize(img):
    w, h = img.size
    crop_size = int(min(w, h) * 5 / 6)
    w_delta = int((w - crop_size) / 2)
    h_delta = int((h - crop_size) / 2)
    box = (w_delta, h_delta, w_delta + crop_size, h_delta + crop_size)
    img = img.crop(box)
    '''
    Image.NEAREST : 低质量
    Image.BILINEAR : 双线性
    Image.BICUBIC : 三次样条插值
    Image.ANTIALIAS : 高质量
    '''
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    return img

feature = pd.read_csv('./label.csv')
"""
for col_name in feature.columns.values:
    print ('{}\t{}'.format(col_name, feature[col_name].dtypes))
"""

pwd = os.getcwd()
path = os.path.join(pwd, 'images')
filename = os.listdir(path) 

train_writer = tf.python_io.TFRecordWriter("./train_data.tfrecords")  
val_writer = tf.python_io.TFRecordWriter("./val_data.tfrecords")
train_count = 0
val_count = 0
skip = [197, 258, 59, 114, 146, 218, 221, 227, 228, 232]
for index, image_path in enumerate(filename):
    cur_path = os.path.join(path, image_path)
    
    first_img_format = os.listdir(cur_path)[0].strip().split('.')[1]
    front_path = os.path.join(cur_path, '1.' + first_img_format)
    front_img = Image.open(front_path).convert('L') 
    front_img = crop_and_resize(front_img)
    front_img_raw = front_img.tobytes()
    
    second_img_format = os.listdir(cur_path)[1].strip().split('.')[1]
    side_path = os.path.join(cur_path, '2.' + second_img_format)
    side_img = Image.open(side_path).convert('L')
    side_img = crop_and_resize(side_img)
    side_img_raw = side_img.tobytes()
   
    if front_img.mode != 'L' or side_img.mode != 'L':
        print (image_path, front_img.mode, side_img.mode)
        print (first_img_format, second_img_format)
    index_str, _, _ = image_path.strip().split('-')
    # if int(index_str) in skip:
    #     continue
    index = int(index_str) - 1
    # sex,height,weight,direction,thigh_bone,shin_bone
    sex = int(feature.loc[index]['sex'])
    height = feature.loc[index]['height']
    weight = feature.loc[index]['weight']
    direction = int(feature.loc[index]['direction'])
    thigh_bone = int(feature.loc[index]['thigh_bone'])
    shin_bone = int(feature.loc[index]['shin_bone'])

    example = tf.train.Example(
        features = tf.train.Features(
            feature={
                "front_img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[front_img_raw])),
                "side_img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[side_img_raw])),
                "sex": tf.train.Feature(int64_list=tf.train.Int64List(value=[sex])),
                "height": tf.train.Feature(float_list=tf.train.FloatList(value=[height])),
                "weight": tf.train.Feature(float_list=tf.train.FloatList(value=[weight])),
                "direction": tf.train.Feature(int64_list=tf.train.Int64List(value=[direction])),
                "thigh_bone": tf.train.Feature(int64_list=tf.train.Int64List(value=[thigh_bone])),
                "shin_bone": tf.train.Feature(int64_list=tf.train.Int64List(value=[shin_bone]))
    }))
    serialized = example.SerializeToString()
    if random.random() <= 0.8:
        train_writer.write(serialized)
        train_count += 1
    else:
        val_writer.write(serialized)
        val_count += 1
print (train_count, val_count)
train_writer.close()
val_writer.close()
