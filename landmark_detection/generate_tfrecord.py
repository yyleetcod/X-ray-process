import tensorflow as tf
import os
from PIL import Image

path = os.getcwd()  
# print (path)
filename = os.listdir(path)  
# print (filename)
label = []
with open("./label.txt", "r") as f:
    for line in f:
        label.append([])
        points = line.strip().split()
        for point in points:
            label[-1].append(float(point))

writer = tf.python_io.TFRecordWriter("./train.tfrecords")  


index = 0
for img_name in filename:
    is_bmp = img_name.strip().split('.')[-1]
    if not is_bmp == "bmp":
        continue
    img_path = path + '/' + img_name  
    img = Image.open(img_path).convert('RGB')
    # print (img_name)
    # print (img.size)
    # print (img.mode)
    # crop the center 5/6 box
    w, h = img.size 
    crop_size = int(min(w, h) * 5 / 6)
    w_delta = int((w - crop_size) / 2)
    h_delta = int((h - crop_size) / 2)
    box = (w_delta, h_delta, w_delta + crop_size, h_delta + crop_size)
    img = img.crop(box)
    # print (img.size)
    '''
    Image.NEAREST : 低质量
    Image.BILINEAR : 双线性
    Image.BICUBIC : 三次样条插值
    Image.ANTIALIAS : 高质量
    '''
    img = img.resize((250, 250), Image.ANTIALIAS) 
    
    save_flag = True
    if save_flag:
        # save resized-image for testing
        save_name = path + '/test_imag/' + img_name.strip().split('.')[0] + '_resize.bmp'
        img.save(save_name)
    
    img_raw = img.tobytes()
    points = label[index]
    # print (type(points))
    index += 1
    example = tf.train.Example(
        features = tf.train.Features(
            feature={
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=points)),
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    serialized = example.SerializeToString()
    writer.write(serialized)
writer.close()
