import tensorflow as tf
import os
from PIL import Image

path = os.getcwd()  
print (path)
filename = os.listdir(path)  
print (filename)
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
    #img = img.convert("RGB")
    img = img.resize((250, 250)) 
    img_raw = img.tobytes()
    points = label[index]
    print (type(points))
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
