import os
import tensorflow as tf
import numpy as np
from PIL import Image

for i in range(1, 11):             # 用来表示文件夹1到10
    
    cwd = 'Original/'+str(i)+'/'                           # 第i个文件夹路径
    path_tfrecord = 'Original_tfrecord/'+str(i)+'/'        # tfrecord文件路径
    
    if not os.path.exists(path_tfrecord):
        os.makedirs(path_tfrecord)
        print(path_tfrecord+"  开始转换")
    else:
        print(path_tfrecord+"  开始转换")
        
    #f = open(path_tfrecord+'fileQueue', 'w')              # 用写的方式打开fileQueue这个文件，并赋给f
    with open(path_tfrecord+'fileQueue', 'w') as f:
        
        # 创建一个writer来写 TFRecords 文件
        writer1 = tf.python_io.TFRecordWriter(path_tfrecord+"train.tfrecords")
        writer2 = tf.python_io.TFRecordWriter(path_tfrecord+"test.tfrecords")

        class_path1 = cwd + 'train0' + '/'
        class_path2 = cwd + 'train1' + '/'
        class_path3 = cwd + 'test0' + '/'
        class_path4 = cwd + 'test1' + '/'

        # os.listdir返回指定的文件夹包含的文件或文件夹的名字的列表，它不包括 '.' 和'..'
        for img in os.listdir(class_path1):  
            
            # print(img)
            f.writelines(img + 'train0' + '\n')
            img_path = class_path1 + img     # 每张图片的地址
            # 读取img文件
            img_raw = Image.open(img_path).convert('L')  
            img_raw = img_raw.resize((28, 28))     # 转换图片大小
            img_raw_new = img_raw.tobytes()       # 将图片转化为原生bytes
            
            # tf.train.Example来定义我们要填入的数据格式，然后使用tf.python_io.TFRecordWriter来写入
            example = tf.train.Example(
                # 一个Example中包含Features，Features里包含Feature（这里没s）的字典。最后，Feature里包含有一个FloatList， 
                # 或者ByteList，或者Int64List
                features=tf.train.Features(
                    feature={
                        # example对象对label和image数据进行封装
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                        "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_new]))}))
            writer1.write(example.SerializeToString())      # 序列化为字符串

        for img in os.listdir(class_path2):
            # print(img)
            f.writelines(img + 'train1' + '\n')
            img_path = class_path2 + img
            img_raw = Image.open(img_path).convert('L')
            img_raw = img_raw.resize((28, 28))
            img_raw_new = img_raw.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                        "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_new]))}))
            writer1.write(example.SerializeToString())
        writer1.close()

        for img in os.listdir(class_path3):
            # print(img)
            f.writelines(img + 'test0' + '\n')
            img_path = class_path3 + img
            img_raw = Image.open(img_path).convert('L')
            img_raw = img_raw.resize((28, 28))
            img_raw_new = img_raw.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                        "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_new]))}))
            writer2.write(example.SerializeToString())

        for img in os.listdir(class_path4):
            # print(img)
            f.writelines(img + 'test1' + '\n')
            img_path = class_path4 + img
            img_raw = Image.open(img_path).convert('L')
            img_raw = img_raw.resize((28, 28))
            img_raw_new = img_raw.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                        "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_new]))}))
            writer2.write(example.SerializeToString())
        writer2.close()
    #f.close()
    print("OVER")

# 生成了TFRecords文件，接下来就可以使用队列（queue）读取数据了
def read_and_decode11(filename):
    filename_queue = tf.train.string_input_producer([filename])  # 根据文件名生成一个队列
    reader = tf.TFRecordReader()                                 # 定义一个 reader ，读取下一个 record
    _, serialized_example = reader.read(filename_queue)

    # 解析读入的一个record
    features = tf.parse_single_example(
        serialized_example,
        features={"label": tf.FixedLenFeature([], tf.int64), "img_raw": tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features["img_raw"], np.int8)           # 将字符串解析成图像对应的像素组
    img = tf.reshape(img, [28 * 28 * 1])
    # img = tf.reshape(img,[28,28,1])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features["label"], tf.int32)
    return img, label

for i in range(1,11:
    path_tfrecord = 'Original_tfrecord/'+str(i)+'/'
    
    img_train, label_train = read_and_decode11(path_tfrecord+"train.tfrecords")
    img_test, label_test = read_and_decode11(path_tfrecord+"test.tfrecords")

    label_train = tf.one_hot(indices=tf.cast(label1, tf.int32), depth=2)  # 将一个值化为一个概率分布的向量
    label_test = tf.one_hot(indices=tf.cast(label2, tf.int32), depth=2)

    # 随机打乱生成batch
    img_batch_train, label_batch_train = tf.train.shuffle_batch([img_train, label_train], batch_size=64, capacity=1000, min_after_dequeue=500)
    img_batch_test, label_batch_test = tf.train.shuffle_batch([img_test, label_test], batch_size=13, capacity=13, min_after_dequeue=0)


