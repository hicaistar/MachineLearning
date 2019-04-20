import tensorflow as tf
import os 
import glob
import numpy as np
import cv2
import random

tf.app.flags.DEFINE_string('train_tfrecords_path',  './tfrecords/train_v1.tfrecord',
                            """Needs to provide the dataset_path  as in training.""")

tf.app.flags.DEFINE_string('train_dir',  './datasets/flowers/flower_small/train',
                            """Needs to provide the summary output dir in training.""")

FLAGS = tf.app.flags.FLAGS



def encode_to_tfrecord(file_path,tfrecord_path,col=None,row=None):
    label_list = os.listdir(file_path)
    label_list = [label for label in label_list if not label.startswith('.')]
    char_dict=dict(zip(label_list,range(len(label_list))))
    files = glob.glob(file_path+'/*/*.jpg')
    files.sort()
    labels = [char_dict[file.split('/')[-2]] for file in files]
    data = zip(files,labels)
    data = list(data)
    random.shuffle(data)
    # data = shuffle_data_and_label(files)
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

# shuffled  test data in file stage
    for image_path,label in data:
            img = cv2.imread(image_path)
            height,width,channel = img.shape
            if col and row:
                img = cv2.resize(img,(col,row))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'channel':tf.train.Feature(int64_list=tf.train.Int64List(value=[channel]))
            }))
            writer.write(example.SerializeToString())
    #     print(label)
    writer.close()

def main(argv=None):
    encode_to_tfrecord(FLAGS.train_dir,FLAGS.train_tfrecords_path)
        
if __name__ == '__main__':
    tf.app.run()

