import os
import random
from PIL import Image
import tensorflow as tf


dataset_dir = "flowers_224x224"
flowers = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
validation_size = 0.3
tfrecord_train = "flowers_train.tfrecord"
tfrecord_validation = "flowers_validation.tfrecord"
photo_files = []

def convert_dataset(filenames, tfrecord_file):
    with tf.python_io.TFRecordWriter(tfrecord_file) as tfrecord_writer:
        for file in filenames:
            img = Image.open(file)
            example = tf.train.Example(features=tf.train.Features(
                feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[flowers.index(os.path.basename(os.path.dirname(file)))])),
                         "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))}))
            print(file, flowers.index(os.path.basename(os.path.dirname(file))))
            tfrecord_writer.write(example.SerializeToString())

for flower in flowers:
    for image in os.listdir(os.path.join(dataset_dir, flower)):
        photo_files.append(os.path.join(dataset_dir, os.path.join(flower, image)))
random.shuffle(photo_files)
num_validation = int(validation_size * len(photo_files))
training_filenames = photo_files[num_validation:]
validation_filenames = photo_files[:num_validation]
convert_dataset(training_filenames, tfrecord_train)
convert_dataset(validation_filenames, tfrecord_validation)
print("train:", len(training_filenames))
print("validation:", len(validation_filenames))