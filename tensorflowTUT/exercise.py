import tensorflow as tf

image_raw = tf.gfile.FastGFile("0.jpg", "rb").read()
with tf.Session() as sess:
    image_data = tf.image.decode_png(image_raw)
    print(image_data.eval())
    image_data = tf.image.resize_images(image_data, [100, 100], method=1)
    image_data = tf.image.convert_image_dtype(image_data, tf.uint8)
    encoded_image = tf.image.encode_png(image_data)
    with tf.gfile.FastGFile("1.png", "wb") as f:
        f.write(encoded_image.eval())