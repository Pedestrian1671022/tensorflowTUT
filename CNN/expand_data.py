import os
from PIL import Image

flowers = {"daisy", "dandelion", "roses", "sunflowers", "tulips"}
size = 224
i = 0
for index, flower in enumerate(flowers):
    num = 0
    for image in os.listdir(os.path.join("flowers_train", flower)):
        img_ = Image.open(os.path.join(os.path.join("flowers_train", flower), image))
        width = size
        height = size
        width_ = int(img_.size[0])
        height_ = int(img_.size[1])
        while width < width_:
            while height < height_:
                img = img_.crop((width-size, height-size, width, height))
                if not os.path.exists(os.path.join("flowers_train", flower+"_copy")):
                    os.makedirs(os.path.join("flowers_train", flower+"_copy"))
                img.save(os.path.join(os.path.join("flowers_train", flower+"_copy"), str(num)+".jpg"))

                # if not os.path.exists(os.path.join("flowers_train", flower+"_flip")):
                #     os.makedirs(os.path.join("flowers_train", flower+"_flip"))
                # img.rotate(90).save(os.path.join(os.path.join("flowers_train", flower+"_flip"), str(num)+".jpg"))
                height += 60
                num += 1
            height = size
            width += 60
        i = i+1
print(i)