import os
import shutil
import random


if __name__ == '__main__':
    data_path = 'data/ILSVRC2012_img'
    dir_names = os.listdir(data_path)
    for dir_name in dir_names:
        dir_path = data_path+'/'+dir_name
        img_names = os.listdir(dir_path)

        random.seed(1)
        random.shuffle(img_names)

        to_path = 'data/ILSVRC2012_train_and_val/valid/'+dir_name
        if not os.path.isdir(to_path):
            os.mkdir(to_path)
        to_path2 = 'data/ILSVRC2012_train_and_val/train/'+dir_name
        if not os.path.isdir(to_path2):
            os.mkdir(to_path2)

        i = 0
        for img_name in img_names:
            from_path = dir_path+'/'+img_name
            if i < 100:
                shutil.move(from_path, to_path)
            else:
                shutil.move(from_path, to_path2)
            i += 1
