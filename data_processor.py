import os
import numpy as np
import cv2
def get_image(path):
    file = os.listdir(path)
    fn = set()
    for i in file:
        if i.split('.')[0] == '':
            continue
        fn.add(path + i.split('.')[0] + '.png')
    return list(fn)

def generator(FLAGS):
    real_image_list = np.array(get_image(FLAGS.real_training_data_path))
    real_lable = np.ones([real_image_list.shape[0]])
    real_image_list = real_image_list[:933]
    real_lable = real_lable[:933]
    fake_image_list = np.array(get_image(FLAGS.fake_training_data_path))
    fake_lable = np.zeros([fake_image_list.shape[0]])
    fake_image_list = real_image_list[:400]
    fake_lable = real_lable[:400]
    image_list = np.concatenate((real_image_list,fake_image_list), axis=0)
    lb = np.concatenate((real_lable,fake_lable), axis=0)

    print('{} training image in {}'.format(image_list,FLAGS.training_data_path))
    index = np.arange(0,image_list.shape[0])
    epoch = 1
    images = []
    lables = []
    while True:
        np.random.shuffle(index)
        for i in index:
            im_fn = image_list[i]
            im = cv2.imread(im_fn)
            cv2.resize(im,dsize=(512,512))

            im = (im / 127,5) - 1.

            images.append(im)
            lables.append(lb[[i]])
            if len(images) == FLAGS.batch_size:
                yield [np.array(images)], [np.array(lables)]
                images = []
                lables = []
        epoch += 1
            
def load_val_data(FLAGS):
    real_image_list = np.array(get_image(FLAGS.real_training_data_path))
    real_lable = np.ones([real_image_list.shape[0]])
    real_image_list = real_image_list[933:]
    real_lable = real_lable[933:]
    fake_image_list = np.array(get_image(FLAGS.fake_training_data_path))
    fake_lable = np.zeros([fake_image_list.shape[0]])
    fake_image_list = real_image_list[400:]
    fake_lable = real_lable[400:]
    image_list = np.concatenate((real_image_list,fake_image_list), axis=0)
    lb = np.concatenate((real_lable,fake_lable), axis=0)
    images = []
    lables = []
    for i in range(400,image_list.shape[0]):
        im_fn = image_list[i]
        im = cv2.imread(im_fn)
        cv2.resize(im,dsize=(512,512))

        im = (im / 127,5) - 1.

        images.append(im)
        lables.append(lb[[i]])
    return [np.array(images)], [np.array(lables)]

def count_sample(FLAGS):
    real_image_list = np.array(get_image(FLAGS.real_training_data_path))
    fake_image_list = np.array(get_image(FLAGS.fake_training_data_path))
    #return (real_image_list.shape[0] + fake_image_list.shape[0]) 
    return 400 + 933