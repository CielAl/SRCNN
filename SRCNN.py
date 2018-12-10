# THIS IS THE MAIN SCRIPT

import tensorflow as tf
import os
from training import train
from build_model import build_model

def main():
    if len(sys.argv) != 11:
        raise Exception("Inappropriate number of arguments. Require 9.")
    training = int(sys.argv[1])
    image_size = int(sys.argv[2])
    ground_image_size = int(sys.argv[3])
    num_channels = int(sys.argv[4])
    f1 = int(sys.argv[5])
    n1 = int(sys.argv[6])
    n2 = int(sys.argv[7])
    f3 = int(sys.argv[8])
    num_epoch = int(sys.argv[9])
    batch_size = int(sys.argv[10])

    if training:


        # Code block for getting data: assume 4-D tensor, tf.uint8, images of the same size, ground_images of the same size
        images, ground_images = get_data() # Will implement after Yufei finishes data processing

        
        model = build_model(image_size, ground_image_size, num_channels, f1=f1,n1=n1,n2=n2,f3=f3)
        with tf.Session() as sess:
            train(images, ground_images, model, sess, num_epoch=num_epoch, batch_size=batch_size)
    else:
        pass

if __name__ == '__main__':
    main()
