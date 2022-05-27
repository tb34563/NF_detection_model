import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

SIZE_X = 128
SIZE_Y = 128

def segment_images(input_file_name):
    # Load the model
    model = keras.models.load_model('/model.h5', compile=False)

    # Load file names of input images
    test_images = os.listdir(f'/{input_file_name}')

    index = 1
    for image in test_images:

        # Read the image in and process it to match requirements
        image = cv2.imread(f'/{input_file_name}/{image}', cv2.IMREAD_COLOR)       
        image = cv2.resize(image, (SIZE_X, SIZE_Y))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Save the image
        os.mkdir('output')
        os.mkdir('output/Input Images')
        plt.imsave(f'/output/Input Images/{index}.png', image, cmap='gray')
        image = np.expand_dims(image, axis=0)

        # Perform semantic segmentation on image
        prediction = model.predict(image)

        # Save the segmentation mask
        segmentation_mask = prediction.reshape((SIZE_X, SIZE_Y))
        os.mkdir('output/Segmentation Masks')
        plt.imsave(f'/output/Segmentation Masks/{index}.jpg', segmentation_mask, cmap='gray')

        # Overlay the mask on the image, display and save it
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.imshow(segmentation_mask, cmap='gray', alpha=0.5)
        fig.show()
        os.mkdir('output/Segmented Images')
        fig.savefig(f'/output/Segmented Images/{index}.jpg')

        index += 1
