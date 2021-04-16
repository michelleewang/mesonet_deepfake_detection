import numpy as np
import matplotlib.pyplot as plt
import time
from classifiers import *
#from pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'train_test',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        #subset='training'
        )

# check class assignment for deepfake and real images
print(generator.class_indices)

# 3 - Prediction Example
# render image X with label y for MesoNet
X, y = generator.next()
# Evaluating prediction
print('Predicted likelihood:', classifier.predict(X), '\nReal class :', y[0])
print('Correct prediction : ', np.round(classifier.predict(X))==y[0])

# 4 - Prediction for a video dataset
"""
classifier.load('weights/Meso4_F2F.h5')

predictions = compute_accuracy(classifier, 'train_test')
for video_name in predictions:
    print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
"""
# show image
"""plt.imshow(np.squeeze(X))
plt.show()"""

# Creating separate lists for correctly classified and misclassified images
correct_real = []
correct_real_pred = []

correct_deepfake = []
correct_deepfake_pred = []

misclassified_real = []
misclassified_real_pred = []

misclassified_deepfake = []
misclassified_deepfake_pred = []

# Generating predictions on validation set, storing in separate lists
for i in range(len(generator.labels)):
    # Loading next picture
    X, y = generator.next()

    if np.round(classifier.predict(X)) == y[0] and y[0] == 1:
        correct_real.append(X)
        correct_real_pred.append(classifier.predict(X))
    elif np.round(classifier.predict(X)) == y[0] and y[0] == 0:
        correct_deepfake.append(X)
        correct_deepfake_pred.append(classifier.predict(X))
    elif np.round(classifier.predict(X)) != y[0] and y[0] == 1:
        misclassified_real.append(X)
        misclassified_real_pred.append(classifier.predict(X))
    else:
        misclassified_deepfake.append(X)
        misclassified_deepfake_pred.append(classifier.predict(X))

    # if i % 1000 == 0:
        # print(i, "predictions completed.")
    print(i)

    if i == len(generator.labels)-1:
        print("All", len(generator.labels), "predictions completed.")

def plotter(images, preds):
    fig = plt.figure(figsize=(16,9))
    subset = np.random.randint(0, len(images)-1, 12)
    for i,j in enumerate(subset):
        fig.add_subplot(3,4,i+1)
        plt.imshow(np.squeeze(images[j]))
        print(preds[j])
        plt.xlabel("Model confidence: ", preds[j])
        plt.tight_layout()
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
    plt.show;
    return

# show correctly predicted real images
plotter(correct_real, correct_real_pred)

# show misclassified real images
plotter(misclassified_real, misclassified_real_pred)

# show correctly predicted deepfake images
plotter(correct_deepfake, correct_real_deepfake)

# show misclassified deepfake images
plotter(misclassified_deepfake, misclassified_deepfake_pred)
