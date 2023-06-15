# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Load the MNIST dataset
mnist = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

# Convert the dataset into a numpy array
mnist = np.array(mnist)

# Shuffle the dataset
np.random.shuffle(mnist)

# Total number of images in the dataset
total_images = len(mnist)

# Split the dataset into training and testing sets (80%/20%)
train_mnist, test_mnist = mnist[:int(total_images * 0.8)].T, mnist[int(total_images * 0.8):].T

# Split the data into images and labels
train_images, train_labels = train_mnist[1:] / 255, train_mnist[0]
test_images, test_labels = test_mnist[1:] / 255, test_mnist[0]

# Get the number of features (pixels) and images in the training set
n_features, n_images = train_images.shape

# Number of classes (digits)
n_classes = m = 10