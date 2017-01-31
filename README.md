# keras-unet
U-Net model for Keras

### The original articles
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Examples 

### Train images
```
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from keras.optimizers import Adam
from unet import UNet, preprocess_input, dice_coef, dice_coef_loss

IMG_HEIGHT = 128
IMG_WIDTH = 128

# Prepare Sample Datasets.
X, y = [], []
for i in range(1000):
    img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))
    label = np.zeros((IMG_WIDTH, IMG_HEIGHT, 1))
    cx, cy = np.random.randint(0, IMG_WIDTH), np.random.randint(0, IMG_HEIGHT)
    w, h = np.random.randint(10, 30), np.random.randint(10, 30)
    color = (np.random.random(3) * 256).astype(int)
    cv2.rectangle(img, (cx, cy), (cx + w, cy + h), color, -1)
    cx, cy = np.random.randint(0, IMG_WIDTH), np.random.randint(0, IMG_HEIGHT)
    r = np.random.randint(10, 40)
    color = (np.random.random(3) * 256).astype(int)
    cv2.circle(img, (cx, cy), r, color, -1)
    cv2.circle(label, (cx, cy), r, color, -1)
    label[label > 0] = 1
    X.append(img)
    y.append(label)
X, y = np.asarray(X), np.asarray(y)
X = preprocess_input(X)
print X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Training.
model = UNet(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=100)
```

