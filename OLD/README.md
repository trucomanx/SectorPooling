# Install SectorPooling

Install SectorPooling following https://github.com/trucomanx/SectorPooling/blob/main/README_install.md 

# SectorPooling example code

The next code shows an example use of SectorPooling library.

```python
import tensorflow as tf
from SectorPooling import Sector4Pooling2D

input_shape=(512, 512,3);

model = tf.keras.Sequential([
    Sector4Pooling2D(factor=0.5,input_shape=input_shape)
])

model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


```

# SectorPooling example files

Example files can be found at [example.py](example.py).
