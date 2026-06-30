#===========================
#IMPORTING LIBRARIES========
#===========================
import tensorflow as tf 
import keras as ks
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.models import Sequential
from keras import layers
#===========================
#LOADING THE DATASET========
#===========================
train_data=ks.utils.image_dataset_from_directory(
    'C:/Training',
    image_size=(128,128),
    batch_size=32
)
test_data=ks.utils.image_dataset_from_directory(
    'C:/Testing',
    image_size=(128,128),
    batch_size=32
)

#===========================
#BUILING CNN================
#===========================
model=Sequential([
layers.Input(shape=(128,128,3)),
#===========================
#DATA AUGMENTATION==========
#===========================
layers.RandomZoom(0.2),
layers.RandomRotation(0.1),
layers.RandomFlip('horizontal'),
#===========================
#RESCALING==================
#===========================
layers.Rescaling(1./255),
#===========================
#FIRST CONVOLATIONAL LAYER==
#===========================
Conv2D(32,(3,3),activation='relu'),
MaxPooling2D((2,2)),
#============================
#SECOND CONVOLATIONAL LAYER==
#============================
Conv2D(64,(3,3),activation='relu'),
MaxPooling2D((2,2)),
Dropout(0.1),
#============================
#SECOND CONVOLATIONAL LAYER==
#============================
Conv2D(128,(3,3),activation='relu'),
MaxPooling2D((2,2)),
Dropout(0.125),
Flatten(),
Dense(128,activation='relu'),
Dropout(0.5),
Dense(4,activation='softmax')
])
#=============================
#MODEL COMPILE================
#=============================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_data,epochs=10)
test_loss,test_acc=model.evaluate(test_data,verbose=2)
print("Accuracy:",test_acc)
model.save("C:/Users/Kamran Ali Shah/neurohope-app/backend/models/brain_tumor_model.keras")