import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#Load The Data For Training
mnist=tf.keras.datasets.mnist

(x_train, y_train),(x_test,y_test)=mnist.load_data()
#Normalize The Data
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test, axis=1)
#Build The Model
model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=7)
model.save("handwritten8.model")

#check for the accuracy
model=tf.keras.models.load_model("handwritten8.model")
loss, accuracy = model.evaluate(x_test, y_test)

image_number=1
#Test
while os.path.isfile(f"data2/{image_number}.png"):

    try:
        img=cv2.imread(f"data2/{image_number}.png")[:,:,0]
        img=np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"the prediction is :{np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()

    except:
        print("Error")

    finally:
        image_number+=1

