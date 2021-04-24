import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from keras import Model
from utils.models import model_lenet_fun, model_incept_fun
from sklearn.metrics import confusion_matrix
from utils.mnist_dataset import load_mnist
from utils.models import load_model


num_epochs = 20
learning_rate = 0.001
batch_size = 64


def plot_distro(y, ttl):
    dist = np.zeros((n_classes,), np.float)
    for i in range(n_classes):
        n_this_class = len((y == i).nonzero()[0])
        dist[i] = n_this_class / len(y_train)
    plt.bar(np.arange(0, n_classes, 1), dist)
    plt.title(ttl)
    plt.show()


(X_train, y_train), (X_test, y_test), (X_valid, y_valid),\
                    (n_train, n_test, n_valid), n_classes, image_shape =\
    load_mnist(n_valid=1000)

# plot sample data
nx = 5
ny = 5
indexes = np.random.randint(0, len(X_train), nx * ny)
fig = plt.figure(figsize=(25, 25))
for i in range(nx):
    for j in range(ny):
        index = i * ny + j
        plt.subplot(nx, ny, index + 1)
        image = X_train[indexes[index]]
        plt.imshow(image, cmap='gray')
        plt.ylabel(y_train[indexes[index]])
fig.savefig('images/images.jpg')
plt.show()

# plot samples of class 5
nx = 5
ny = 5
indexes = (y_train == 5).nonzero()[0][: 25]
fig = plt.figure(figsize=(25, 25))
for i in range(nx):
    for j in range(ny):
        index = i * ny + j
        plt.subplot(nx, ny, index + 1)
        image = X_train[indexes[index]]
        plt.imshow(image, cmap='gray')
fig.savefig('images/sample_5.jpg')
plt.show()

# plot distribution of training and testing data
plot_distro(y_train, 'Occurance probablity of each class in training data')
plot_distro(y_test, 'Occurance probablity of each class in testing data')


# load models
model_lenet = load_model(model_lenet_fun, image_shape,
                         n_classes, learning_rate, None)

model_incept = load_model(model_incept_fun, image_shape,
                          n_classes, learning_rate, None)

# Train, Validate and Test the Model
history_lenet = model_lenet.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
                                shuffle=True, validation_data=(X_valid, y_valid), verbose=1)

history_incept = model_incept.fit(X_train, y_train, batch_size=batch_size,
                                  epochs=num_epochs, shuffle=True, validation_data=(X_valid, y_valid), verbose=1)

# saving models
model_incept.save_weights('./models/incept.h5')
model_lenet.save_weights('./models/lenet.h5')

# plot history
plt.plot(history_lenet.history['accuracy'], '-o')
plt.plot(history_lenet.history['val_accuracy'], '-*')
plt.legend(['training acc', 'validation acc'])
plt.title('Lenet model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('images/lenet.jpg')
plt.show()

plt.plot(history_incept.history['accuracy'], '-o')
plt.plot(history_incept.history['val_accuracy'], '-*')
plt.legend(['training acc', 'validation acc'])
plt.title('Inception model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('images/incept.jpg')
plt.show()

# validation set
scores = model_lenet.evaluate(X_test, y_test)
print('------Lenet Evaluation------')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

scores = model_incept.evaluate(X_test, y_test)
print('------Inception Evaluation------')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# plot confusion matrix
y_pred = np.argmax(model_lenet.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = np.log(.0001 + cm)
fig = plt.figure(figsize=(12, 12))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Log of normalized Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
