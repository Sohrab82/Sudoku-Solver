import numpy as np
import matplotlib.pyplot as plt
from utils.models import model_lenet_fun, model_incept_fun
from sklearn.metrics import confusion_matrix
from utils.mnist_dataset import load_mnist
from utils.models import load_model


learning_rate = 0.001
batch_size = 64

(X_train, y_train), (X_test, y_test), (X_valid, y_valid),\
                    (n_train, n_test, n_valid), n_classes, image_shape =\
    load_mnist(n_valid=1000)

# load models
model_lenet = load_model(model_lenet_fun, image_shape,
                         n_classes, learning_rate, './models/lenet.h5')

model_incept = load_model(model_incept_fun, image_shape,
                          n_classes, learning_rate, './models/incept.h5')


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
