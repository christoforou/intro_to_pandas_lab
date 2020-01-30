import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import itertools




def showDecisionBoundary2D(clf):

    x_min = -2.0; x_max = 2.0
    y_min = -2.0; y_max = 2.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.bwr)


def showDataPoint2D(X,y):
    X = np.array(X)
    y = np.array(y)

    c_slow_index= np.where(y==1)    #itemindex = numpy.where(array==item)
    c_fast_index= np.where(y==0)

    plt.scatter(X[c_slow_index,0],X[c_slow_index,1],  c='g', alpha=1.0, marker=r'$\clubsuit$', label="Class A")
    plt.scatter(X[c_fast_index,0],X[c_fast_index,1],  c='c', alpha=1.0, marker=r'$\clubsuit$', label="Class B")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend(loc=2)



# plots a confisuon matrix.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
