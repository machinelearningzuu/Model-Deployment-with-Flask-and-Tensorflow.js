from util import get_data
from variables import saved_weights
import os
from iris_classifier import IrisClassifier
import numpy as np
from sklearn.utils import shuffle

current_dir = os.getcwd()
saved_weights = os.path.join(current_dir,saved_weights)

if __name__ == "__main__":
    Xtrain, Xtest, Ytrain, Ytest = get_data()
    Xtest, Ytest = shuffle(Xtest, Ytest)
    classifier = IrisClassifier()
    if os.path.exists(saved_weights):
        print("Loading existing model !!!")
        classifier.load_model()
    else:
        print("Training the model  and saving!!!")
        classifier.mnist_model()
        classifier.train()
        classifier.save_model()
    
    # idx = np.random.randint(len(Xtest))
    idx = 12
    classifier.predict(Xtest[idx],Ytest[idx])