from flask import Flask
from flask import request
from test_model import classifier
from util import get_data
from sklearn.utils import shuffle

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

_, Xtest, _, Ytest = get_data()
Xtest, Ytest = shuffle(Xtest, Ytest)

@app.route('/')
def home():
    return '<h1> Deployment of Iris classifier using simple webapp </h1>'

@app.route('/predict')
def predict(X=Xtest,Y=Ytest):
    classifier.load_model()
    idx = 22 # insert random integer for testing
    y_pred,loss,accuracy = classifier.predict(X[idx],Y[idx])
    return "<h3>Prediction : {} <br>loss : {} <br>accuracy : {}</h3>".format(y_pred,loss,accuracy)

if __name__ == "__main__":
    app.run(debug=True)    