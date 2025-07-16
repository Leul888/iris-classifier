from pydantic import BaseModel
from fastapi import FastAPI
import pickle
import numpy as np


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI()

with open('Iris_model.pkl','rb') as f:
    model = pickle.load(f)

@app.post('/predict')
def predict(features:IrisFeatures):
    data = np.array([[features.sepal_length,features.sepal_width,features.petal_length,features.petal_width]])

    prediction = model.predict(data)[0]

    species= ['setosa','versicolor','virginica']
    predicted_species = species[prediction]

    return {'Prediction':predicted_species}


