from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

gmm_pipe = load('../models/gmm_pipeline.joblib')

# Inside the main.py file, create a function called read_root() that will return a
# dictionary with Hello as key and World as value. Add a decorator to it in order
# to add a GET endpoint to app on the root
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Inside the main.py file, create a function called healthcheck() that will
# return GMM Clustering is all ready to go!. Add a decorator to it in order
# to add a GET endpoint to app on /health with status code 200
@app.get('/health', status_code=200)
def healthcheck():
    return 'GMM Clustering is all ready to go!'

#Inside the main.py file, create a function called format_features() with
# genre, age, income and spending as input parameters that will return a
# dictionary with the names of the features as keys and the inpot parameters as lists
def format_features(genre: str,	age: int, income: int, spending: int):
  return {
        'Gender': [genre],
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [spending]
    }

#Inside the main.py file, Define a function called predict with the following logics:
#input parameters: genre, age, income and spending
#logics: format the input parameters as dict, convert it to a dataframe and make prediction with gmm_pipe
#output: prediction as json
#Add a decorator to it in order to add a GET endpoint to app on /mall/customers/segmentation
@app.get("/mall/customers/segmentation")
def predict(genre: str,	age: int, income: int, spending: int):
    features = format_features(genre,	age, income, spending)
    obs = pd.DataFrame(features)
    pred = gmm_pipe.predict(obs)
    return JSONResponse(pred.tolist())
