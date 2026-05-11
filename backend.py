import numpy as np
import pickle
from config import NEW_DATA_INPUT_FEATURES_NAME, PLACEMENT_EQUALS_1_PRIOR
import scipy.stats as s
from fastapi import FastAPI, Query
from typing import Annotated
from pydantic import BaseModel


def determine_normalizing_probability(placement_equals_0_likelihood, placement_equals_1_likelihood):
    return (placement_equals_0_likelihood * (1 - PLACEMENT_EQUALS_1_PRIOR)) + \
        (placement_equals_1_likelihood * PLACEMENT_EQUALS_1_PRIOR)

def determine_placement_posterior_probability(input_features):

    input_features = np.array(input_features)
    input_features = input_features.reshape(1,input_features.shape[0])
    eig_vectors = np.load("eigen_vectors.npy")
    new_input_features = np.matmul(input_features,eig_vectors)

    placement_equals_1_likelihood = 1.0
    placement_equals_0_likelihood = 1.0

    with open("likelihood_distribution_params.pkl","rb") as file_handle:
        likelihood_distribution_params = pickle.load(file_handle)

    for input_feat, input_feat_value in zip(NEW_DATA_INPUT_FEATURES_NAME, new_input_features):
        mu_0, sigma_0 = likelihood_distribution_params[0][input_feat]
        mu_1, sigma_1 = likelihood_distribution_params[1][input_feat]
        
        p_input_feature_on_0_placement = s.norm.pdf(input_feat_value,mu_0,sigma_0)
        p_input_feature_on_1_placement = s.norm.pdf(input_feat_value,mu_1,sigma_1)

        placement_equals_0_likelihood = placement_equals_0_likelihood * p_input_feature_on_0_placement
        placement_equals_1_likelihood = placement_equals_1_likelihood * p_input_feature_on_1_placement

    normalizing_probability = determine_normalizing_probability(placement_equals_0_likelihood, placement_equals_1_likelihood) 
    
    placement_equals_0_posterior = placement_equals_0_likelihood * (1 - PLACEMENT_EQUALS_1_PRIOR) / normalizing_probability
    placement_equals_1_posterior = placement_equals_1_likelihood * PLACEMENT_EQUALS_1_PRIOR / normalizing_probability

    if placement_equals_1_posterior[0] > placement_equals_0_posterior[0]:
        return {"result":"Given your inputs, most likeliy you are going to get placed and the probability of you getting placed is roughly {}".format(placement_equals_1_posterior[0])}
    else:
        return {"result":"Given your inputs, most likely you are not going to get placed and the probability of you getting placed is roughly {}".format(placement_equals_1_posterior[0])}


class InputFeatureVector(BaseModel):
    iq: Annotated[int, Query(ge=40, le=160)]
    previous_semester_result: Annotated[float, Query(ge=0.0, le=10.0)]
    cgpa: Annotated[float, Query(ge=0.0, le=10.0)]
    communication_skills: Annotated[int, Query(ge=0, le=10)]
    projects_completed: Annotated[int, Query(ge=0, le=5)]
    
app = FastAPI()
@app.get("/")
def home_page():
    return "This Web API tells the probability of a student getting placed based on his/her IQ, previous semester result, CGPA, communication skills and number of projects completed."

@app.post("/compute_probability")
     
def compute_probability(input_features: InputFeatureVector):
    input_features_values_list = list()
    for input_feature_name ,input_features_values in InputFeatureVector.model_fields.items():
        input_features_values_list.append(getattr(input_features, input_feature_name))
    
    response = determine_placement_posterior_probability(input_features_values_list)

    return response