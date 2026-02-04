import numpy as np
import json

def model_fn(model_dir):
    model = np.load(f"{model_dir}/heart_model.npy", allow_pickle=True).item()
    return model

def predict_fn(input_data, model):
    w = model["weights"]
    b = model["bias"]
    z = np.dot(input_data, w) + b
    prediction = 1 / (1 + np.exp(-z))
    return prediction.tolist()

def input_fn(request_body, request_content_type):
    data = json.loads(request_body)
    return np.array(data)

def output_fn(prediction, content_type):
    return json.dumps(prediction)
