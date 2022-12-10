import torch
import whisper
import os
import base64
import boto3
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global s3

    model = whisper.load_model("base")
    s3 = boto3.client('s3')

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global s3

    # Parse out your arguments
    s3Url = model_inputs.get('s3Url', None)
    if s3Url == None:
        return {'message': "No input provided"}

    # Extract the bucket and key from the S3 URL
    bucket, key = s3Url.split('/')[2], '/'.join(s3Url.split('/')[3:])

    # Download the file from S3
    s3.download_file(bucket, key, 'input.mp3')
    
    # Run the model
    result = model.transcribe("input.mp3")
    output = {"text":result["text"]}
    os.remove("input.mp3")
    # Return the results as a dictionary
    return output
