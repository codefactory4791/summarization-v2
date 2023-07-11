from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Body, Request
from pathlib import Path
import torch
import gcsfs
import os
import pandas as pd
from torch import nn
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


__version__ = "1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

max_input_length = 1024
max_target_length = 128


model_path = os.path.abspath("/app/app/model/model_artifacts")
model_checkpoint ='t5-small'

fs = gcsfs.GCSFileSystem(project='call-summarizatiion')
with fs.open('summarization_bucket_2023/pytorch_model.bin', 'rb') as f:
    print("writing model...")
    full_model_path = f"{model_path}/pytorch_model.bin"
    open(full_model_path, 'wb').write(f.read())


# add the below logic to load the model
# if os.path.isdir(model_id):
#             if MODEL_HEAD_NAME in os.listdir(model_id):
#                 model_head_file = os.path.join(model_id, MODEL_HEAD_NAME)
#             else:
#                 model_head_file = None



def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

print("current working directory")

directory = os.getcwd()
subfolders = fast_scandir(directory)
print(subfolders)





if os.path.isdir(model_path):
    print("Model Loaded")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(full_model_path)
else:
    print("No Model Artifacts Found")
    tf_model = None


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# input is of type list of string List[str], where each item in the list is an article which needs to be summarized
def predict_pipeline(text_input):

    text = ["summarize : " + item for item in text_input]

    inputs = tokenizer(text, max_length=max_input_length, truncation=True, padding='max_length', return_tensors="pt").input_ids

    outputs = t5_model.generate(inputs, max_length=max_target_length, do_sample=False, num_beams = 3)

    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions = [pred.strip() for pred in predictions]

    # result = pd.DataFrame(list(zip(text_input, predictions)))
    # result.columns = ['Text_Input','Summary']
    # result.to_csv("text_summary.csv")


    return predictions



app = FastAPI()


class TextIn(BaseModel):
    text: list[str]


class PredictionOut(BaseModel):
    summary: list[str]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
	
    predicted_summary = predict_pipeline(payload.text)
    return PredictionOut(summary = predicted_summary)


