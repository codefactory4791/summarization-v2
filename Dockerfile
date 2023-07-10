FROM python:3.9

# 
WORKDIR /app

# 
COPY ./requirements.txt /app/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y


# create enviroinment
ENV PATH $PATH:~/gcloud/google-cloud-sdk/bin

# 
COPY . /app

#move model to artifacts folder
RUN gsutil cp gs://summarization_bucket_2023/pytorch_model.bin /app/app/model/model_artifacts/

# download the model weights in the image
RUN python /app/app/model/model.py

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]