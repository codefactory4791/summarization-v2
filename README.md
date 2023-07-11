# summarization-v2
Repository for hosting Summarization Project

```bash
gcloud auth login

docker build -t summarization .
docker run -v ~/.config:/root/.config -e GOOGLE_CLOUD_PROJECT=call-summarizatiion summarization
```