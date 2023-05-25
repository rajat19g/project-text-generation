# syntax=docker/dockerfile:1

FROM python:3.11-bullseye

WORKDIR /work

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./p2_infer.py ./p2_infer.py
COPY ./P2.0-model ./P2.0-model


CMD ["uvicorn", "p2_infer:app" , "--host", "0.0.0.0", "--port", "80"]]


