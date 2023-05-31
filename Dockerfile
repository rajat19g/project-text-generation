FROM python:3.11-bullseye
WORKDIR /work

RUN pip3 --timeout=100 --no-cache-dir install 'transformers[torch]'
RUN pip3 --no-cache-dir install fastapi
RUN pip3 --no-cache-dir install uvicorn

WORKDIR /work

COPY ./p2_infer.py ./p2_infer.py
COPY ./P2.0-model ./P2.0-model


CMD ["uvicorn", "p2_infer:app" , "--host", "0.0.0.0", "--port", "80"]]