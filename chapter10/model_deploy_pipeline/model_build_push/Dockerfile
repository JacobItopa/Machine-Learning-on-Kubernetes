FROM python:3.11-slim-bookworm

WORKDIR /microservice

COPY base_requirements.txt /microservice/
RUN pip install --no-cache-dir -r base_requirements.txt

COPY requirements.txt* /microservice/
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

COPY *.py *.pkl* /microservice/

CMD seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE --grpc-port ${GRPC_PORT} --metrics-port ${METRICS_PORT} --http-port ${HTTP_PORT} --log-level DEBUG
