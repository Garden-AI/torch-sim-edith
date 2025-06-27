FROM python:3.12
ARG ENDPOINT_VERSION
RUN pip install globus-compute-endpoint==${ENDPOINT_VERSION}
