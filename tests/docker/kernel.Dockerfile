# Stand-in for the notebook's "sagemaker-demo" kernel: Python 3.10 (as
# conda_python3 on notebook-al2023-v1) with requirements.txt, exactly what
# the lifecycle script installs.
FROM python:3.10-slim
COPY requirements.txt /requirements.txt
RUN pip install -q --no-cache-dir -r /requirements.txt nbformat==5.11.1
