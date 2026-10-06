# Stand-in for the SageMaker scikit-learn 1.4-2-py312 image, for local tests.
# The real image is in an AWS-owned ECR registry and needs AWS credentials
# to pull; this one is built from public sources only:
#   - the same Python (3.12) and the pins of ml/requirements.txt
#   - the serving/training packages and pins from the container's own
#     requirements.txt at tag v1.4-2-py312
#   - sagemaker-sklearn-container itself, from GitHub at that tag
#   - the image's collections.Mapping patch for Python 3.12
#   - PEP 668: the real image's Python is the distribution's, which refuses
#     `pip install` into it ("externally-managed-environment"). The marker
#     file below makes this one refuse too, as the endpoint does.
FROM python:3.12-slim
RUN apt-get -qq update && apt-get -qq install -y --no-install-recommends gcc libc6-dev git >/dev/null \
 && rm -rf /var/lib/apt/lists/*
COPY ml/requirements.txt /ml-requirements.txt
RUN pip install -q --no-cache-dir -r /ml-requirements.txt \
 && pip install -q --no-cache-dir -c /ml-requirements.txt \
      sagemaker-containers==2.8.6.post2 sagemaker-inference==1.2.0 sagemaker-training==4.8.0 \
      Flask==1.1.1 itsdangerous==2.0.1 "jinja2<3.0" "MarkupSafe<2.0" Werkzeug==2.0.3 \
      protobuf==3.20.2 retrying==1.3.3 psutil gunicorn==23.0.0 setuptools==80.9.0 \
      boto3==1.43.108 pytest==9.1.1 nbformat==5.11.1 \
 && python3 -c "import sagemaker_containers, os; p = os.path.join(os.path.dirname(sagemaker_containers.__file__), '_mapping.py'); s = open(p).read(); open(p, 'w').write(s.replace('collections.Mapping', 'collections.abc.Mapping'))" \
 && pip install -q --no-cache-dir --force-reinstall six==1.16.0 \
 && pip install -q --no-cache-dir --no-deps "git+https://github.com/aws/sagemaker-scikit-learn-container@v1.4-2-py312" \
 && printf '[externally-managed]\nError=distribution Python, as in the SageMaker image\n' \
      > "$(python3 -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')/EXTERNALLY-MANAGED"
