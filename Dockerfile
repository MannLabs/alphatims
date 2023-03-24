
# FROM python:3.9-slim
FROM --platform=linux/amd64 python:3.9-bullseye

RUN apt-get update && apt-get install -y build-essential gcc python3-dev

RUN adduser worker
USER worker
WORKDIR /home/worker

COPY --chown=worker:worker dist/*.whl /home/worker

# RUN python3 -m pip install ".[plotting-stable]" # Image is 1.6gb with plotting
# The size is reduced to 847 mb without it.
RUN python3 -m pip install --disable-pip-version-check --no-cache-dir --user /home/worker/*.whl
RUN ls /home/worker/.local/lib/python3.9/site-packages/alphatims/ext/timsdata.so
RUN chmod 777 /home/worker/.local/lib/python3.9/site-packages/alphatims/ext/timsdata.so

RUN python3 -m pip cache purge
ENV PATH="/home/worker/.local/bin:${PATH}"

ENTRYPOINT [ "alphatims" ]
