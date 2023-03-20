
FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential gcc python3-dev
COPY dist/*.whl /app/

# RUN python3 -m pip install ".[plotting-stable]" # Image is 1.6gb with plotting
# The size is reduced to 847 mb without it.
RUN python3 -m pip install /app/*.whl
RUN python3 -m pip cache purge

ENTRYPOINT [ "alphatims" ]
