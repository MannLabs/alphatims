# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 python:3.9-bookworm

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/alphatimsuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    alphatimsuser

COPY requirements requirements

RUN pip install --no-cache-dir  -r requirements/requirements.txt
RUN pip install --no-cache-dir  -r requirements/requirements_plotting.txt

COPY alphatims alphatims
COPY MANIFEST.in MANIFEST.in
COPY LICENSE.txt LICENSE.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install --no-cache-dir ".[stable,plotting-stable]"

RUN chmod 777 /usr/local/lib/python3.9/site-packages/alphatims/ext/timsdata.so

ENV PORT=5006
EXPOSE 5006

# to allow other host ports than 5006
ENV BOKEH_ALLOW_WS_ORIGIN=localhost

#USER alphatimsuser

CMD ["/usr/local/bin/alphatims", "gui", "--port", "5006"]

# build & run:
# docker build --progress=plain -t alphatims .
# DATA_FOLDER=/path/to/local/data
# docker run -p 5006:5006 -v $DATA_FOLDER:/app/data/ -t alphatims