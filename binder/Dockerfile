FROM jvail/plantgl-jupyter:latest

ARG NB_USER=jovyan
ARG NB_UID=1000

COPY . /home/${NB_USER}
USER root
SHELL ["/bin/bash", "-c"]
RUN chown -R ${NB_UID} /home/${NB_USER} && pip install -e .
WORKDIR /home/${NB_USER}

USER ${NB_USER}
