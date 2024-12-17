ARG VARIANT=3.12-bookworm
ARG VESPA_CLI_VERSION=8.448.13
FROM mcr.microsoft.com/devcontainers/python:${VARIANT}

ENV PYTHONUNBUFFERED 1
ENV TZ Asia/Tokyo

RUN apt-get update && apt-get install -y openjdk-17-jdk
# Set JDK_HOME and JAVA_HOME environment variables
ENV JDK_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64