FROM ubuntu:20.04

# Prepare system packages
RUN apt-get update

# wget is need to get Elastix package
# libgomp1 is required by elastix
RUN apt-get -qq install wget libgomp1 -y

COPY uploads/bin/* /usr/local/bin/
COPY uploads/lib/* /usr/lib/

CMD elastix
