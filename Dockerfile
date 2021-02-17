FROM ubuntu:20.04

# Add labels to image
LABEL documentation="https://github.com/SuperElastix/elastix/wiki"
LABEL license="Apache License Version 2.0"
LABEL modelzoo="https://elastix.lumc.nl/modelzoo/"

# Prepare system packages, libgomp1 is required by elastix
RUN apt-get update && apt-get -qq install libgomp1 -y

# Copy necessary files
COPY uploads/bin/* /usr/local/bin/
COPY uploads/lib/* /usr/lib/

COPY uploads/LICENSE /
COPY uploads/NOTICE /

CMD elastix
