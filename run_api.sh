#!/bin/bash

# Pull docker container
docker pull ivonnem3/state_farm_api:0.1

# Run the Docker container
docker run -d -p 1313:1313 ivonnem3/state_farm_api:0.1
