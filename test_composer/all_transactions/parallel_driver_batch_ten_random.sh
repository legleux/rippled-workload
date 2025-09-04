#!/usr/bin/env bash

for _ in $(seq 10); do
    curl --silent http://workload:8000/batch/random
done
