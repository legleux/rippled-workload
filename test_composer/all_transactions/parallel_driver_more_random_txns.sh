#!/usr/bin/env bash

for _ in $(seq 20); do
    curl --silent http://localhost:8000/fill
done
