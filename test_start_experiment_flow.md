# Local workload test flow

export RIPPLED_IMAGE="rippled:latest"
export SIDECAR_IMAGE="sidecar:latest"
export WORKLOAD_IMAGE="workload:latest"
export CONFIG_IMAGE="config:latest"
export TEST_NETWORK_DIR="testnet"
export REPO="https://github.com/XRPLF/rippled.git"
export COMMIT="develop"
export NETWORK_NAME="antithesis_net"
export NUM_VALIDATORS=5 
export RIPPLED_NAME=rippled 
export VALIDATOR_NAME=val 

1. Checkout rippled-antithesis
2. Checkout rippled-workload _in_ rippled-antithesis
3. Generate network configuration
    uvx --from legleux-generate-ledger gen \
        --config_only False \
        --include_services sidecar-compose.yml \
        --include_services workload-compose.yml
4. Build sidecar image
    cd workload
    docker build sidecar \
            --file sidecar/Dockerfile \
            --tag ${SIDECAR_IMAGE} 
5. Build workload image (still in workload dir)
    docker build $PWD \
        --file Dockerfile.workload \
        --tag ${WORKLOAD_IMAGE}

6. Build config image (still in workload dir)
    # prep network.sh mvs the testnet dir into PWD and massages some values. Will be removed
    ./prep_network.sh
    docker image build $PWD \
      --file Dockerfile.config \
      --build-arg TEST_NETWORK_DIR \
      --tag ${CONFIG_IMAGE}

6.a Make sure it looks good
docker export \
    "$(docker create --name temp "${CONFIG_IMAGE}" true)" | \
    tar -tvf - && docker rm temp

7. Build rippled image (still in workload dir)
    docker build $PWD \
        --file Dockerfile.rippled \
        --tag ${RIPPLED_IMAGE} \
        --build-arg RIPPLED_REPO=${REPO} \
        --build-arg RIPPLED_COMMIT=${COMMIT}

8. Fire it all up!
    
    cd testnet && docker compose up -d
    open https://custom.xrpl.org/localhost:6006/ in your browser or 
    docker exec -it rippled rippled --silent server_info | tail -n+4 | jq .result.info.complete_ledgers

    workload=$(docker inspect workload | jq -r '.[0].NetworkSettings.Networks[].IPAddress')
    # if using a ledgerfile you can
    curl -s "${workload}:8000/accounts" | jq 
    otherwise
