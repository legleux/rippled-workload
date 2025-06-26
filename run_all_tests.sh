#!/bin/bash

# docker exec workload eventually_payment.sh                    # 1 Payment
docker exec workload parallel_driver_add_accounts.sh          # 1 Payment
docker exec workload parallel_driver_cancel_offer.sh          # 2 OfferCancel [done]
docker exec workload parallel_driver_nftoken_create_offer.sh  # 6 NFTokenCreateOffer [done]
docker exec workload parallel_driver_nftoken_accept_offer.sh  # 7NFTokenAcceptOffer [done]
docker exec workload parallel_driver_nftoken_burn.sh          # 8NFTokenBurn [done]
docker exec workload parallel_driver_nftoken_mint.sh          # 9NFTokenMint [done]
docker exec workload parallel_driver_offer.sh                 # 3 OfferCreate [done]
docker exec workload parallel_driver_payment.sh               # 1 Payment [done]
docker exec workload parallel_driver_ticket.sh                # 4 TicketCreate [done]
docker exec workload parallel_driver_trustset.sh              # 5TrustSet [done]

## test locally
#docker run \
#    --rm \
#    --volume ./workload:/opt/antithesis/catalog/workload \
#    --volume ./test_composer/tc_commands:/opt/antithesis/catalog/tc_commands \
#    -e RIPPLED_NAME=atrippled \
#    -e VALIDATOR_NAME=atval \
#    -e NUM_VALIDATORS=5 \
#    --network atrippled-net \
#    --name "${image_name}" \
#    -it "${image_name}" \
#    bash
