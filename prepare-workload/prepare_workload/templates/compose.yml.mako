services:
${validators}
${peers}
% if use_unl:
${unl_service}
% endif
networks:
  ${network_name}:
    name: ${network_name}
