#!/bin/bash

# Save parameters, they will be passed over to the optimum-cli serve command
export ARGS_CLI="${@}"

# Define the prefix for environment variables to look for
PREFIX="SM_ON_"
ARG_PREFIX="--"

# Initialize an array for storing the arguments
# port 8080 required by sagemaker, https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-code-container-response
ARGS=(--port 8080)

# Loop through all environment variables
while IFS='=' read -r key value; do
    # Remove the prefix from the key, convert to lowercase
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]')

    # Add the argument name and value to the ARGS array
    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

# Pass the collected arguments to the optimum-cli serve command
exec optimum-cli neuron serve "${ARGS[@]}" ${ARGS_CLI}
