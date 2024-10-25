# NeuronX TGI benchmark using multiple replicas

## Local environment setup

These configurations are tested and run on an inf2.48xlarge with the Hugging Face Deep Learning AMI from the AWS Marketplace.

Copy the configurations down using

```shell
$ git clone https://github.com/huggingface/optimum-neuron.git
$ cd optimum-neuron/benchmark/text-generation-inference/
```


## Select model and configuration

Edit the `.env` file to select the model to use for the benchmark and its configuration.

The following instructions assume that you are testing a locally built image, so docker would have stored image neuronx-tgi:latest.

You can confirm this by running:

```shell
$ docker image ls
```

If you have not built it locally, you can download it and retag it using the following commands

```shell
$ docker pull ghcr.io/huggingface/neuronx-tgi:latest
$ docker tag ghcr.io/huggingface/neuronx-tgi:latest neuronx-tgi:latest
```
You should then see the single IMAGE ID with two different sets of tags:

```shell
$ docker image ls
REPOSITORY                        TAG       IMAGE ID       CREATED        SIZE
neuronx-tgi                       latest    f5ba57f8517b   12 hours ago   11.3GB
ghcr.io/huggingface/neuronx-tgi   latest    f5ba57f8517b   12 hours ago   11.3GB
```


Alternatively, you can edit the appropriate docker-compose.yaml to supply the fully path by changing ```neuronx-tgi:latest``` to ```ghcr.io/huggingface/neuronx-tgi:latest```

## Start the servers

If you are using a gated model, first export your credentials token.

```shell
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

Start the container servers with a single command:

```shell
$ docker compose -f llama3.1-8b/docker-compose.yaml --env-file llama3.1-8b/.env up
```

Note: you can edit the .env file to use a different model configuration

## Run the benchmark

### Install `guidellm`

```shell
$ pip install guidellm
```

### Launch benchmark run

The benchmark script takes the `model_id` and number of concurrent users as parameters.
The `model_id` must match the one corresponding to the selected `.env` file.

```shell
$ ./benchmark.sh "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

Note that the evaluated model **must** be an `Instruct` model.

At the end of the benchmark, the results will be saved in a `.json` file and a
summary will be displayed on the console.

To obtain a summary from the raw benchmark output files, use the following command:

```shell
$ python generate_csv.py --dir <path_to_result_files>
```

### Compiling the model

If you are trying to run a configuration or a model that is not available in the cache, you can compile the model before you run it, then load it locally.

See the [llama3-70b-trn1.32xlarge](llama3-70b-trn1.32xlarge) as an example.

It is best to compile the model with the software in the container you will be using to ensure all library versions match.

As an example, you can compile with the following command.  

**If you make changes, make sure your batch size, sequence length, and num_cores for compilation match the MAX_BATCH_SIZE, and MAX_TOTAL_TOKENS settings in the .env file and the HF_NUM_CORES setting in the docker-compose file.  
MAX_INPUT_LENGTH needs to be less than sequence_length/MAX_TOTAL_TOKENS.  The directory at the end of the compile command needs to match the MODEL_ID in the .env file.**  

```
docker run -p 8080:80 \
-v $(pwd):/data \
--device=/dev/neuron0 \
--device=/dev/neuron1 \
--device=/dev/neuron2 \
--device=/dev/neuron3 \
--device=/dev/neuron4 \
--device=/dev/neuron5 \
--device=/dev/neuron6 \
--device=/dev/neuron7 \
--device=/dev/neuron8 \
--device=/dev/neuron9 \
--device=/dev/neuron10 \
--device=/dev/neuron11 \
--device=/dev/neuron12 \
--device=/dev/neuron13 \
--device=/dev/neuron14 \
--device=/dev/neuron15 \
-ti \
--entrypoint "optimum-cli" neuronx-tgi:latest \
export neuron --model NousResearch/Meta-Llama-3-70B-Instruct \
--sequence_length 4096 \
--batch_size 4 \
--num_cores 32 \
/data/exportedmodel/
```
See the [Hugging Face documentation](https://huggingface.co/docs/optimum-neuron/en/guides/export_model#exporting-a-model-to-neuron-using-the-cli) for more information on compilation.  

Note that the .env file has a path for MODEL_ID to load the model from the /data directory.

Also, the docker-compose.yaml file includes an additional parameter to map the volume to the current working directory, as well as additional Neuron device mappings because trn1.32xlarge has 32 cores (16 devices).

Make sure you run the above command and the docker compose command from the same directory since it maps the /data directory to the current working directory.

For this example:
```
$ docker compose -f llama3-70b-trn1.32xlarge/docker-compose.yaml --env-file llama3-70b-trn1.32xlarge/.env up
```


