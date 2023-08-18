# Text-generation-inference docker image

## Build image

```
docker build --rm -f Dockerfile  -t neuronx-tgi .
```

## Run service

```
docker run --device=/dev/neuron0 neuronx-tgi --model-id gpt2
```
