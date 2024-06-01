#!/bin/bash -v
# This is made to be run on the Hugging Face DLAMI on an inferentia/trainium system

# at the end of this script, run 
# python generate_csv.py


# change the modelname on the next line.
set modelname=$1
echo on
#set for your environment if not already set
#export LLMPerf=/home/ubuntu/llmperf

for concurrency in 1 2 4 8 16 32 64 128 256 512
    do

    ./benchmark.sh $modelname $concurrency


done
