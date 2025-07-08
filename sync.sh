#!/bin/bash
HOSTNAME=${1:-ec2-23-23-72-196.compute-1.amazonaws.com}
echo "Sending to $HOSTNAME"
rsync -avzr -e 'ssh -i /Users/michaelbenayoun/Documents/Cles/michael-keypair.pem -i /Users/michaelbenayoun/Documents/Cles/michael-us-east-2.pem -i /Users/michaelbenayoun/Documents/Cles/michael-us-west-2.pem -i /Users/michaelbenayoun/Documents/Cles/michael-keypair-eu-north-1.pem -o StrictHostKeyChecking=no' ~/projects/optimum-neuron  ubuntu@$HOSTNAME:/home/ubuntu
