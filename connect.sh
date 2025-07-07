#!/bin/bash
HOSTNAME=${1:-ec2-23-23-72-196.compute-1.amazonaws.com}
echo "Connecting to $HOSTNAME"
ssh -i /Users/michaelbenayoun/Documents/Cles/michael-keypair.pem -i /Users/michaelbenayoun/Documents/Cles/michael-us-east-2.pem -i /Users/michaelbenayoun/Documents/Cles/michael-us-west-2.pem -i /Users/michaelbenayoun/Documents/Cles/michael-keypair-eu-north-1.pem -o StrictHostKeyChecking=no  -o ServerAliveInterval=60 -o ServerAliveCountMax=3 ubuntu@$HOSTNAME
