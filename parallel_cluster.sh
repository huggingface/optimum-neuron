#!/bin/bash
HOSTNAME=${1:-ec2-34-237-66-4.compute-1.amazonaws.com}
echo "Connecting to $HOSTNAME"
ssh -i /Users/michaelbenayoun/Documents/Cles/michael-keypair.pem -o StrictHostKeyChecking=no  ubuntu@$HOSTNAME
