#!/bin/bash
set -e

# Configure Linux for Neuron repository updates
. /etc/os-release

sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages
sudo apt-get update -y

# Install git
sudo apt-get install git -y

# Remove preinstalled packages and Install Neuron Driver and Runtime
sudo apt-get remove aws-neuron-dkms -y
sudo apt-get remove aws-neuronx-dkms -y
sudo apt-get remove aws-neuronx-oci-hook -y
sudo apt-get remove aws-neuronx-runtime-lib -y
sudo apt-get remove aws-neuronx-collectives -y
sudo apt-get install aws-neuronx-dkms=2.* -y
sudo apt-get install aws-neuronx-oci-hook=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.** -y
sudo apt-get install aws-neuronx-collectives=2.* -y

# Install EFA Driver(only required for multiinstance training)
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
cat aws-efa-installer.key | gpg --fingerprint
wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig &&  gpg --verify ./aws-efa-installer-latest.tar.gz.sig

tar -xvf aws-efa-installer-latest.tar.gz
cd aws-efa-installer && sudo bash efa_installer.sh --yes
sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

# Remove pre-installed package and Install Neuron Tools
sudo apt-get remove aws-neuron-tools  -y
sudo apt-get remove aws-neuronx-tools  -y
sudo apt-get install aws-neuronx-tools=2.* -y

export PATH=/opt/aws/neuron/bin:$PATH

# Install Python venv and activate Python virtual environment to install
# Neuron pip packages.
sudo apt install python3.8-venv -y

cd /home/ubuntu

. "/etc/parallelcluster/cfnconfig"

set -e
version=${1:-"latest"}

if [[ $cfn_node_type == "HeadNode" ]]; then
  python3.8 -m venv aws_neuron_venv_pytorch
  source aws_neuron_venv_pytorch/bin/activate

  pip install -U pip

  if [ $version = "latest" ]; then
      REPO_URL="https://pip.repos.neuron.amazonaws.com"
  else
      if [ $version = "alpha" ]; then
          SECRET_ID=arn:aws:secretsmanager:us-east-1:125563815299:secret:HuggingFace-NQpWO0
      elif [ $version = "beta" ]; then
          SECRET_ID=arn:aws:secretsmanager:us-east-1:013207657731:secret:Jasnah-NuDPcl
      elif [ $version = "rc" ]; then
          SECRET_ID=arn:aws:secretsmanager:us-east-1:125563815299:secret:HuggingFace-NQpWO0
      else
          echo "Invalid SDK target"; exit 1
      fi
      REPO_SECRET=$(aws secretsmanager get-secret-value --secret-id $SECRET_ID --region us-east-1)
      REPO_PASSWORD=$(echo "$REPO_SECRET" | jq -r '.SecretString' | jq -r '.password')
      REPO_USER=HuggingFace
      REPO_URL="https://$REPO_USER:$REPO_PASSWORD@pip.repos.$version.neuron.annapurna.aws.a2z.com"
  fi

  echo $REPO_URL

  python -m pip install --force-reinstall --pre --extra-index-url $REPO_URL \
                        torch_neuronx neuronx-cc==2.* transformers-neuronx neuronx-distributed

  chown ubuntu:ubuntu -R aws_neuron_venv_pytorch
else
  DNS_SERVER=""
  grep Ubuntu /etc/issue &>/dev/null && DNS_SERVER=$(resolvectl dns | awk '{print $4}' | sort -r | head -1)
  IP="$(host $HOSTNAME $DNS_SERVER | tail -1 | awk '{print $4}')"
  DOMAIN=$(jq .cluster.dns_domain /etc/chef/dna.json | tr -d \")
  sudo sed -i "/$HOSTNAME/d" /etc/hosts
  sudo bash -c "echo '$IP $HOSTNAME.${DOMAIN::-1} $HOSTNAME' >> /etc/hosts"
fi
