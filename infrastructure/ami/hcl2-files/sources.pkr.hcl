source "amazon-ebs" "ubuntu" {
  ami_name      = "huggingface-neuron-dl-ami-ubuntu-20.04_{{timestamp}}"
  instance_type = var.instance_type
  region        = var.region
  source_ami    = var.source_ami
  ssh_username  = var.ssh_username
  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    // encrypted             = true
    volume_size           = 512
    volume_type           = "gp2"
    delete_on_termination = true
  }
  ami_users   = var.ami_users
  ami_regions = var.ami_regions
}