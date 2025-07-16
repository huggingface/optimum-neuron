source "amazon-ebs" "ubuntu" {
  ami_name      = "huggingface-neuron-{{isotime \"2006-01-02T15-04-05Z\"}}"
  instance_type = var.instance_type
  region        = var.region
  source_ami    = var.source_ami
  ssh_username  = var.ssh_username
  ssh_clear_authorized_keys = true
  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 512
    volume_type           = "gp2"
    delete_on_termination = true
  }
  ami_users   = var.ami_users
  ami_regions = var.ami_regions
}
