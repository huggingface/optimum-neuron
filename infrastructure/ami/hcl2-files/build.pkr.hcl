build {
  name = "build-hf-dl-neuron"
  sources = [
    "source.amazon-ebs.ubuntu"
  ]
  provisioner "shell" {
    script = "scripts/validate-neuron.sh"
  }
  provisioner "shell" {
    script = "scripts/install-huggingface-libraries.sh"
    environment_vars = [
      "TRANSFORMERS_VERSION=${var.transformers_version}",
      "OPTIMUM_NEURON_TAG=${var.optimum_neuron_tag}",
    ]
  }
  provisioner "shell" {
    inline = ["echo 'source /opt/aws_neuron_venv_pytorch/bin/activate' >> /home/ubuntu/.bashrc"]
  }
}