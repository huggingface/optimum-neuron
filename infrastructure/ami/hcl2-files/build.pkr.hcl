build {
  name = "build-hf-dl-neuron"
  sources = [
    "source.amazon-ebs.ubuntu"
  ]
  provisioner "shell" {
    script = "scripts/setup-neuron.sh"
  }
  provisioner "shell" {
    script = "scripts/install-huggingface-libraries.sh"
  }
}