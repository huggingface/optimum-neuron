build {
  name = "build-hf-dl-neuron"
  sources = [
    "source.amazon-ebs.ubuntu"
  ]
  provisioner "shell" {
    script = "scripts/validate_neuron_exist.sh"
  }
  provisioner "shell" {
    script = "scripts/install_huggingface_libraries.sh"
  }
}