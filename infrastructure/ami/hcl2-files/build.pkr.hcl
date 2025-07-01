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
      "OPTIMUM_VERSION=${var.optimum_version}",
    ]
  }
  provisioner "shell" {
    inline = [
      "echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' | sudo tee -a /home/ubuntu/.bashrc",
      "echo 'source /opt/aws_neuronx_venv_pytorch_2_7/bin/activate' | sudo tee -a /home/ubuntu/.bashrc"
    ]
  }
  provisioner "file" {
    source      = "scripts/welcome-msg.sh"
    destination = "/tmp/99-custom-message"
  }
  provisioner "shell" {
    inline = [
      "sudo mv /tmp/99-custom-message /etc/update-motd.d/",
      "sudo chmod +x /etc/update-motd.d/99-custom-message",
    ]
  }
}
