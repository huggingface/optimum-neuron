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
    inline = ["echo 'source /opt/aws_neuron_venv_pytorch/bin/activate' >> /home/ubuntu/.bashrc"]
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