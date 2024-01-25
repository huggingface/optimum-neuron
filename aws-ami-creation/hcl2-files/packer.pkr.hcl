packer {
  required_plugins {
    docker = {
      version = ">= 1.2.8"
      source  = "github.com/hashicorp/amazon"
    }
  }
}