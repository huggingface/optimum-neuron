variable "region" {
  default     = "us-east-1"
  description = "The AWS region you're using"
  type        = string
}

variable "instance_type" {
  default     = "trn1.2xlarge"
  description = "EC2 machine type for building AMI"
  type        = string
}

variable "source_ami" {
  default     = "ami-077399be2d0ae7e0c"
  description = "Base Image"
  type        = string
}

variable "ssh_username" {
  default     = "ubuntu"
  description = "Username to connect to SSH with"
  type        = string
}