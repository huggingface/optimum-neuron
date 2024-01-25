variable "region" {
  default     = "us-west-2"
  description = "The AWS region you're using"
  type        = string
}

variable "instance_type" {
  default     = "trn1.2xlarge"
  description = "EC2 machine type for building AMI"
  type        = string
}

variable "source_ami" {
  default     = "ami-0328507805a665ec1"
  description = "Base Image"
  type        = string
}

variable "ssh_username" {
  default     = "ubuntu"
  description = "Username to connect to SSH with"
  type        = string
}