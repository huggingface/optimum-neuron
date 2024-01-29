variable "region" {
  description = "The AWS region"
  type        = string
}

variable "instance_type" {
  default     = "trn1.2xlarge"
  description = "EC2 machine type for building AMI"
  type        = string
}

variable "source_ami" {
  default     = "ami-0fbea04d7389bcd4e"
  description = "Base Image"
  type        = string
}

variable "ssh_username" {
  default     = "ubuntu"
  description = "Username to connect to SSH with"
  type        = string
}

variable "optimum_neuron_tag" {
  default     = "v0.0.17"
  description = "Optimum Neuron version to install"
  type        = string
}

variable "transformers_version" {
  default     = "4.36.2"
  description = "Transformers version to install"
  type        = string
}

variable "ami_users" {
  default     = ["754289655784", "558105141721"]
  description = "AWS accounts to share AMI with"
  type        = list(string)
}

variable "ami_regions" {
  default     = ["eu-west-1"]
  description = "AWS regions to share AMI with"
  type        = list(string)
}