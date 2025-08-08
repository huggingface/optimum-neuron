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
  default     = "ami-0aea5bf206fcd4a09"
  description = "Base Image"
  type        = string
  /*
  To get latest value, run the following command:
  aws ec2 describe-images \
      --region us-east-1 \
      --owners amazon \
      --filters 'Name=name,Values=Deep Learning AMI Neuron ???????????????????????' 'Name=state,Values=available' \
      --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' \
      --output text
  */
}

variable "ssh_username" {
  default     = "ubuntu"
  description = "Username to connect to SSH with"
  type        = string
}

variable "optimum_version" {
  description = "Optimum Neuron version to install"
  type        = string
}

variable "transformers_version" {
  default     = "4.51.0"
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
