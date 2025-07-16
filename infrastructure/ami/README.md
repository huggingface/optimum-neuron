# Building AMI with Packer

This directory contains the files for building AMI using [Packer](https://github.com/hashicorp/packer) that is later published as a AWS Marketplace asset.


## Folder Structure

- [hcl2-files](./hcl2-files/) - Includes different files which are used by a Packer pipeline to build an AMI. The files are:
  - [build.pkr.hcl](./hcl2-files/build.pkr.hcl): contains the [build](https://developer.hashicorp.com/packer/docs/templates/hcl_templates/blocks/build) block, defining the builders to start, provisioning them using [provisioner](https://developer.hashicorp.com/packer/docs/templates/hcl_templates/blocks/build/provisioner), and specifying actions to take with the built artifacts using `post-process`.
  - [variables.pkr.hcl](./hcl2-files/variables.pkr.hcl): contains the [variables](https://developer.hashicorp.com/packer/docs/templates/hcl_templates/blocks/variable) block, defining variables within your Packer configuration.
  - [sources.pkr.hcl](./hcl2-files/sources.pkr.hcl): contains the [source](https://developer.hashicorp.com/packer/docs/templates/hcl_templates/blocks/source) block, defining reusable builder configuration blocks.
  - [packer.pkr.hcl](./hcl2-files/packer.pkr.hcl): contains the [packer](https://developer.hashicorp.com/packer/docs/templates/hcl_templates/blocks/packer) block, used to configure some behaviors of Packer itself, such as the minimum required Packer version needed to apply to your configuration.
- [scripts](./scripts): contains scripts used by [provisioner](https://developer.hashicorp.com/packer/docs/templates/hcl_templates/blocks/build/provisioner) for installing additonal packages/softwares.


### Prerequisites
 - [Packer](https://developer.hashicorp.com/packer/docs/intro): Packer is an open source tool for creating identical machine images for multiple platforms from a single source configuration.

 - AWS Credentials: You need to have AWS credentials configured on your machine. You can configure AWS credentials using [AWS CLI](https://github.com/aws/aws-cli) or by setting environment variables.

 #### Install Packer on Ubuntu/Debian
 ```bash
 curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
 sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
 sudo apt-get update && sudo apt-get install packer
 ```

You can also install Packer for other OS from [here](https://developer.hashicorp.com/packer/tutorials/docker-get-started/get-started-install-cli).

#### Configure AWS Credentials

Using Environment Variables:
```bash
export AWS_ACCESS_KEY_ID=<access_key>
export AWS_SECRET_ACCESS_KEY=<secret_key>
```

Using AWS CLI:
```bash
aws configure sso
```

There are other ways to configure AWS credentials. You can read more about it [here](https://github.com/aws/aws-cli?tab=readme-ov-file#configuration).

### Build AMI

#### Format Packer blocks
You can format your HCL2 files locally. This command will update your files in place.

Format a single file:
```bash
packer fmt build.pkr.hcl
```

Format all files in a directory:
```bash
packer fmt ./hcl2-files
```

#### Validate Packer blocks
You can validate the syntax and configuration of your files locally. This command will return a zero exit status on success, and a non-zero exit status on failure.

```bash
packer validate -var 'region=us-west-2' -var 'optimum_version=v0.0.17' ./hcl2-files
```

#### Run Packer build
You can run Packer locally. This command will build the AMI and upload it to AWS.

You need to set variables with no default values using `-var` flag. For example:
```bash
packer build -var 'region=us-west-2' -var 'optimum_version=v0.0.17' ./hcl2-files
```

To trigger a github action workflow manually, you can use GitHub CLI:
```bash
gh workflow run build-ami.yml -f tag=<tag>
```
