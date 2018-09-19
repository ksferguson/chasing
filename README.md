# Chasing the Tail (of the Word Frequency Distribution)
NLP Transfer Learning with Rare Words

## Project Structure:
- **src** : Source code for production within structured directory
- **tests** : Source code for testing in an easy to find location
- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)
- **data** : Include example a small amount of data in the Github repository so tests can be run to validate installation
- **build** : Include scripts that automate building of a standalone environment
- **static** : Any images or content to include in the README or web framework if part of the pipeline

## Requisites
- Packages and software needed to build the environment

#### Base environment
- (Easy Way) EC2 Instance on AWS: Deep Learning AMI (Ubuntu) Version 13.0 (ami-00499ff523cc859e6), includes prebuilt env 'pytorch_p36' (i.e. source activate pytorch_p36):
  - Anaconda
  - Python 3.6
  - PyTorch 0.4.1

- (Alternative) Build an env:
  - Install Anaconda and setup environment
  - Install Python 3.6
  - Install PyTorch 0.4.1

#### Project Code
##### Note: Project includes a copy of https://github.com/salesforce/awd-lstm-lm

```bash
# Clone project
git clone https://github.com/ksferguson/chasing-the-tail
```




## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
