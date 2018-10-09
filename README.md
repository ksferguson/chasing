# Chasing the Flagellum
Rare Words for Text Generation

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

```bash
#assumes Anaconda installed with 'base' environment
. activate base
conda create --name chasing python=3.6
. activate chasing
conda install pytorch=0.1.12 -c soumith
conda install cython
pip install streamlit
```



#### Project Code
##### Note: Project includes a copy of https://github.com/salesforce/awd-lstm-lm, pulled with Tag: PyTorch==0.1.12

```bash
# Clone project
git clone https://github.com/ksferguson/chasing
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

## Train Model
### Initial training
```bash
python main.py --batch_size 20 --data data/penn --dropouti 0.4 --seed 28 --epoch 300 --save PTB.pt
```
### Second Stage training
```bash
python finetune.py --batch_size 20 --data data/penn --dropouti 0.4 --seed 28 --epoch 300 --save PTB.pt
```

### Generate
```bash

```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
