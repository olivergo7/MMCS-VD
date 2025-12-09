# MMCS-VD: LLM-Enhanced Multimodal Feature Fusion with Meta-Learning for Code Vulnerability Detection

## Overview
This repository contains the code for the paper **"Enhancing Vulnerability Detection via Multi-modal Fusion and Meta-Learning"**. 

## Prerequisites
To run this project, you need to install the following dependencies:
1. **tree-sitter**: A library to parse the source code and generate ASTs.
2. **transformers**: The HuggingFace `transformers` library to use pre-trained models such as `CodeBERT`.
3. **openai**: Required for generating code comments using GPT-4 (or GPT-4 Mini).
You can install them using pip:
```bash
pip install tree-sitter transformers openai
```
If you want to reproduce the program, please follow these steps:

## Code Simplification
The order of code simplification is as follows: move to data_processing dir ,Simplification dir, run gpt4.py

## Move to 'slicing' dir.
To reproduce the slicing, you must download 'Joern'; we recommend version 1.0 for "Joern". We have placed the specific steps for reproducing in the 'slicing' directory.

## Move to 'processing' dir.
The order of executing files is as follows:

step 1: run input.py

step 2: run extract.py

step 3: run data.py

step 4: run script.py

For the rest files, the remove file.py is used to remove specific files, and word2vec.py is used to train word-to-vector models. You can also choose not to train the word vector model.

### Running the Models
Once the data is preprocessed, you can train and test the models for different datasets.
To train and test on Devign, Reveal, and Big-Vul, simply run the following scripts:

Execute the main.py file.

Note that the model here requires downloading CodeBERT-base. Please refer to this Huggingface url-link.
