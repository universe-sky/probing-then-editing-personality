# Probing then Editing Response Personality of Large Language Models

Reproduction code for the paper “[Probing then Editing Response Personality of Large Language Models](https://arxiv.org/abs/2504.10227)”.

## Requirements

All dependencies are specified in `llama3_environment.yml`. To create and activate the conda environment:

```bash
# Create the environment (replace <env-name> with the name in your YAML)
conda env create -f llama3_environment.yml

# Activate the environment
conda activate <env-name>
```

## Datasets

All datasets used in this project are stored in the `datasets/` directory.

## Instructions

### Baseline

## Our work

### Stage 1: Probing Experiment

We prompt the LLM to answer every question in the training set under varied personality settings. During inference, we extract the hidden-layer activations and visualize them (e.g., via PCA or t-SNE) to assess whether distinct clusters emerge for different personalities. Finally, we train a classifier on these representations to quantify the separability of personality-conditioned states.

### Stage 2: Personality Editing Experiment

On the test set—where each prompt specifies an initial personality—we employ hook functions to intervene in the model’s hidden representations layer by layer. Specifically, we extract the activations at each layer, use the classifier trained in the probing experiment to compute targeted perturbations toward our desired personality, and re-inject the modified activations into the next layer. By iteratively applying these interventions across all layers, we compel the model to produce responses in the edited target personality, even when the original prompt called for a different one.

All of the above experiments can be run from the project root by executing:

```bash
python demo.py
```

## Performance Evaluation

We evaluate both the unperturbed and the perturbed models on the MMLU benchmark. From the project root, run:

```bash
python mmlu_chat.py
# For the base model, use:
python mmlu_base.py
```