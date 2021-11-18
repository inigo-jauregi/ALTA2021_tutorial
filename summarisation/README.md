# Document Summarisation Model Training
________

## Introduction

In this section of the tutorial we show how quickly to train a
state-of-the-art summarisation model.

We provide the code to leverage pretrained BART (Lewis et al., 2019) and Longformer Encoder Decoder (Beltagy et al., 2020) 
models available in [Huggingface](https://huggingface.co/).

## Code structure

```python
src
  - __init__.py
  - summarisation_dataset.py
  - summarisation_lightning_model.py
requirements.txt
train_and_eval_notebook.ipynb
```

The `src` folder contains the code needed by the notebook to train a summarisation model with 
`pytorch_lightning` (`summarisation_lightning_model.py`), to load the train, dev and 
test sets (`summarisation_dataset.py`), and to calculate different evaluation metrics.

The `requirements.txt` file contains the python packages needed for this section of the tutorial.

`train_and_eval_notebook.ipynb` is the python notebook that we run in the tutorial.

## Notebook Tasks

The notebook is easy to follow, with multiple comments at each step to understand
what each block of code does.

The notebook is divided in two main tasks:

1. Train and evaluate a summarisation model
   1. The BART-base and LED-base models are used to speed up training
2. Compare the performance of both these models via a qualitative example


## Getting Started

The notebook can be run in Jupyter Notebooks or in Google Colab. The following 
section shows how to start using the notebook in Google Colab.

### Google Colab

1. Open [Google Colab](https://colab.research.google.com/?utm_source=scs-index)
2. Select GitHub as the source of notebook and paste this repository's URL (https://github.com/ijauregiCMCRC/ALTA2021_tutorial)
3. Two notebooks will be available. Select the **summarisation** notebook (summarisation/train_and_eval_notebook.ipynb)
4. That should open the notebook in Google Colab. Select the instance type (i.e. CPU, GPU, TPU) and follow the notebook instructions.

## References

Lewis, Mike, et al. "Bart: Denoising sequence-to-sequence pre-training for natural language 
generation, translation, and comprehension." arXiv preprint arXiv:1910.13461 (2019).

Beltagy, Iz, et al. "Longformer: The long-document transformer." arXiv:2004.05150 (2020).
