# Tackling Low-Resource Sign Language Translation: UPC at WMT-SLT 22

This repository contains the implementation for the WMT-SLT22 UPC team submission. The paper will be linked and available soon.

## First steps

Clone this repository, create the conda environment and install Fairseq:

```bash
git clone -b wmt-slt22 git@github.com:mt-upc/fairseq.git
cd fairseq

conda env create -f ./examples/sign_language/environment.yml
conda activate sign-language

pip install --editable .
```

The execution of scripts is managed with [Task](https://taskfile.dev/). Please follow the [installation instructions](https://taskfile.dev/installation/) in the official documentation.

We recommend using the following
```bash
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b path-to-env/sign-language/bin
```

## Pre-processing steps
Fnid the different tasks related to the pre-processing steps defined inside `Taskfile.yml`. The nomenclature of the tasks is `(challenge):dataset:(partition):task` where tasks are defined as:
- download: downloads the dataset
- extract: decompresses the dataset
- convert2video: from frames to video, necessary for the phoenix dataset.
- videos_to_25fps: converts videos from any fps to 25 fps. Necessary for FocusNews.
- extract_mediapipe: extracts mediapipe following the pose-format library.
- generate_tsv: generates the tsv files for the dataset necessary in Fairseq.
- train_sentencepipece: trains the sentencepiece model with the provided dataset data.

Tip: you can create an .env file that contains all local variables such as paths, WandB project, etc.

After the environment is set up, you can run the following command to run the different tasks:
```bash
task (challenge):dataset:(partition):task
```

## Training
We provide the script to `train.sh`. The experiment launched should have a corresponding `.yaml` file, you can find the different `.yaml` used in the configs folder. The script creates a folder with the name of the experiment and saves the checkpoints, the logs and the wandb files.

## Test
Similarly to the pre-processing steps, we have created a task to generate the predictions. The task are called `generate` and `generate_no_target`. 

## Citations
- Some scripts from this repository use the GNU Parallel software.
  > Tange, Ole. (2022). GNU Parallel 20220722 ('Roe vs Wade'). Zenodo. https://doi.org/10.5281/zenodo.6891516
- If you use this code, please cite the following paper:
  > @inproceedings{EMNLP WMT-SLT 2022,
  > author = {Laia Tarrés, Gerard Ion Gállego, Xavier Giró-i-Nieto, Jordi Torres},
  > title = {Tackling Low-Resource Sign Language Translation: UPC at WMT-SLT 22},
  > booktitle = {},
  > year = {2022}
  > }
