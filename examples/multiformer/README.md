# Multiformer

Implementation of the paper "Multiformer: A Head-Configurable Transformer-Based Model for Direct Speech Translation"

## Abstract

*Transformer-based models have been achieving state-of-the-art results in several fields of Natural Language Processing. However, its direct application to speech tasks is not trivial. The nature of this sequences carries problems such as long sequence lengths and redundancy between adjacent tokens. Therefore, we believe that regular self-attention mechanism might not be well suited for it.*

*Different approaches have been proposed to overcome this problems, such as the use of efficient attention mechanisms. However, the use of this methods usually comes with a cost, which is a performance reduction caused by information loss. In this study, we present the Multiformer, a Transformer-based model which allows the use of different attention mechanisms on each head. By doing this, the model is able to bias the self-attention towards the extraction of more diverse token interactions, and the information loss is reduced. Finally, we perform an analysis of the head contributions, and we observe that those architectures where all heads relevance is uniformly distributed obtain better results. Our results show that mixing attention patterns along the different heads and layers outperforms our baseline by up to 0.7 BLEU.*

## Installation


Load the required versions of Singularity (alternative to Docker) and CUDA:

```bash
module load singularity/3.2.1 cuda/11.1

# Make sure to have installed singularity version 3.2.1 (https://sylabs.io/guides/3.2/user-guide/installation.html) and cuda version 11.1 (https://developer.nvidia.com/cuda-11.1.0-download-archive).
```

Set the environment variables:
```bash
export FAIRSEQ_ROOT=...             # where you'll clone our Fairseq fork
export CONTAINER=.../multiformer    # specify a path for the multiformer container
export MUSTC_ROOT=...               # where must-c dataset is located
export MKL_THREADING_LAYER=GNU      # To avoid errors with the container
```

Clone this Fairseq branch:
```bash
git clone -b multiformer https://github.com/mt-upc/fairseq-internal.git ${FAIRSEQ_ROOT}
```

Create the container:
```bash
singularity build --sandbox ${CONTAINER} docker://gegallego/repro-containers:base-pytorch1.9.0-cu11.1
```

Install the required packages into the container:
```bash
singularity run --userns --nv ${CONTAINER} \
pip install local-attention==1.4.3 performer-pytorch==1.1.3 \
pip install --editable ${FAIRSEQ_ROOT}
pip install pandas torchaudio soundfile sentencepiece
```

## Data Preparation

Preprocess the data as follows: :
```bash
# Generate TSV manifests, features, vocabulary
# and configuration for each language

singularity run --userns --nv \
--bind ${MUSTC_ROOT} ${CONTAINER} ${FAIRSEQ_ROOT} \ # Directories where the container needs to access
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr \
  --vocab-type unigram --vocab-size 5000
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type unigram --vocab-size 8000
```

## Train

**Pretraining**

Pretrain the model in ASR with:

```bash
singularity run --userns --nv \
--bind ${MUSTC_ROOT} ${ASR_SAVE_DIR} ${CONTAINER} \ # Directories where the container needs to access
fairseq-train ${MUSTC_ROOT}/en-${TGT} \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 32000 --fp16 --batch-size 256 \
  --max-update 50000 --task speech_to_text --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 --report-accuracy --arch ${MULTIFORMER_ARCH} --optimizer adam \
  --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 \
  --seed 1 --update-freq 10
```
Where `${TGT}` is the target lenguage, `${ASR_SAVE_DIR}` the path where the ASR checkpoints will be stored and `${MULTIFORMER_ARCH}` the Multiformer architecture to be trained.

**Training**

Train the model for ST with:

```bash
singularity run --userns --nv \
--bind ${S2T_ROOT} ${CONTAINER} ${ASR_SAVE_DIR} ${MUSTC_ROOT} \ # Directories where the container needs to access
fairseq-train ${MUSTC_ROOT}/en-${TGT} \
  --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${ST_SAVE_DIR} --num-workers 4 --max-tokens 32000 --max-update 50000 --batch-size 256 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch ${MULTIFORMER_ARCH} --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 10 \
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}/checkpoint_best.pt
```
With `${ST_SAVE_DIR}` as the path where the ST checkpoints will be stored.

## Evaluation

**Averaging Checkpoints**

To average the 7 checkpoints around the best one:

```bash
singularity run --userns --nv \
--bind ${ST_SAVE_DIR} ${CONTAINER} ${MUSTC_ROOT} \ # Directories where the container needs to access
python /home/usuaris/veu/gerard.muniesa/repositories/fairseq-multiformer/scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 7 --checkpoint-upper-bound=${3_AFTER_BEST} \
  --output ${ST_SAVE_DIR}/avg_7_around_best.pt
```
Where `${3_AFTER_BEST}` is number of of the best checkpoint plus 3.

**BLEU**

To calculate the BLEU of the resulting averaged checkpoint, use:

```bash
singularity run --userns --nv \
--bind ${ST_SAVE_DIR} ${OUTPUT_DIR} ${CONTAINER} ${MUSTC_ROOT} \ # Directories where the container needs to access
PYTHONIOENCODING=utf-8 fairseq-generate ${MUSTC_ROOT}/en-${TGT} \
  --config-yaml config_st.yaml --gen-subset tst-COMMON_st --task speech_to_text \
  --path ${ST_SAVE_DIR}/avg_7_around_best.pt --quiet \
  --max-tokens 50000 --beam 5 --scoring sacrebleu \
  --log-format json >> ${OUTPUT_DIR}
```

Where `${OUTPUT_DIR}` is the path of the .txt file where the BLEU score will be written.

## Heads Contribution Analysis

To perform the analysis of heads, use the [heads_contribution.ipynb](https://github.com/mt-upc/fairseq-internal/blob/multiformer/examples/multiformer/heads_contribution.ipynb) jupyter notebook.

![Heads Contribution Analysis](/media/Analisi_heads.png)