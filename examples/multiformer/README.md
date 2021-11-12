# Multiformer

Set the environment variables:
```bash
export FAIRSEQ_ROOT=...      # where you'll clone our Fairseq fork
```

Clone this Fairseq branch:
```bash
git clone -b s2t-local https://github.com/mt-upc/fairseq.git ${FAIRSEQ_ROOT}
```

Create a conda environment using the environment.yml file and activate it:
```bash
conda env create -f ${FAIRSEQ_ROOT}/examples/multiformer/environment.yml && \
conda activate multiformer
```

Install Fairseq:
```bash
pip install --editable ${FAIRSEQ_ROOT}
```
