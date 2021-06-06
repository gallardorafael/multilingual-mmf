# Multilingual MMF


## Instructions
### Multilingual M4C Captioner:
Train on TextCaps dataset:

``
MMF_USER_DIR="projects/ml_m4c_captioner/" mmf_run datasets=textcaps \
    model=ml_m4c_captioner \
    config=projects/ml_m4c_captioner/experiments/en_m4c_captioner/textcaps/defaults.yaml \
    env.save_dir=./projects/ml_m4c_captioner/pretrained/en_m4c_captioner/defaults \
    run_type=train_val \
    training.num_workers=0
``

### M4C Captioner (with FastText vectors):
Train on TextCaps dataset:

``
mmf_run datasets=textcaps \
    model=m4c_captioner \
    config=projects/m4c_captioner/configs/m4c_captioner/textcaps/defaults.yaml \
    env.save_dir=./pretrained/m4c_captioner/defaults \
    run_type=train_val
    training.num_workers=0 \
``

Generate prediction JSON files for the **TextCaps validation set** (custom trained model):

``
mmf_predict datasets=textcaps \
    model=m4c_captioner \
    config=projects/m4c_captioner/configs/m4c_captioner/textcaps/defaults.yaml \
    env.save_dir=./save/m4c_captioner/defaults \
    run_type=val \
    checkpoint.resume = True \
    checkpoint.resume_best = True \
``

Generate prediction JSON files for the **TextCaps test set** (custom trained model):

``
mmf_predict datasets=textcaps \
    model=m4c_captioner \
    config=projects/m4c_captioner/configs/m4c_captioner/textcaps/defaults.yaml \
    env.save_dir=./save/m4c_captioner/defaults \
    run_type=test \
    checkpoint.resume = True \
    checkpoint.resume_best = True \
``

Generate prediction JSON files for the **TextCaps validation set** (zoo model):

``
mmf_predict datasets=textcaps \
    model=m4c_captioner \
    config=projects/m4c_captioner/configs/m4c_captioner/textcaps/defaults.yaml \
    env.save_dir=./save/m4c_captioner/defaults \
    run_type=val \
    checkpoint.resume_zoo=m4c_captioner.textcaps.defaults
``

Generate prediction JSON files for the **TextCaps test set** (zoo model):

``
mmf_predict datasets=textcaps \
    model=m4c_captioner \
    config=projects/m4c_captioner/configs/m4c_captioner/textcaps/defaults.yaml \
    env.save_dir=./save/m4c_captioner/defaults \
    run_type=test \
    checkpoint.resume_zoo=m4c_captioner.textcaps.defaults
``

Evaluate generated JSON predictions (val or test):

``
export MMF_DATA_DIR=~/.cache/torch/mmf/data

python projects/m4c_captioner/scripts/textcaps_eval.py \
    --set val \
    --annotation_file ${MMF_DATA_DIR}/datasets/textcaps/defaults/annotations/imdb_val.npy \
    --pred_file YOUR_VAL_PREDICTION_FILE
``

### Botom-Up Top-Down Attention (BUTD)
Train on TextCaps dataset:

``
mmf_run config=projects/butd/configs/textcaps/defaults.yaml \
    model=butd \
    dataset=textcaps \
    run_type=train \
    env.save_dir=./pretrained/butd/defaults \
    training.num_workers=0 \
``

Evaluate on TextCaps dataset (greedy decoding):

``
mmf_run config=projects/butd/configs/textcaps/defaults.yaml \
    model=butd \
    dataset=textcaps \
    run_type=val \
    checkpoint.resume_file=./pretrained/butd/defaults/<model.pth> \
``

Evaluate on TextCaps dataset (beam search decoding):

``
mmf_run config=projects/butd/configs/textcaps/beam_search.yaml \
    model=butd \
    dataset=textcaps \
    run_type=val \
    checkpoint.resume_file=./pretrained/butd/defaults/<model.pth> \
``

Test on TextCaps dataset (beam search decoding):

``
mmf_run config=projects/butd/configs/textcaps/beam_search.yaml \
    model=butd \
    dataset=textcaps \
    run_type=test \
    checkpoint.resume_file=./pretrained/butd/defaults/<model.pth> \
``

Generate predictions with BUTD model, pretrained on COCO:

``
mmf_predict config=projects/m4c_captioner/configs/butd/textcaps/beam_search.yaml \
    model=butd \
    dataset=textcaps \
    run_type=val \
    checkpoint.resume_zoo=<need revision>
``
