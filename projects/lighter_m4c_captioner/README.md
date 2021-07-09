# Lighter M4C-Captioner


## Instructions
### Usage of lighter_m4c_captioner (with GloVe context processor):
Train on **TextCaps training set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_run datasets=textcaps \
  model=lighter_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/defaults.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/defaults \
  run_type=train_val
``

Generate prediction JSON files for the **TextCaps validation set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_predict datasets=textcaps \
  model=lighter_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/context_glove.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/context_glove \
  run_type=val \
  checkpoint.resume=True \
  checkpoint.resume_best=True
``

Generate prediction JSON files for the **TextCaps test set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_predict datasets=textcaps \
  model=lighter_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/context_glove.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/context_glove \
  run_type=test \
  checkpoint.resume=True \
  checkpoint.resume_best=True
``

Evaluate generated JSON predictions (val or test):

``
export MMF_DATA_DIR=~/.cache/torch/mmf/data
``

``
export MMF_DATA_DIR=~/.cache/torch/mmf/data
python projects/m4c_captioner/scripts/textcaps_eval.py \
--set val \
--annotation_file ${MMF_DATA_DIR}/datasets/textcaps/defaults/annotations/imdb_val.npy \
--pred_file PRED_FILE_PATH
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
    checkpoint.resume=True \
    checkpoint.resume_best=True
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
    checkpoint.resume_zoo=m4c_captioner.textcaps.defaults \
    training.batch_size=8 \
    training.num_workers=0
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
