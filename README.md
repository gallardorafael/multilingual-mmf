# Multilingual MMF


## Instructions
### Usage of ml_m4c_captioner with original TextBert:
Train on TextCaps dataset with original TextBert (en_m4c_captioner):

``
MMF_USER_DIR="projects/ml_m4c_captioner/" mmf_run datasets=textcaps \
  model=ml_m4c_captioner \
  config=projects/ml_m4c_captioner/experiments/en_m4c_captioner/textcaps/defaults.yaml \
  env.save_dir=./projects/ml_m4c_captioner/pretrained/en_m4c_captioner/defaults \
  run_type=train_val \
  training.num_workers=0 \
  training.batch_size=16
``

Generate JSON on TextCaps validation (en_m4c_captioner):

``
MMF_USER_DIR="projects/ml_m4c_captioner/" mmf_predict datasets=textcaps \
  model=ml_m4c_captioner \
  config=projects/ml_m4c_captioner/experiments/en_m4c_captioner/textcaps/defaults.yaml \
  env.save_dir=./projects/ml_m4c_captioner/pretrained/en_m4c_captioner_/defaults \
  run_type=val \
  checkpoint.resume=True \
  checkpoint.resume_best=True \
  training.num_workers=0 \
  training.batch_size=16
``

Evaluate generated JSON predictions (val or test):

``
export MMF_DATA_DIR=~/.cache/torch/mmf/data

python projects/m4c_captioner/scripts/textcaps_eval.py \
--set val \
--annotation_file ${MMF_DATA_DIR}/datasets/textcaps/defaults/annotations/imdb_val.npy \
--pred_file PRED_FILE_PATH
``

### Usage of ml_m4c_captioner with multilingual TextBert (with English FastText vectors):
Train on TextCaps dataset with multilingual TextBert (en_m4c_captioner):

``
MMF_USER_DIR="projects/ml_m4c_captioner/" mmf_run datasets=textcaps \
  model=ml_m4c_captioner \
  config=projects/ml_m4c_captioner/experiments/en_m4c_captioner/textcaps/text_bert_multilingual.yaml \
  env.save_dir=./projects/ml_m4c_captioner/pretrained/en_m4c_captioner/text_bert_multilingual \
  run_type=train_val \
  training.num_workers=0 \
  training.batch_size=16
``

### Usage of ml_m4c_captioner with multilingual TextBert (with Spanish FastText vectors):
Train on TextCaps dataset (es_m4c_captioner):

``
MMF_USER_DIR="projects/ml_m4c_captioner/" mmf_run datasets=textcaps \
    model=ml_m4c_captioner \
    config=projects/ml_m4c_captioner/experiments/es_m4c_captioner/textcaps/defaults.yaml \
    env.save_dir=./projects/ml_m4c_captioner/pretrained/es_m4c_captioner/defaults \
    run_type=train_val \
    training.num_workers=2 \
    training.batch_size=16
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
