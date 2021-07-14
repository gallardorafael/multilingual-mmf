# Lighter M4C-Captioner


## Instructions
### Usage of M4C Captioner (with GloVe context processor and BERT base):
Train on **TextCaps training set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_run datasets=textcaps \
  model=glove_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/glove.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/glove \
  run_type=train_val
``

Generate prediction JSON files for the **TextCaps validation set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_predict datasets=textcaps \
  model=glove_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/glove.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/glove \
  run_type=val \
  checkpoint.resume=True \
  checkpoint.resume_best=True
``

Generate prediction JSON files for the **TextCaps test set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_predict datasets=textcaps \
  model=glove_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/glove.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/glove \
  run_type=test \
  checkpoint.resume=True \
  checkpoint.resume_best=True
``

Evaluate generated JSON predictions (val):

``
export MMF_DATA_DIR=~/.cache/torch/mmf/data
python projects/m4c_captioner/scripts/textcaps_eval.py \
--set val \
--annotation_file ${MMF_DATA_DIR}/datasets/textcaps/defaults/annotations/imdb_val.npy \
--pred_file PRED_FILE_PATH
``

### Usage of Lighter M4C Captioner (with FastText context processor and DistilBERT base):
Train on **TextCaps training set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_run datasets=textcaps \
  model=lighter_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/lm4c.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/lm4c \
  run_type=train_val
``

Generate prediction JSON files for the **TextCaps validation set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_predict datasets=textcaps \
  model=lighter_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/lm4c.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/lm4c \
  run_type=val \
  checkpoint.resume=True \
  checkpoint.resume_best=True
``

Generate prediction JSON files for the **TextCaps test set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_predict datasets=textcaps \
  model=lighter_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/lm4c.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/lm4c \
  run_type=test \
  checkpoint.resume=True \
  checkpoint.resume_best=True
``

### Usage of Lighter M4C Captioner (with GloVe context processor and DistilBERT base):
Train on **TextCaps training set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_run datasets=textcaps \
  model=lighter_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/lm4c_glove.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/lm4c_glove \
  run_type=train_val
``

Generate prediction JSON files for the **TextCaps validation set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_predict datasets=textcaps \
  model=lighter_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/lm4c_glove.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/lm4c_glove \
  run_type=val \
  checkpoint.resume=True \
  checkpoint.resume_best=True
``

Generate prediction JSON files for the **TextCaps test set**:

``
MMF_USER_DIR="projects/lighter_m4c_captioner/" mmf_predict datasets=textcaps \
  model=lighter_m4c_captioner \
  config=projects/lighter_m4c_captioner/experiments/textcaps/lm4c_glove.yaml \
  env.save_dir=./projects/lighter_m4c_captioner/pretrained/lm4c_glove \
  run_type=test \
  checkpoint.resume=True \
  checkpoint.resume_best=True
``
