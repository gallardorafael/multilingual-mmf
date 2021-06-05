# Towards Language-agnostic Architectures for Image Captioning with Reading Comprehension


## Instructions ro run for each model
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
Val on TextCaps dataset with COCO pretrained model(beam search decoding):
``
mmf_run config=projects/butd/configs/textcaps/beam_search.yaml \
    model=butd \
    dataset=textcaps \
    run_type=val \
    checkpoint.resume_zoo=butd
``
