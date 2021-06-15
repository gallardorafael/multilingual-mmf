# Datasets translation and related files

Note: The translation script works with PyTorch<1.8 due to a conflict in Transformers library (File: modeling_bart.py, Class: SinusoidalPositionalEmbedding)

### To translate a .npy annotation file of TextCaps to a target language:

``
python translate_captioning_datasets/translate_texcaps.py \
--input_path PATH_TO_THE_NPY_FILE
--output_path PATH_TO_SAVE_THE_TRANSLATION
``
