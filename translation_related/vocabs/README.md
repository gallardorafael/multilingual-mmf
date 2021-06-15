# Usage
### To get a vocab file from an annotation file

``
python translation_related/vocabs/extract_vocab.py \
--input_path translation_related/annotations/es_imdb_train.npy \
--output_path translation_related/vocabs/textcaps_es_freq5.txt \
--min_freq 10
``
