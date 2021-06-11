import os
import argparse
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to check if GPU is usable
def get_device():
    device = 'cpu'
    if torch.cuda.is_available(): #check if GPU device is available
        device = 'cuda' # assign the gpu to the device

    return device

# Function that read arguments from command line
def get_args():
    parser = argparse.ArgumentParser(description='Translate a TextCaps annotation (.npy) file from english to spanish.')
    parser.add_argument('--input_path', help='Path to the annotation file in english.')
    parser.add_argument('--output_path', help='Path to save the annotation file in spanish.')
    args = parser.parse_args()

    return args

# Function to load the npy file
def load_npy(path_to_captions):
    npy_file = np.load(path_to_captions, allow_pickle=True)
    print("File",path_to_captions,"succesfully loaded.")

    return npy_file

# Function to load the translation model
def load_en_es_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")

    return tokenizer, model.to(get_device())

# Función to translate a string
def translate_str(input, tokenizer, model, device):
    input = tokenizer.encode(input, return_tensors='pt')
    outputs = model.generate(input.to(device))
    out_str = [tokenizer.decode(t) for t in outputs][0][6:]

    return out_str

# Function to translate a list of strings
def translate_list_strs(input_list, tokenizer, model, device):
    translated_strs = []
    for str in input_list:
        translated_strs.append(translate_str(str, tokenizer, model, device))

    return translated_strs

# Function to translate a list of tokens
def translate_tokens(tokens, tokenizer, model, device):
    translated_tokens = ["<s>"]
    for token in tokens:
        if token != "<s>" and token != "</s>":
            translated_tokens.append(translate_str(token, tokenizer, model, device))
    translated_tokens.append('</str>')

    return translated_tokens

# Function to translate a list of lists of tokens
def translate_list_tokens(tokens_lists, tokenizer, model, device):
    translated_lists = []
    for tokens_list in tokens_lists:
        translated_lists.append(translate_tokens(tokens_list,tokenizer, model, device))

    return translated_lists

# Function to translate the str field of npy file
def translate_data(npy_data, tokenizer, model, device):
    for i, caption in enumerate(npy_data[1:]):
        if i % 1000 == 0:
            print(i,"traducciones realizadas.")

        caption['caption_str'] = translate_str(caption['caption_str'],
                                                tokenizer,
                                                model,
                                                device)

        caption['reference_strs'] = translate_list_strs(caption['reference_strs'],
                                                        tokenizer,
                                                        model,
                                                        device)

    print("All captions translated.")

    return npy_data

# Function to save the NPY output file in the specified path
def save_json(path_to_save, translated_data):
    # <name>.es.npy
    path = path_to_save
    print("Saving file to: ", path)
    np.save(path, translated_data)

# Main
def main():
    # Parseando argumentos
    args = get_args()

    # Cargando el modelo de traducción
    tokenizer, model = load_en_es_model()

    # device
    device = get_device()
    print("Running on:", device)

    # Cargando el dataset
    npy_data = load_npy(args.input_path)

    # Traduciendo el dataset
    translated_data = translate_data(npy_data, tokenizer, model, device)

    # Guardando el dataset traducido
    save_json(args.output_path, translated_data)

if __name__ == "__main__":
    main()
