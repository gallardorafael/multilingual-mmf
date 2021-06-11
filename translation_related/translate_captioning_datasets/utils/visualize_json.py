# Script to pretty print a JSON file.

import json
import argparse
import codecs

# Función que lee y parsea los argumentos
def get_args():
    parser = argparse.ArgumentParser(description='Script to pretty print a JSON file')
    parser.add_argument('--input_path', help='Path to the input file.')
    args = parser.parse_args()

    return args

# Función que guarda un diccionario (como JSON) en la ruta especificada
def save_json(path_to_save, json_data):
    path = path_to_save
    print("Guardando el archivo en: ", path)
    with open(path, 'w', encoding='utf8') as f:
        json.dump(json_data, f, ensure_ascii=False)

# Main
def main():
    # Parsing argumentos
    args = get_args()

    with open(args.input_path) as f:
        data = json.load(f)

        print(json.dumps(data, indent = 4, sort_keys=True))

        if input("Do you want to save the file with UTF-8 encoding?") == 'Y':
            save_json(args.input_path+".new", data)

if __name__ == "__main__":
    main()
