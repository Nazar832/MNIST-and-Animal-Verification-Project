from transformers import pipeline
import argparse


def extract_animal(text, model_path="./animal_ner_model"):
    ner_pipe = pipeline("ner", model=model_path, tokenizer=model_path)
    results = ner_pipe(text)

    found_animals = []
    for res in results:
        if res['entity'] == 'LABEL_1':
            word = res['word'].replace("##", "")
            found_animals.append(word)

    return found_animals[0] if found_animals else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    animal = extract_animal(args.text)
    print(f"Extracted animal: {animal}")