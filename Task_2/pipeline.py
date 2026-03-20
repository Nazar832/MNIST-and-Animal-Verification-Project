import argparse
import torch
from CV_model.inference import load_and_predict
from NER_model.inference import extract_animal


def run_pipeline(text, image_path, ner_path, img_model_path):
    extracted_animal = extract_animal(text, model_path=ner_path)
    
    if not extracted_animal:
        print("NER: No animal mentioned in text.")
        return False
    
    extracted_animal = extracted_animal.lower()

    predicted_animal, confidence = load_and_predict(image_path, model_path=img_model_path)
    predicted_animal = predicted_animal.lower()

    is_confident = confidence >= 0.5
    
    print(f"Text says: {extracted_animal} | Image shows: {predicted_animal if is_confident else 'Uncertain'}")

    match = (extracted_animal == predicted_animal) and is_confident
    return bool(match)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ner_model", type=str, default="./NER_model/animal_ner_model")
    parser.add_argument("--img_model", type=str, default="./CV_model/animal_classifier.h5")

    args = parser.parse_args()

    result = run_pipeline(args.text, args.image, args.ner_model, args.img_model)
    print(f"Result: {result}")