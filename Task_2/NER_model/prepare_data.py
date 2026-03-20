import json
import random
from faker import Faker

with open('../class_names.json', 'r') as f:
    animals = json.load(f)
fake = Faker()


def generate_ner_dataset(num_samples=500):
    dataset = []

    for _ in range(num_samples):
        has_animal = random.random() > 0.2

        if has_animal:
            animal = random.choice(animals)
    
            sentence_type = random.choice(['template', 'fake_context'])

            if sentence_type == 'template':
                templates = [
                    f"There is a {animal} in this picture.",
                    f"I can see a {animal} over there.",
                    f"Look, a {animal}!",
                    f"The {animal} looks very hungry.",
                    f"Is that a {animal} on the photo?",
                    f"A beautiful {animal} is standing near the tree."
                ]
                text = random.choice(templates)
            else:
                context = fake.sentence(nb_words=random.randint(5, 12))
                tokens = context.split()
                insert_idx = random.randint(0, len(tokens) - 1)
                tokens[insert_idx] = animal
                text = " ".join(tokens)
        else:
            text = fake.sentence(nb_words=8)

        tokens = text.replace(".", "").replace("!", "").replace("?", "").split()
        ner_tags = []
        for token in tokens:
            if token in animals:
                ner_tags.append(1)  # B-ANIMAL
            else:
                ner_tags.append(0)  # O (Outside)

        dataset.append({"tokens": tokens, "ner_tags": ner_tags})

    return dataset


data = generate_ner_dataset(600)
with open("./ner_data.json", "w") as f:
    json.dump(data, f)

print(f"Generated {len(data)} samples for NER training.")