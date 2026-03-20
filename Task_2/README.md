```markdown
# Task 2: Animal Verification Pipeline

## Overview
A multi-modal system that verifies if a text description (e.g., "There is a dog") matches an uploaded image.

## Components
### 1. Named Entity Recognition (NER)
- **Model:** Fine-tuned `distilbert-base-uncased`.
- **Logic:** Identifies animal names in raw sentences.
- **Inference:** Managed by `NER_model/inference.py`.

### 2. Computer Vision (CV)
- **Model:** MobileNetV2 with Transfer Learning.
- **Preprocessing:** Integrated `Rescaling` layer (0-255 to -1, 1).
- **Inference:** Managed by `CV_model/inference.py`.

### 3. Verification Pipeline
The `pipeline.py` script imports both inference modules and performs a logic check:
`Match = (Text_Entity == Predicted_Class) AND (Confidence > 0.5)`

## How to Run
- **Train NER:** Follow scripts in `NER_model/`.
- **Train CV:** Follow scripts in `CV_model/`.
- **Test Pipeline:**
  ```bash
  python Task_2/pipeline.py --text "Look at this cat" --image "path/to/img.jpg"

  NER-model is too large for GitHub. You should train the model to generate the animal_ner_model folder locally.