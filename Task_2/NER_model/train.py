import argparse
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification


def main(args):
    with open(args.data_path, "r") as f:
        raw_data = json.load(f)

    dataset = Dataset.from_list(raw_data)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
            label_ids[0] = -100  # [CLS]
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./ner_results",
        eval_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    trainer.train()
    model.save_pretrained(args.model_path)
    tokenizer.save_pretrained(args.model_path)
    print(f"NER Model saved to {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./ner_data.json")
    parser.add_argument("--model_path", type=str, default="./animal_ner_model")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)