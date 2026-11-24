import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from data.chexpert_labeler import CHEXPERT_CLASSES, rule_based_label
from data.iu_xray_dataset import IUXrayDataset
from data.tokenizer import SimpleTokenizer
from model.full_model import MulticriteriaTransformer


def load_model(checkpoint_path, vocab_path, device):
    tokenizer = SimpleTokenizer.load(str(vocab_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    mlc_key = next(
        (
            key
            for key, value in state.items()
            if "mlc_head" in key and "weight" in key
            and getattr(value, "ndim", 0) == 2
        ),
        None,
    )
    num_labels = int(state[mlc_key].shape[0]) if mlc_key else 65

    model_config = ModelConfig()
    model_config.vocab_size = len(tokenizer)
    model_config.pad_idx = tokenizer.vocab["<pad>"]
    model_config.num_mlc_labels = num_labels
    model = MulticriteriaTransformer(model_config, TrainConfig()).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: {len(missing)} checkpoint keys missing")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected checkpoint keys")
    model.eval()
    return model, tokenizer


def generate_report(model, images, bos_idx, eos_idx, tokenizer, device):
    images = images.to(device)
    ids = model.generate(
        images,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        max_len=60,
        beam_size=3,
        repetition_penalty=1.20,
        visual_guidance=0.8,
        no_repeat_ngram_size=3,
    )
    return [tokenizer.decode(row.tolist()) for row in ids]


def decode_reference(item, tokenizer):
    if item.get("report"):
        return str(item["report"])
    return tokenizer.decode(item.get("report_ids", []))


def binary_metrics(y_true, y_pred):
    rows = []
    true_sum = int(np.asarray(y_true).sum())
    pred_sum = int(np.asarray(y_pred).sum())
    true_positive = int(np.asarray(y_true)[np.asarray(y_pred) == 1].sum())
    false_positive = int(pred_sum - true_positive)
    false_negative = int(true_sum - true_positive)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1


def calculate_metrics(references, predictions):
    true_vectors = np.asarray(
        [rule_based_label(reference) for reference in references],
        dtype=np.float32,
    )
    pred_vectors = np.asarray(
        [rule_based_label(prediction) for prediction in predictions],
        dtype=np.float32,
    )
    per_class = {}
    for index, name in enumerate(CHEXPERT_CLASSES):
        precision, recall, f1 = binary_metrics(
            true_vectors[:, index],
            pred_vectors[:, index],
        )
        per_class[name] = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        }

    per_precision = np.asarray([values["Precision"] for values in per_class.values()])
    per_recall = np.asarray([values["Recall"] for values in per_class.values()])
    per_f1 = np.asarray([values["F1-score"] for values in per_class.values()])

    micro_precision, micro_recall, micro_f1 = binary_metrics(
        true_vectors.reshape(-1),
        pred_vectors.reshape(-1),
    )
    return per_class, {
        "Macro": {
            "Precision": float(per_precision.mean()),
            "Recall": float(per_recall.mean()),
            "F1-score": float(per_f1.mean()),
        },
        "Micro": {
            "Precision": micro_precision,
            "Recall": micro_recall,
            "F1-score": micro_f1,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate clinical CheXpert label efficacy of generated reports."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data_dir", default="data/archive")
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", default="artifacts/clinical_efficacy_test.csv")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    test_json = data_dir / "processed_test.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not test_json.is_file():
        raise FileNotFoundError(f"Test split not found: {test_json}")

    vocab_path = (
        Path(args.vocab)
        if args.vocab
        else data_dir / "vocab.json"
    )
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(checkpoint, vocab_path, device)
    dataset = IUXrayDataset(
        str(test_json),
        image_root=str(data_dir),
        max_len=60,
        img_size=448,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    with test_json.open("r", encoding="utf-8") as file:
        test_items = json.load(file)

    references = []
    predictions = []
    for batch_index, (images, _, _) in enumerate(loader):
        batch_predictions = generate_report(
            model,
            images,
            tokenizer.vocab["<bos>"],
            tokenizer.vocab["<eos>"],
            tokenizer,
            device,
        )
        predictions.extend(batch_predictions)
        start = batch_index * args.batch_size
        for item in test_items[start:start + len(batch_predictions)]:
            references.append(decode_reference(item, tokenizer))

    per_class, aggregates = calculate_metrics(references, predictions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Label", "Precision", "Recall", "F1-score"])
        for label, values in per_class.items():
            writer.writerow([
                label,
                f'{values["Precision"]:.6f}',
                f'{values["Recall"]:.6f}',
                f'{values["F1-score"]:.6f}',
            ])
        for aggregate_name, values in aggregates.items():
            writer.writerow([
                aggregate_name,
                f'{values["Precision"]:.6f}',
                f'{values["Recall"]:.6f}',
                f'{values["F1-score"]:.6f}',
            ])

    print("CheXpert clinical efficacy on Test")
    print("Label | Precision | Recall | F1-score")
    for label, values in per_class.items():
        print(
            f"{label} | {values['Precision']:.4f} | "
            f"{values['Recall']:.4f} | {values['F1-score']:.4f}"
        )
    for aggregate_name, values in aggregates.items():
        print(
            f"{aggregate_name} | {values['Precision']:.4f} | "
            f"{values['Recall']:.4f} | {values['F1-score']:.4f}"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
