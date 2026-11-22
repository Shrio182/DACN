import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from data.iu_xray_dataset import IUXrayDataset
from data.tokenizer import BOS, EOS, PAD, SimpleTokenizer
from model.full_model import MulticriteriaTransformer
from model.m2transformer_baseline import M2Transformer
from model.r2gen_baseline import R2Gen


SEED = 42
EXPECTED_SPLIT_COUNTS = {"train": 2351, "val": 293, "test": 293}
MODEL_NAMES = ("m2transformer", "r2gen", "pure_transformer")
RESULT_FIELDS = [
    "model",
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "ROUGE-L",
    "CIDEr",
    "METEOR",
    "parameters",
    "parameter_size_MB",
    "train_seconds_per_epoch",
    "inference_seconds_per_case",
    "best_epoch",
    "epochs_trained",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_split(path, expected_count):
    import json

    if not path.is_file():
        raise FileNotFoundError(f"Required IU-Xray split is missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        items = json.load(file)
    if len(items) != expected_count:
        raise ValueError(
            f"{path.name} has {len(items)} cases; expected {expected_count} "
            "for the IU-Xray Archive benchmark."
        )
    return items


def validate_dataset(data_dir, vocab_path):
    splits = {
        name: read_split(
            data_dir / f"processed_{name}.json", EXPECTED_SPLIT_COUNTS[name]
        )
        for name in EXPECTED_SPLIT_COUNTS
    }
    identifiers = {}
    for split_name, items in splits.items():
        ids = [str(item.get("id", "")) for item in items]
        if any(not identifier for identifier in ids):
            raise ValueError(f"{split_name} contains a sample without an id.")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{split_name} contains duplicate sample ids.")
        identifiers[split_name] = set(ids)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = identifiers[left] & identifiers[right]
        if overlap:
            raise ValueError(
                f"Sample leakage between {left} and {right}: "
                f"{len(overlap)} ids overlap."
            )

    if not vocab_path.is_file():
        raise FileNotFoundError(f"Archive vocabulary is missing: {vocab_path}")
    tokenizer = SimpleTokenizer.load(str(vocab_path))
    for token, expected_id in ((PAD, 0), (BOS, 1), (EOS, 2)):
        if tokenizer.vocab.get(token) != expected_id:
            raise ValueError(
                f"{vocab_path} must map {token!r} to id {expected_id}."
            )
    if sorted(tokenizer.vocab.values()) != list(range(len(tokenizer))):
        raise ValueError("Vocabulary ids must be contiguous and start at zero.")

    label_count = None
    for split_name, items in splits.items():
        for item in items:
            image_paths = item.get("image_path", [])
            if len(image_paths) != 2:
                raise ValueError(
                    f"{split_name} sample {item['id']} must contain exactly "
                    "Frontal and Lateral image paths."
                )
            if not item.get("report_ids"):
                raise ValueError(f"{split_name} sample {item['id']} has no tokenized report.")
            if "mlc_labels" not in item:
                raise ValueError(f"{split_name} sample {item['id']} has no MLC labels.")
            if any(
                not isinstance(token_id, int) or token_id < 0 or token_id >= len(tokenizer)
                for token_id in item["report_ids"]
            ):
                raise ValueError(
                    f"{split_name} sample {item['id']} contains a token id outside the archive vocabulary."
                )
            current_label_count = len(item["mlc_labels"])
            if label_count is None:
                label_count = current_label_count
            elif current_label_count != label_count:
                raise ValueError("MLC label vectors have inconsistent lengths.")
            for image_path in image_paths:
                if not (data_dir / image_path).is_file():
                    raise FileNotFoundError(
                        f"Image for {split_name} sample {item['id']} is missing: "
                        f"{data_dir / image_path}"
                    )
    return splits, tokenizer


def make_loader(data_dir, split, batch_size, workers, shuffle=False):
    dataset = IUXrayDataset(
        str(data_dir / f"processed_{split}.json"),
        image_root=str(data_dir),
        max_len=60,
        img_size=448,
    )
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def compute_term_weights(train_items, tokenizer, device):
    counts = torch.zeros(len(tokenizer), dtype=torch.float64)
    total = 1
    for item in train_items:
        if "report" in item:
            ids = [
                tokenizer.vocab.get(token, tokenizer.vocab["<unk>"])
                for token in tokenizer.tokenize(item["report"])
            ]
        else:
            ids = item["report_ids"]
        for token_id in ids:
            token_id = int(token_id)
            if token_id > 3 and token_id < len(tokenizer):
                counts[token_id] += 1
                total += 1
    weights = torch.ones(len(tokenizer), dtype=torch.float32)
    used = counts > 0
    weights[used] = 1.0 + torch.log(
        torch.tensor(float(total), dtype=torch.float64) / (1.0 + counts[used])
    ).float()
    return (weights / weights.mean()).to(device)


def make_model(model_name, tokenizer, num_mesh_labels, device):
    mcfg = ModelConfig()
    mcfg.vocab_size = len(tokenizer)
    mcfg.pad_idx = tokenizer.vocab[PAD]
    mcfg.max_seq_len = 60
    mcfg.img_size = 448
    mcfg.use_paired_views = True
    mcfg.num_mlc_labels = num_mesh_labels

    if model_name == "m2transformer":
        model = M2Transformer(mcfg)
    elif model_name == "r2gen":
        model = R2Gen(mcfg)
    elif model_name == "pure_transformer":
        model = MulticriteriaTransformer(mcfg, TrainConfig())
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def build_metric_scorers():
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as error:
        raise RuntimeError(
            "The benchmark requires pycocoevalcap (BLEU, ROUGE-L, CIDEr, "
            "METEOR) and its Java runtime. Install it before running."
        ) from error
    return {
        "tokenizer": PTBTokenizer(),
        "bleu": Bleu(4),
        "rouge": Rouge(),
        "cider": Cider(),
        "meteor": Meteor(),
    }


def tokenize_for_metrics(scorers, references, hypotheses):
    return (
        scorers["tokenizer"].tokenize(references),
        scorers["tokenizer"].tokenize(hypotheses),
    )


def calculate_cider(scorers, references, hypotheses):
    reference_tokens, hypothesis_tokens = tokenize_for_metrics(
        scorers, references, hypotheses
    )
    score, _ = scorers["cider"].compute_score(reference_tokens, hypothesis_tokens)
    return float(score)


def calculate_all_metrics(scorers, references, hypotheses):
    reference_tokens, hypothesis_tokens = tokenize_for_metrics(
        scorers, references, hypotheses
    )
    bleu_scores, _ = scorers["bleu"].compute_score(
        reference_tokens, hypothesis_tokens
    )
    rouge_l, _ = scorers["rouge"].compute_score(
        reference_tokens, hypothesis_tokens
    )
    cider, _ = scorers["cider"].compute_score(
        reference_tokens, hypothesis_tokens
    )
    meteor, _ = scorers["meteor"].compute_score(
        reference_tokens, hypothesis_tokens
    )
    return {
        "BLEU-1": float(bleu_scores[0]),
        "BLEU-2": float(bleu_scores[1]),
        "BLEU-3": float(bleu_scores[2]),
        "BLEU-4": float(bleu_scores[3]),
        "ROUGE-L": float(rouge_l),
        "CIDEr": float(cider),
        "METEOR": float(meteor),
    }


@torch.no_grad()
def generate_predictions(model, loader, tokenizer, device, scorers, max_len=60):
    model.eval()
    references = {}
    hypotheses = {}
    sample_index = 0
    inference_seconds = 0.0
    for images, report_ids, _ in loader:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        generation_args = {
            "bos_idx": tokenizer.vocab["<bos>"],
            "eos_idx": tokenizer.vocab["<eos>"],
            "max_len": max_len,
            "beam_size": 3,
            "no_repeat_ngram_size": 3,
        }
        if isinstance(model, MulticriteriaTransformer):
            generation_args.update(repetition_penalty=1.0, visual_guidance=0.0)
        generated = model.generate(images, **generation_args)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - started

        for reference, hypothesis in zip(report_ids, generated.cpu()):
            references[sample_index] = [tokenizer.decode(reference.tolist())]
            hypotheses[sample_index] = [tokenizer.decode(hypothesis.tolist())]
            sample_index += 1
    if sample_index == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return references, hypotheses, inference_seconds / sample_index


def evaluate_cider(model, loader, tokenizer, device, scorers):
    references, hypotheses, _ = generate_predictions(
        model, loader, tokenizer, device, scorers
    )
    return calculate_cider(scorers, references, hypotheses)


def train_one_model(
    model_name,
    model,
    data_dir,
    tokenizer,
    term_weights,
    device,
    scorers,
    epochs,
    batch_size,
    workers,
    patience,
    output_dir,
):
    train_loader = make_loader(
        data_dir, "train", batch_size, workers, shuffle=True
    )
    val_loader = make_loader(data_dir, "val", batch_size, workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_cider = -math.inf
    best_epoch = 0
    patience_count = 0
    epoch_durations = []
    checkpoint_path = output_dir / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        loss_sum = 0.0
        batches = 0
        for images, captions, labels in train_loader:
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if model_name == "pure_transformer":
                    labels = labels.to(device, non_blocking=True)
                    loss, loss_parts = model(
                        images, captions, labels, term_weights
                    )
                else:
                    loss, loss_parts = model(images, captions)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss_parts["total"])
            batches += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - epoch_started
        epoch_durations.append(epoch_time)
        val_cider = evaluate_cider(
            model, val_loader, tokenizer, device, scorers
        )
        print(
            f"[{model_name}] epoch {epoch}/{epochs}: "
            f"loss={loss_sum / max(batches, 1):.4f}, "
            f"val CIDEr={val_cider:.4f}, train={epoch_time:.1f}s"
        )

        if val_cider > best_cider:
            best_cider = val_cider
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(
                    f"[{model_name}] early stopping after {patience} "
                    "non-improving validation epochs."
                )
                break

    if best_epoch == 0:
        raise RuntimeError(f"No best checkpoint was produced for {model_name}.")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    return best_epoch, len(epoch_durations), float(np.mean(epoch_durations)), checkpoint_path


def evaluate_test(
    model, model_name, data_dir, tokenizer, device, scorers, batch_size, workers
):
    test_loader = make_loader(data_dir, "test", batch_size, workers)
    references, hypotheses, inference_seconds = generate_predictions(
        model, test_loader, tokenizer, device, scorers
    )
    metrics = calculate_all_metrics(scorers, references, hypotheses)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_size_mb = (
        sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        / (1024 ** 2)
    )
    return {
        "model": model_name,
        **metrics,
        "parameters": parameter_count,
        "parameter_size_MB": parameter_size_mb,
        "inference_seconds_per_case": inference_seconds,
    }


def save_results(results, output_dir):
    import matplotlib.pyplot as plt

    csv_path = output_dir / "fair_benchmark_archive.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    metric_names = [
        "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
        "ROUGE-L", "CIDEr", "METEOR",
    ]
    x = np.arange(len(metric_names))
    width = 0.8 / max(len(results), 1)
    for index, result in enumerate(results):
        axes[0, 0].bar(
            x + index * width,
            [result[name] for name in metric_names],
            width,
            label=result["model"],
        )
    axes[0, 0].set_xticks(x + width * (len(results) - 1) / 2, metric_names)
    axes[0, 0].set_title("NLG scores (same test split and decoding)")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.25)

    plot_specs = [
        (axes[0, 1], "parameter_size_MB", "Parameter size (MiB)"),
        (axes[1, 0], "train_seconds_per_epoch", "Training seconds / epoch"),
        (axes[1, 1], "inference_seconds_per_case", "Inference seconds / case"),
    ]
    for axis, field, title in plot_specs:
        axis.bar([result["model"] for result in results], [result[field] for result in results])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    png_path = output_dir / "fair_benchmark_archive.png"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved results: {csv_path} and {png_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fair IU-Xray Archive benchmark for M2Transformer, R2Gen and Pure Transformer."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Mounted directory with images/ and processed_{train,val,test}.json.",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("artifacts/archive/vocab_archive.json"),
        help="Existing archive vocabulary; it is read-only.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="Models to train and evaluate, in execution order.",
    )
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive.")
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models must not contain duplicates.")

    set_seed(SEED)
    data_dir = args.data_dir.resolve()
    vocab_path = args.vocab.resolve()
    output_dir = args.output_dir.resolve()
    baseline_dir = output_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    splits, tokenizer = validate_dataset(data_dir, vocab_path)
    num_mesh_labels = len(splits["train"][0]["mlc_labels"])
    if num_mesh_labels < 1:
        raise ValueError("Training samples must contain at least one MLC label.")
    term_weights = compute_term_weights(splits["train"], tokenizer, "cpu")
    scorers = build_metric_scorers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device={device}; split sizes="
        f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}; "
        f"vocabulary={len(tokenizer)}; seed={SEED}; "
        f"batch={args.batch_size}; lr=1e-4"
    )
    if device.type != "cuda":
        print("Warning: CUDA is unavailable; this full benchmark may be very slow on CPU.")

    results = []
    try:
        for model_name in args.models:
            set_seed(SEED)
            model = make_model(
                model_name, tokenizer, num_mesh_labels, device
            )
            model.to(device)
            best_epoch, epochs_trained, train_time, checkpoint_path = train_one_model(
                model_name,
                model,
                data_dir,
                tokenizer,
                term_weights.to(device),
                device,
                scorers,
                args.epochs,
                args.batch_size,
                args.num_workers,
                args.patience,
                baseline_dir,
            )
            row = evaluate_test(
                model,
                model_name,
                data_dir,
                tokenizer,
                device,
                scorers,
                args.batch_size,
                args.num_workers,
            )
            row["train_seconds_per_epoch"] = train_time
            row["best_epoch"] = best_epoch
            row["epochs_trained"] = epochs_trained
            results.append(row)
            print(
                f"[{model_name}] best epoch={best_epoch}, "
                f"test CIDEr={row['CIDEr']:.4f}, "
                f"checkpoint={checkpoint_path}"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        scorers["meteor"].close()

    save_results(results, output_dir)
    for result in results:
        print(
            f"{result['model']}: "
            + ", ".join(f"{key}={result[key]:.4f}" for key in (
                "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
                "ROUGE-L", "CIDEr", "METEOR",
            ))
        )


if __name__ == "__main__":
    main()
