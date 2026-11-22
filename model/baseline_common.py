import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models import resnet101


class ResNet101VisualEncoder(nn.Module):
    def __init__(self, d_model, img_size=448, weights="DEFAULT"):
        super().__init__()
        backbone = resnet101(weights=weights)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        for module in self.features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.projection = nn.Conv2d(2048, d_model, kernel_size=1)
        self.position = nn.Parameter(
            torch.randn(1, d_model, math.ceil(img_size / 32), math.ceil(img_size / 32))
            * 0.02
        )
        self.view_embedding = nn.Parameter(torch.randn(1, 2, 1, d_model) * 0.02)
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _encode_view(self, images):
        if images.size(1) == 1:
            images = images.expand(-1, 3, -1, -1)
        elif images.size(1) != 3:
            raise ValueError(f"Expected 1-channel or 3-channel images, got {images.size(1)}")

        if images.detach().amin().item() < 0:
            images = (images + 1.0) / 2.0
        images = (images - self.imagenet_mean) / self.imagenet_std

        if self.training:
            feature_map = checkpoint_sequential(
                self.features, segments=5, input=images, use_reentrant=False
            )
        else:
            feature_map = self.features(images)

        feature_map = self.projection(feature_map)
        position = F.interpolate(
            self.position, size=feature_map.shape[-2:], mode="bilinear", align_corners=False
        )
        return (feature_map + position).flatten(2).transpose(1, 2)

    def forward(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError("Expected images with shape (B,V,C,H,W) or (B,C,H,W)")
        if images.size(1) < 1 or images.size(1) > 2:
            raise ValueError(f"Expected one or two views, got {images.size(1)}")

        view_features = []
        for view_index in range(images.size(1)):
            features = self._encode_view(images[:, view_index])
            features = features + self.view_embedding[:, view_index]
            view_features.append(features)
        regional_features = torch.cat(view_features, dim=1)
        global_features = regional_features.mean(dim=1)
        return regional_features, global_features


def caption_cross_entropy(logits, captions, pad_idx):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        captions[:, 1:].reshape(-1),
        ignore_index=pad_idx,
    )


def _banned_ngram_tokens(sequence, ngram_size):
    tokens = sequence.tolist()
    if ngram_size <= 0 or len(tokens) < ngram_size - 1:
        return set()
    prefix = tuple(tokens[-(ngram_size - 1):]) if ngram_size > 1 else ()
    banned = set()
    for start in range(len(tokens) - ngram_size + 1):
        if tuple(tokens[start:start + ngram_size - 1]) == prefix:
            banned.add(tokens[start + ngram_size - 1])
    return banned


@torch.no_grad()
def beam_search(
    model,
    visual_features,
    bos_idx,
    eos_idx,
    max_len=60,
    beam_size=3,
    no_repeat_ngram_size=3,
    length_penalty=0.7,
):
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")
    batch_size = (
        visual_features[0].size(0)
        if isinstance(visual_features, (list, tuple))
        else visual_features.size(0)
    )
    device = (
        visual_features[0].device
        if isinstance(visual_features, (list, tuple))
        else visual_features.device
    )
    best_sequences = []

    for batch_index in range(batch_size):
        if isinstance(visual_features, (list, tuple)):
            visual = [features[batch_index:batch_index + 1] for features in visual_features]
        else:
            visual = visual_features[batch_index:batch_index + 1]
        initial = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
        beams = [(initial, 0.0, False)]

        for _ in range(max_len - 1):
            if all(done for _, _, done in beams):
                break
            candidates = []
            for sequence, score, done in beams:
                if done:
                    candidates.append((sequence, score, True))
                    continue
                logits = model.decode_logits(sequence, visual)[:, -1, :].squeeze(0)
                for token_id in _banned_ngram_tokens(sequence[0], no_repeat_ngram_size):
                    logits[token_id] = -torch.inf
                log_probs = torch.log_softmax(logits, dim=-1)
                top_scores, top_tokens = torch.topk(log_probs, beam_size)
                for token_score, token in zip(top_scores, top_tokens):
                    token_tensor = token.reshape(1, 1)
                    next_sequence = torch.cat((sequence, token_tensor), dim=1)
                    next_score = score + token_score.item()
                    candidates.append(
                        (next_sequence, next_score, token.item() == eos_idx)
                    )

            candidates.sort(
                key=lambda candidate: candidate[1]
                / (candidate[0].size(1) ** length_penalty),
                reverse=True,
            )
            beams = candidates[:beam_size]

        best_sequences.append(beams[0][0])

    result = torch.full(
        (batch_size, max(sequence.size(1) for sequence in best_sequences)),
        model.pad_idx,
        dtype=torch.long,
        device=device,
    )
    for index, sequence in enumerate(best_sequences):
        result[index, :sequence.size(1)] = sequence[0]
    return result
