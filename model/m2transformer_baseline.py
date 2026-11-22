import torch
import torch.nn as nn

from .baseline_common import ResNet101VisualEncoder, beam_search, caption_cross_entropy


class MemoryAugmentedEncoder(nn.Module):
    def __init__(
        self, d_model, nhead, num_layers, dim_feedforward, dropout, num_memory_slots
    ):
        super().__init__()
        self.memory_slots = nn.Parameter(torch.randn(1, num_memory_slots, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, visual_features):
        memory = self.memory_slots.expand(visual_features.size(0), -1, -1)
        hidden = torch.cat((visual_features, memory), dim=1)
        layer_outputs = []
        for layer in self.layers:
            hidden = layer(hidden)
            layer_outputs.append(hidden)
        return layer_outputs


class MeshedDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                for _ in range(num_encoder_layers)
            ]
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, encoder_outputs, causal_mask, alpha_logits):
        attended, _ = self.self_attention(
            hidden, hidden, hidden, attn_mask=causal_mask, need_weights=False
        )
        hidden = self.norms[0](hidden + self.dropout(attended))

        alpha = torch.softmax(alpha_logits, dim=0)
        meshed = torch.zeros_like(hidden)
        for weight, cross_attention, encoder_output in zip(
            alpha, self.cross_attentions, encoder_outputs
        ):
            attended, _ = cross_attention(
                hidden, encoder_output, encoder_output, need_weights=False
            )
            meshed = meshed + weight * attended
        hidden = self.norms[1](hidden + self.dropout(meshed))
        hidden = self.norms[2](hidden + self.feed_forward(hidden))
        return hidden


class M2Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=60,
        pad_idx=0,
        num_memory_slots=40,
        img_size=448,
        weights="DEFAULT",
    ):
        super().__init__()
        if not isinstance(vocab_size, int):
            config = vocab_size
            vocab_size = config.vocab_size
            d_model = config.d_model
            nhead = config.nhead
            num_encoder_layers = config.num_encoder_layers
            num_decoder_layers = config.num_decoder_layers
            dim_feedforward = config.dim_feedforward
            dropout = config.dropout
            max_seq_len = config.max_seq_len
            pad_idx = config.pad_idx
            img_size = config.img_size

        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len
        self.visual_encoder = ResNet101VisualEncoder(d_model, img_size, weights)
        self.encoder = MemoryAugmentedEncoder(
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            num_memory_slots,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )
        self.decoder_layers = nn.ModuleList(
            [
                MeshedDecoderLayer(
                    d_model,
                    nhead,
                    num_encoder_layers,
                    dim_feedforward,
                    dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.alpha = nn.Parameter(torch.zeros(num_decoder_layers, num_encoder_layers))
        self.decoder_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode_images(self, images):
        visual_features, _ = self.visual_encoder(images)
        return self.encoder(visual_features)

    def decode_logits(self, captions, encoder_outputs):
        length = captions.size(1)
        if length > self.max_seq_len:
            raise ValueError(
                f"Caption length {length} exceeds configured maximum {self.max_seq_len}"
            )
        hidden = self.token_embedding(captions) + self.position_embedding[:, :length]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=captions.device),
            diagonal=1,
        )
        layer_outputs = []
        for index, layer in enumerate(self.decoder_layers):
            hidden = layer(hidden, encoder_outputs, causal_mask, self.alpha[index])
            layer_outputs.append(hidden)
        hidden = torch.stack(layer_outputs, dim=0).mean(dim=0)
        return self.lm_head(self.decoder_norm(hidden))

    def forward(self, images, captions):
        encoder_outputs = self.encode_images(images)
        logits = self.decode_logits(captions[:, :-1], encoder_outputs)
        loss = caption_cross_entropy(logits, captions, self.pad_idx)
        return loss, {"report_generation": loss.item(), "total": loss.item()}

    @torch.no_grad()
    def generate(
        self,
        images,
        bos_idx,
        eos_idx,
        max_len=60,
        beam_size=3,
        no_repeat_ngram_size=3,
    ):
        encoder_outputs = self.encode_images(images)
        return beam_search(
            self,
            encoder_outputs,
            bos_idx,
            eos_idx,
            max_len=min(max_len, self.max_seq_len),
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
