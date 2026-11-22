import torch
import torch.nn as nn

from .baseline_common import ResNet101VisualEncoder, beam_search, caption_cross_entropy


class MemoryConditionedLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.condition = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def forward(self, hidden, memory):
        scale, bias = self.condition(memory).chunk(2, dim=-1)
        return (1.0 + scale.unsqueeze(1)) * self.layer_norm(hidden) + bias.unsqueeze(1)


class RelationalMemoryCell(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.update_gate = nn.GRUCell(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory, step_input):
        joint = torch.cat((memory, step_input.unsqueeze(1)), dim=1)
        attended, _ = self.attention(joint, joint, joint, need_weights=False)
        attended_memory = attended[:, :memory.size(1)]
        batch, slots, width = memory.shape
        updated = self.update_gate(
            attended_memory.reshape(batch * slots, width),
            memory.reshape(batch * slots, width),
        ).reshape(batch, slots, width)
        return self.norm(updated)


class R2GenDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.relational_memory = RelationalMemoryCell(d_model, nhead)
        self.visual_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = MemoryConditionedLayerNorm(d_model)
        self.norm2 = MemoryConditionedLayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, memory, visual_features):
        memory = self.relational_memory(memory, hidden[:, 0])
        condition = memory.mean(dim=1)
        hidden = self.norm1(hidden, condition)
        attended, _ = self.visual_attention(
            hidden, visual_features, visual_features, need_weights=False
        )
        hidden = self.norm2(hidden + self.dropout(attended), condition)
        hidden = hidden + self.feed_forward(hidden)
        return hidden, memory


class R2Gen(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=60,
        pad_idx=0,
        memory_slots=3,
        img_size=448,
        weights="DEFAULT",
    ):
        super().__init__()
        if not isinstance(vocab_size, int):
            config = vocab_size
            vocab_size = config.vocab_size
            d_model = config.d_model
            nhead = config.nhead
            num_decoder_layers = config.num_decoder_layers
            dim_feedforward = config.dim_feedforward
            dropout = config.dropout
            max_seq_len = config.max_seq_len
            pad_idx = config.pad_idx
            img_size = config.img_size

        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len
        self.visual_encoder = ResNet101VisualEncoder(d_model, img_size, weights)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )
        self.initial_memory = nn.Parameter(torch.randn(1, memory_slots, d_model) * 0.02)
        self.decoder_layers = nn.ModuleList(
            [
                R2GenDecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.output_norm = MemoryConditionedLayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode_images(self, images):
        regional_features, _ = self.visual_encoder(images)
        return regional_features

    def decode_logits(self, captions, visual_features):
        length = captions.size(1)
        if length > self.max_seq_len:
            raise ValueError(
                f"Caption length {length} exceeds configured maximum {self.max_seq_len}"
            )
        hidden = self.token_embedding(captions) + self.position_embedding[:, :length]
        memories = [
            self.initial_memory.expand(captions.size(0), -1, -1)
            for _ in self.decoder_layers
        ]
        outputs = []
        for step in range(length):
            step_hidden = hidden[:, step:step + 1]
            for index, layer in enumerate(self.decoder_layers):
                step_hidden, memories[index] = layer(
                    step_hidden, memories[index], visual_features
                )
            condition = memories[-1].mean(dim=1)
            outputs.append(self.output_norm(step_hidden, condition))
        return self.lm_head(torch.cat(outputs, dim=1))

    def forward(self, images, captions):
        visual_features = self.encode_images(images)
        logits = self.decode_logits(captions[:, :-1], visual_features)
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
        visual_features = self.encode_images(images)
        return beam_search(
            self,
            visual_features,
            bos_idx,
            eos_idx,
            max_len=min(max_len, self.max_seq_len),
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
