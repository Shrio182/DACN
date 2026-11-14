# tokenizer.py
# Bộ tách từ và mã hóa (Word-Level Tokenizer) chuyên dụng cho báo cáo y tế tiếng Anh.
# Xây dựng từ điển dựa trên tần số xuất hiện của từ trong tập huấn luyện (Vocabulary).
# Các ký hiệu đặc biệt quy ước:
# <pad> = 0 : Đệm chuỗi cho bằng độ dài batch
# <bos> = 1 : Bắt đầu chuỗi (Beginning of Sentence)
# <eos> = 2 : Kết thúc chuỗi (End of Sentence)
# <unk> = 3 : Từ không có trong từ điển (Unknown token)

import re
import json
from collections import Counter

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL = [PAD, BOS, EOS, UNK]


class SimpleTokenizer:
    def __init__(self, vocab: dict = None):
        self.vocab = vocab or {}
        # Từ điển ngược chuyển từ mã ID sang từ vựng (itos: index-to-string)
        self.itos = {i: s for s, i in self.vocab.items()}

    @staticmethod
    def tokenize(text: str):
        # Tách từ giữ lại các ký tự chữ, số và từ viết tắt y tế (ví dụ: patient's, no')
        return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(text).lower())

    def build_vocab(self, reports, min_freq: int = 1):
        # Đếm tần số xuất hiện của từng từ trong toàn bộ tập văn bản
        counter = Counter()
        for r in reports:
            counter.update(self.tokenize(r))
        # Khởi tạo 4 token đặc biệt
        self.vocab = {s: i for i, s in enumerate(SPECIAL)}
        # Thêm các từ đạt ngưỡng tần suất tối thiểu min_freq
        for w, c in counter.most_common():
            if c >= min_freq:
                self.vocab[w] = len(self.vocab)
        self.itos = {i: s for s, i in self.vocab.items()}
        return self.vocab

    def encode(self, text: str, max_len: int):
        # Chuyển chuỗi văn bản thành danh sách ID số nguyên, kẹp bởi <bos> và <eos>
        tokens = self.tokenize(text)[: max_len - 2]
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in tokens]
        return [self.vocab[BOS]] + ids + [self.vocab[EOS]]

    def decode(self, ids):
        # Giải mã danh sách ID số nguyên ngược lại thành chuỗi văn bản tiếng Anh
        words = [self.itos.get(int(i), UNK) for i in ids]
        if EOS in words:
            words = words[: words.index(EOS)]
        return " ".join(w for w in words if w not in (PAD, BOS))

    def save(self, path: str):
        # Lưu từ điển ra file định dạng JSON
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        # Nạp từ điển từ file JSON có sẵn
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def __len__(self):
        return len(self.vocab)
