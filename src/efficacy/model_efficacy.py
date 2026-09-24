"""VCPE-rna efficacy head: ASO sequence + chemistry + context -> target inhibition %.

Trains on ASO Atlas (190,927 gapmer ASOs, 343 genes, patent-derived). This is the
"efficacy half" of the VCPE engine: for a given ASO (sequence + sugar/backbone
chemistry + dose + cell line), predict how deeply the target mRNA is knocked down.
Combined with the response half (CRISPR-trained deviation head), this completes the
Tier-2 chain: ASO -> off-targets -> response x depth.

Architecture (~3M trainable):
  - per-position embedding: base(16) + sugar(16) + backbone(16), concat 48-dim,
    projected to d=128
  - learned positional embedding over the 30 sequence positions, added after the
    projection (see POSITIONAL ENCODING below)
  - 2-layer transformer encoder over positions (d=128) -> masked mean+max pool
  - target-gene context: frozen ESM2 table (5120) -> proj 128  [mechanism axis reuse]
  - cell_line embedding (16) + dosage log10 + missing flag
  - MLP head -> inhibition %

POSITIONAL ENCODING (v4 fix, changes model behaviour)
-----------------------------------------------------
Releases up to v3 fed the per-position embeddings straight into
`nn.TransformerEncoder` with no positional signal. Self-attention is
permutation-equivariant and mean/max pooling is permutation-invariant, so the
whole sequence branch collapsed to a BAG of (base, sugar, backbone) triples: the
model could not distinguish a 5-10-5 MOE gapmer from the same nucleotides in any
other order, and could only read off per-type counts. That directly contradicts
the claim that modification x sequence-context interactions are captured by
attention -- with no positions there is no context, only composition.

`use_pos_emb=True` (the default) adds a learned positional embedding. Set it to
False to reproduce the legacy bag-of-triples behaviour; the with/without
comparison is the ablation that establishes whether position actually matters
for this task, and it should be reported rather than assumed.

PADDING INDICES (v4 fix)
------------------------
Releases up to v3 used `padding_idx=0` for the sugar and backbone embeddings,
where index 0 is "DNA" and "PO" respectively -- i.e. the embedding rows for the
gap chemistry of every gapmer, and for the unmodified backbone, were pinned to
zero and never received gradient. A dedicated PAD index is used instead, so no
real chemistry class is frozen. Padded positions are additionally masked via
`src_key_padding_mask`, as before.
"""
import torch
import torch.nn as nn

# Index 0 is reserved for padding in every vocabulary, so that no chemically
# meaningful class shares an index with PAD.
PAD = 0
BASES = {"PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "U": 4, "N": 5}
SUGARS = {"PAD": 0, "DNA": 1, "MOE": 2, "cEt": 3, "F": 4, "other": 5}
BACKBONES = {"PAD": 0, "PO": 1, "PS": 2}
MAX_LEN = 30

# Legacy (<= v3) vocabularies, kept so that released checkpoints remain loadable.
LEGACY_BASES = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": 4}
LEGACY_SUGARS = {"DNA": 0, "MOE": 1, "cEt": 2, "F": 3, "other": 4}
LEGACY_BACKBONES = {"PO": 0, "PS": 1}


def vocabs(legacy=False):
    """Return (BASES, SUGARS, BACKBONES) for the requested architecture generation."""
    if legacy:
        return LEGACY_BASES, LEGACY_SUGARS, LEGACY_BACKBONES
    return BASES, SUGARS, BACKBONES


def load_efficacy_head(ckpt, esm_matrix, n_cell_lines, strict=True):
    """Rebuild an EfficacyHead from a checkpoint dict, honouring its architecture.

    Checkpoints written by v4+ carry `arch_config`. Anything older predates both
    the positional embedding and the PAD-index fix, so it is reconstructed in
    legacy mode with a warning -- loading such a checkpoint into the current
    architecture would silently reinterpret every sugar/backbone index.

    Also tolerates the `esm_table` entry that older checkpoints carry: the ESM2
    table is a frozen copy of a public lookup table and is now a non-persistent
    buffer, which removes ~400 MB of redundant bytes from every checkpoint.
    """
    cfg = ckpt.get("arch_config")
    if cfg is None:
        cfg = dict(use_pos_emb=False, legacy_vocab=True)
        print("[efficacy] checkpoint has no arch_config: assuming pre-v4 layout "
              "(no positional embedding, legacy vocabularies). Predictions from "
              "it are bag-of-triples over chemistry, not position-aware.",
              flush=True)
    model = EfficacyHead(esm_matrix, n_cell_lines=n_cell_lines,
                         use_pos_emb=cfg.get("use_pos_emb", False),
                         legacy_vocab=cfg.get("legacy_vocab", True))
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k != "esm_table"}
    model.load_state_dict(sd, strict=strict)
    return model, cfg


class EfficacyHead(nn.Module):
    def __init__(self, esm_matrix, n_cell_lines=128, d=128, hidden=256, dropout=0.1,
                 use_pos_emb=True, legacy_vocab=False):
        super().__init__()
        esm_matrix = esm_matrix.float()
        self.register_buffer("esm_table", esm_matrix, persistent=False)
        self.esm_dim = esm_matrix.shape[1]
        self.use_pos_emb = use_pos_emb
        self.legacy_vocab = legacy_vocab

        b, s, k = vocabs(legacy_vocab)
        if legacy_vocab:
            # Reproduces the <= v3 layout exactly, including the padding_idx
            # collisions documented above, for checkpoint compatibility.
            self.base_emb = nn.Embedding(len(set(b.values())), 16, padding_idx=4)
            self.sugar_emb = nn.Embedding(len(set(s.values())), 16, padding_idx=0)
            self.bb_emb = nn.Embedding(len(set(k.values())), 16, padding_idx=0)
        else:
            self.base_emb = nn.Embedding(len(set(b.values())), 16, padding_idx=PAD)
            self.sugar_emb = nn.Embedding(len(set(s.values())), 16, padding_idx=PAD)
            self.bb_emb = nn.Embedding(len(set(k.values())), 16, padding_idx=PAD)

        self.in_proj = nn.Linear(48, d)
        self.pos_emb = nn.Embedding(MAX_LEN, d) if use_pos_emb else None
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=2 * d,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.esm_proj = nn.Linear(self.esm_dim, d)
        self.cell_emb = nn.Embedding(n_cell_lines, 16)
        self.head = nn.Sequential(
            nn.Linear(2 * d + d + 16 + 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def arch_config(self):
        """Serialise the architecture switches, so checkpoints reconstruct exactly."""
        return dict(use_pos_emb=self.use_pos_emb, legacy_vocab=self.legacy_vocab,
                    max_len=MAX_LEN)

    def forward(self, base_ids, sugar_ids, bb_ids, pad_mask, gene_rows,
                cell_ids, dose_log, dose_missing):
        """base/sugar/bb [B, L] long; pad_mask [B, L] bool (True=pad);
        gene_rows [B] long; cell_ids [B] long; dose_log/missing [B] float."""
        x = self.in_proj(torch.cat([self.base_emb(base_ids),
                                    self.sugar_emb(sugar_ids),
                                    self.bb_emb(bb_ids)], dim=-1))  # [B, L, 48] -> d
        if self.pos_emb is not None:
            L = x.shape[1]
            pos = torch.arange(L, device=x.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = torch.cat([(x * valid).sum(1) / valid.sum(1).clamp(min=1.0),
                            x.masked_fill(pad_mask.unsqueeze(-1), -1e4).max(1).values],
                           dim=-1)                                 # [B, 2d]
        g = self.esm_proj(self.esm_table[gene_rows])               # [B, d]
        f = torch.cat([pooled, g, self.cell_emb(cell_ids),
                       dose_log.unsqueeze(-1), dose_missing.unsqueeze(-1)], dim=-1)
        return self.head(f).squeeze(-1)
