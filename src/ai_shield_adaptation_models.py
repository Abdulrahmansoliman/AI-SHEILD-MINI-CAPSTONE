"""Research models: original semantic head, audited loaders, strict checkpoints."""
import math
from contextlib import nullcontext
import torch
from torch import nn

from src.ai_shield_adaptation import SEMANTIC_ID, SEMANTIC_REVISION


class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.15):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch_size, num_tokens, _ = x.shape
        x = x.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch_size, num_heads, num_tokens, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, num_tokens, num_heads * head_dim)

    def forward(self, x):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)
        attention_for_summary = attention
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)
        context = self._merge_heads(context)
        output = self.out_proj(context)
        return output, attention_for_summary


class SemanticTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=2.0, dropout=0.15):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CustomMultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_input = self.norm1(x)
        attn_output, attention = self.attention(attn_input)
        x = x + self.dropout1(attn_output)
        mlp_input = self.norm2(x)
        x = x + self.mlp(mlp_input)
        return x, attention


class SemanticTransformerHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        d_model=256,
        num_heads=4,
        depth=1,
        mlp_ratio=2.0,
        attention_dim=256,
        classifier_hidden_dim=256,
        dropout=0.15,
    ):
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, d_model)
        self.blocks = nn.ModuleList([
            SemanticTransformerBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.pool_key = nn.Linear(d_model, attention_dim)
        self.pool_energy = nn.Linear(attention_dim, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def _attention_summary(self, attention_maps):
        eps = 1e-8
        summaries = []
        for attention in attention_maps:
            num_tokens = attention.shape[-1]
            entropy = -(attention * torch.log(attention + eps)).sum(dim=-1)
            entropy = entropy / math.log(num_tokens)
            focus = attention.max(dim=-1).values
            summaries.append((entropy.mean(dim=(1, 2)), focus.mean(dim=(1, 2))))
        entropy = torch.stack([item[0] for item in summaries], dim=0).mean(dim=0)
        focus = torch.stack([item[1] for item in summaries], dim=0).mean(dim=0)
        return entropy, focus

    def forward(self, patch_tokens):
        x = self.input_projection(patch_tokens)
        self_attention_maps = []
        for block in self.blocks:
            x, attention = block(x)
            self_attention_maps.append(attention)
        x = self.final_norm(x)

        pool_energy = self.pool_energy(torch.tanh(self.pool_key(x))).squeeze(-1)
        pool_weights = torch.softmax(pool_energy, dim=1)
        pooled = torch.sum(pool_weights.unsqueeze(-1) * x, dim=1)
        logits = self.classifier(pooled).squeeze(-1)

        eps = 1e-8
        pool_entropy = -(pool_weights * torch.log(pool_weights + eps)).sum(dim=1)
        pool_entropy = pool_entropy / math.log(pool_weights.shape[1])
        pool_focus = pool_weights.max(dim=1).values
        self_entropy, self_focus = self._attention_summary(self_attention_maps)

        return {
            "logits": logits,
            "attention_weights": pool_weights,
            "attention_entropy": pool_entropy,
            "attention_focus": pool_focus,
            "self_attention_entropy": self_entropy,
            "self_attention_focus": self_focus,
            "semantic_feature": pooled,
        }

class EfficientNetBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1, drop_rate=0.2)

    def forward(self, x):
        return self.model(x).squeeze(1)


class ViTSemanticAttentionClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.vit = backbone
        self.semantic_head = SemanticTransformerHead(hidden_size=backbone.config.hidden_size)
        self.vit.requires_grad_(False)

    def forward(self, pixel_values):
        # The saved audit used cuDNN's default TF32 convolution arithmetic.
        # Scope it to ViT so CF and the other branches retain their own settings.
        precision = torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic,
            allow_tf32=True,
        ) if pixel_values.device.type == "cuda" else nullcontext()
        with precision:
            output = self.vit(pixel_values=pixel_values, return_dict=True)
        return self.semantic_head(output.last_hidden_state[:, 1:, :])


def load_original_models(config):
    from transformers import AutoImageProcessor, AutoModel
    # Original files are user's own training artifacts; preflight verifies their hashes.
    effnet = EfficientNetBinaryClassifier()
    saved = torch.load(config.checkpoints["effnet"], map_location="cpu", weights_only=False)
    state = saved["model_state"]
    if not all(k.startswith("model.") for k in state):
        state = {"model." + k: v for k, v in state.items()}
    effnet.load_state_dict(state, strict=True)

    processor = AutoImageProcessor.from_pretrained(SEMANTIC_ID, revision=SEMANTIC_REVISION, use_fast=True)
    backbone = AutoModel.from_pretrained(SEMANTIC_ID, revision=SEMANTIC_REVISION)
    semantic = ViTSemanticAttentionClassifier(backbone)
    saved = torch.load(config.checkpoints["semantic"], map_location="cpu", weights_only=False)
    if saved.get("config", {}).get("freeze_vit_backbone", True) is not True:
        raise ValueError("Cannot restore only the head of a trained backbone")
    state = {k.removeprefix("semantic_head."): v for k, v in saved["model_state"].items()
             if k.startswith("semantic_head.")}
    if not state:
        raise ValueError("Semantic head weights are missing")
    semantic.semantic_head.load_state_dict(state, strict=True)
    return {"effnet": effnet.eval(), "semantic": semantic.eval()}, processor

