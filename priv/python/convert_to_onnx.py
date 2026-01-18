#!/usr/bin/env python3
"""
Convert ExPhil model weights to ONNX format.

Supports LSTM, GRU, and Mamba backbones. Loads weights from NumPy .npz files
exported by scripts/export_numpy.exs.

Usage:
    python convert_to_onnx.py --weights exports/weights.npz --config exports/config.json --output policy.onnx

Options:
    --weights PATH      Path to NumPy weights file (.npz)
    --config PATH       Path to config JSON file
    --output PATH       Output ONNX file path
    --backbone TYPE     Override backbone type (lstm, gru, mamba)
    --opset VERSION     ONNX opset version (default: 17)

Examples:
    # Convert LSTM model
    python convert_to_onnx.py --weights exports/weights.npz --output policy.onnx

    # Convert Mamba model
    python convert_to_onnx.py --weights exports/weights.npz --backbone mamba --output policy_mamba.onnx

Requirements:
    pip install torch onnx numpy
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("\nInstall required packages:")
    print("  pip install torch onnx numpy")
    sys.exit(1)


# ============================================================================
# Model Architectures
# ============================================================================

class SimpleMambaBlock(nn.Module):
    """
    Simplified Mamba block for ONNX export.

    Uses gated linear units instead of full selective SSM to ensure
    ONNX compatibility (no dynamic control flow).
    """

    def __init__(self, hidden_size: int, state_size: int = 16,
                 expand_factor: int = 2, conv_size: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.inner_size = hidden_size * expand_factor

        # Input projection
        self.in_proj = nn.Linear(hidden_size, self.inner_size * 2)

        # Causal 1D convolution
        self.conv1d = nn.Conv1d(
            self.inner_size, self.inner_size,
            kernel_size=conv_size,
            padding=conv_size - 1,  # Causal padding
            groups=self.inner_size
        )

        # SSM parameters (simplified to linear projection)
        self.x_proj = nn.Linear(self.inner_size, state_size * 2)
        self.dt_proj = nn.Linear(self.inner_size, self.inner_size)

        # Output projection
        self.out_proj = nn.Linear(self.inner_size, hidden_size)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: [batch, seq, hidden]
        batch, seq_len, _ = x.shape
        residual = x
        x = self.norm(x)

        # Input projection with gating
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # Causal conv1d
        x_conv = x_inner.transpose(1, 2)  # [batch, inner, seq]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim to causal
        x_conv = x_conv.transpose(1, 2)  # [batch, seq, inner]

        # Simplified SSM: use gated linear combination
        x_conv = torch.nn.functional.silu(x_conv)

        # SSM-like gating
        bc = self.x_proj(x_conv)
        b, c = bc.chunk(2, dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(x_conv))

        # Gated combination (simplified from full selective scan)
        gate = torch.sigmoid(dt.mean(dim=-1, keepdim=True))
        bc_gate = torch.sigmoid((b * c).sum(dim=-1, keepdim=True))
        gated = gate * bc_gate * x_conv

        # Cumulative context via exponential moving average
        # Use a simple linear combination for ONNX compatibility
        alpha = 0.9
        context = torch.zeros_like(gated)
        for t in range(seq_len):
            if t == 0:
                context[:, t] = gated[:, t]
            else:
                context[:, t] = alpha * context[:, t-1].clone() + (1 - alpha) * gated[:, t]

        # Combine with input gate
        y = gated + (1 - gate) * context

        # Output gate and projection
        y = y * torch.nn.functional.silu(z)
        y = self.out_proj(y)

        return y + residual


class MambaBackbone(nn.Module):
    """Stacked Mamba blocks with final sequence extraction."""

    def __init__(self, embed_size: int, hidden_size: int = 256,
                 state_size: int = 16, expand_factor: int = 2,
                 conv_size: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Linear(embed_size, hidden_size)

        self.layers = nn.ModuleList([
            SimpleMambaBlock(hidden_size, state_size, expand_factor, conv_size)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: [batch, seq, embed]
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        x = self.final_norm(x)

        # Return last timestep
        return x[:, -1, :]


class LSTMBackbone(nn.Module):
    """LSTM backbone with sequence_last extraction."""

    def __init__(self, embed_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            embed_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        # x: [batch, seq, embed]
        output, _ = self.lstm(x)
        # Return last timestep
        return output[:, -1, :]


class GRUBackbone(nn.Module):
    """GRU backbone with sequence_last extraction."""

    def __init__(self, embed_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.gru = nn.GRU(
            embed_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        output, _ = self.gru(x)
        return output[:, -1, :]


class PolicyNetwork(nn.Module):
    """
    Full policy network with backbone and action heads.

    Outputs: buttons (8), main_stick (2), c_stick (2), shoulder (1)
    """

    def __init__(self, embed_size: int, hidden_size: int, backbone_type: str,
                 num_layers: int = 2, state_size: int = 16,
                 expand_factor: int = 2, conv_size: int = 4, dropout: float = 0.1):
        super().__init__()

        # Build backbone
        if backbone_type == 'mamba':
            self.backbone = MambaBackbone(
                embed_size, hidden_size, state_size, expand_factor,
                conv_size, num_layers, dropout
            )
        elif backbone_type == 'lstm':
            self.backbone = LSTMBackbone(embed_size, hidden_size, num_layers, dropout)
        elif backbone_type == 'gru':
            self.backbone = GRUBackbone(embed_size, hidden_size, num_layers, dropout)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # Action heads
        self.button_head = nn.Linear(hidden_size, 8)  # 8 buttons
        self.main_stick_head = nn.Linear(hidden_size, 2)  # x, y
        self.c_stick_head = nn.Linear(hidden_size, 2)  # x, y
        self.shoulder_head = nn.Linear(hidden_size, 1)  # trigger

    def forward(self, x):
        # x: [batch, seq, embed]
        features = self.backbone(x)

        buttons = torch.sigmoid(self.button_head(features))
        main_stick = torch.tanh(self.main_stick_head(features))
        c_stick = torch.tanh(self.c_stick_head(features))
        shoulder = torch.sigmoid(self.shoulder_head(features))

        return buttons, main_stick, c_stick, shoulder


# ============================================================================
# Weight Loading
# ============================================================================

def load_weights_from_npz(model: nn.Module, npz_path: str, config: dict) -> None:
    """Load weights from NumPy .npz file into PyTorch model."""

    weights = np.load(npz_path)

    # Map Axon parameter names to PyTorch names
    # This mapping depends on how export_numpy.exs structures the output
    state_dict = model.state_dict()

    for name, param in state_dict.items():
        # Try to find matching weight in npz
        # Axon uses different naming conventions
        npz_name = name.replace('.', '_')

        if npz_name in weights:
            loaded = torch.from_numpy(weights[npz_name])
            if loaded.shape == param.shape:
                state_dict[name] = loaded
                print(f"  Loaded: {name} {tuple(param.shape)}")
            else:
                print(f"  Shape mismatch: {name} (expected {tuple(param.shape)}, got {tuple(loaded.shape)})")
        else:
            print(f"  Missing: {name} (using random init)")

    model.load_state_dict(state_dict)


# ============================================================================
# ONNX Export
# ============================================================================

def export_to_onnx(model: nn.Module, output_path: str, embed_size: int,
                   seq_len: int = 60, opset_version: int = 17) -> None:
    """Export PyTorch model to ONNX format."""

    model.train(False)

    # Create dummy input
    dummy_input = torch.randn(1, seq_len, embed_size)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['buttons', 'main_stick', 'c_stick', 'shoulder'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'seq_len'},
            'buttons': {0: 'batch_size'},
            'main_stick': {0: 'batch_size'},
            'c_stick': {0: 'batch_size'},
            'shoulder': {0: 'batch_size'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"\n[OK] Exported to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert ExPhil weights to ONNX')
    parser.add_argument('--weights', type=str, required=True, help='Path to weights.npz')
    parser.add_argument('--config', type=str, help='Path to config.json')
    parser.add_argument('--output', type=str, default='policy.onnx', help='Output path')
    parser.add_argument('--backbone', type=str, choices=['lstm', 'gru', 'mamba'],
                        help='Override backbone type')
    parser.add_argument('--embed-size', type=int, default=1991, help='Embedding size')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--state-size', type=int, default=16, help='Mamba state size')
    parser.add_argument('--expand-factor', type=int, default=2, help='Mamba expand factor')
    parser.add_argument('--conv-size', type=int, default=4, help='Mamba conv size')
    parser.add_argument('--seq-len', type=int, default=60, help='Sequence length')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    args = parser.parse_args()

    print("""
========================================================================
                      ExPhil ONNX Converter
========================================================================
""")

    # Load config if provided
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
        print(f"Loaded config from: {args.config}")

    # Determine backbone type
    backbone = args.backbone or config.get('backbone', 'lstm')
    embed_size = args.embed_size or config.get('embed_size', 1991)
    hidden_size = args.hidden_size or config.get('hidden_size', 256)
    num_layers = args.num_layers or config.get('num_layers', 2)

    print(f"Configuration:")
    print(f"  Backbone:    {backbone}")
    print(f"  Embed size:  {embed_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers:  {num_layers}")

    if backbone == 'mamba':
        print(f"  State size:  {args.state_size}")
        print(f"  Expand:      {args.expand_factor}x")
        print(f"  Conv size:   {args.conv_size}")

    # Build model
    print(f"\nBuilding {backbone.upper()} model...")
    model = PolicyNetwork(
        embed_size=embed_size,
        hidden_size=hidden_size,
        backbone_type=backbone,
        num_layers=num_layers,
        state_size=args.state_size,
        expand_factor=args.expand_factor,
        conv_size=args.conv_size,
    )

    # Load weights if provided
    if Path(args.weights).exists():
        print(f"\nLoading weights from: {args.weights}")
        load_weights_from_npz(model, args.weights, config)
    else:
        print(f"\nWarning: Weights file not found: {args.weights}")
        print("  Using random initialization (for testing only)")

    # Export to ONNX
    print(f"\nExporting to ONNX (opset {args.opset})...")
    export_to_onnx(model, args.output, embed_size, args.seq_len, args.opset)

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model validation passed")
    except Exception as e:
        print(f"[WARN] ONNX validation warning: {e}")

    print("\nDone! Next steps:")
    print(f"  1. Quantize: python priv/python/quantize_onnx.py {args.output} policy_int8.onnx")
    print(f"  2. Benchmark with onnxruntime")


if __name__ == '__main__':
    main()
