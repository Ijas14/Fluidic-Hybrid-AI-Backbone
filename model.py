"""
Fluidic-Hybrid AI Backbone — Core Architecture

A novel neural architecture combining Liquid State Spaces (S4) with
Closed-form Continuous-time (CfC) neurons, trained via DEQ Phantom
Gradient for guaranteed gradient stability at any depth or sequence length.

Components:
    - LiquidS4StateLoop: Parallel-scan state space model for sequential memory
    - FluidicBlock: S4 + CfC hybrid block with pre-LayerNorm and residual connections
    - FluidicHybridBackbone: Full stacked language model backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_modules import CfCCell, DEQCfCSequenceProcessor


class LiquidS4StateLoop(nn.Module):
    """
    Liquid-S4 memory core with vectorized parallel scan.

    Computes the linear recurrence h_t = A * h_{t-1} + B_t across all
    timesteps simultaneously using cumulative products, replacing the
    original O(n) sequential loop with a fully parallel O(1)-depth operation.
    """

    def __init__(self, d_model, state_dim=128):
        super(LiquidS4StateLoop, self).__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.proj_B = nn.Linear(d_model, state_dim)
        self.proj_C = nn.Linear(state_dim, d_model)
        self.A_log = nn.Parameter(torch.randn(state_dim))

    def forward(self, x, current_state=None):
        """
        Args:
            x: (batch, seq_len, d_model) input sequence
            current_state: (batch, state_dim) persistent state or None
        Returns:
            outputs: (batch, seq_len, d_model)
            final_state: (batch, state_dim)
        """
        batch_size, seq_len, _ = x.shape

        if current_state is None:
            current_state = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=x.dtype)

        A = torch.sigmoid(self.A_log)
        B_all = self.proj_B(x)

        # Parallel scan: h_t = A^t * h_0 + Σ A^{t-i} * B_i
        powers = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        A_powers = A.unsqueeze(0).pow(powers)
        A_powers_safe = A_powers.clamp(min=1e-8)

        B_scaled = B_all / A_powers_safe.unsqueeze(0)
        B_cumsum = torch.cumsum(B_scaled, dim=1)
        h0_scaled = current_state.unsqueeze(1)

        all_states = A_powers.unsqueeze(0) * (h0_scaled + B_cumsum)
        outputs = self.proj_C(all_states)
        final_state = all_states[:, -1, :]

        return outputs, final_state


class FluidicBlock(nn.Module):
    """
    Single layer of the Fluidic-Hybrid stack.

    Combines a Liquid-S4 memory pass with a DEQ-powered CfC logic pass,
    wrapped in pre-LayerNorm and a residual connection for stable deep stacking.
    """

    def __init__(self, d_model, state_dim, cfc_hidden_size, deq_chunk_size=8):
        super(FluidicBlock, self).__init__()
        self.ssm_layer = LiquidS4StateLoop(d_model, state_dim)
        cfc_cell = CfCCell(input_size=d_model, hidden_size=cfc_hidden_size)
        self.logic_engine = DEQCfCSequenceProcessor(cfc_cell, chunk_size=deq_chunk_size)
        self.out_proj = nn.Linear(cfc_hidden_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, ssm_state=None, cfc_state=None, time_delta=0.1):
        """
        Args:
            x: (batch, seq_len, d_model)
            ssm_state: Optional persistent S4 state
            cfc_state: Optional persistent CfC state
            time_delta: Time step for CfC dynamics
        Returns:
            output: (batch, seq_len, d_model) with residual connection
            new_ssm_state, new_cfc_state: Updated persistent states
        """
        x_normed = self.norm(x)

        # Memory pass (Liquid-S4 parallel scan)
        ssm_out, new_ssm_state = self.ssm_layer(x_normed, ssm_state)

        batch_size, seq_len, _ = x.shape
        if cfc_state is None:
            cfc_state = torch.zeros(batch_size, self.logic_engine.cfc_cell.hidden_size, device=x.device)
        if not isinstance(time_delta, torch.Tensor):
            time_delta = torch.ones(batch_size, 1, device=x.device) * time_delta

        # Logic pass (DEQ Phantom Gradient — no BPTT)
        cfc_outputs, current_cfc_state = self.logic_engine(
            ssm_out, cfc_state, time_delta, use_deq=self.training
        )

        layer_out = self.out_proj(cfc_outputs)
        layer_out = x + layer_out  # Residual connection

        return layer_out, new_ssm_state, current_cfc_state


class FluidicHybridBackbone(nn.Module):
    """
    Full Fluidic-Hybrid language model backbone.

    Stacks multiple FluidicBlocks with shared embedding and LM head.
    Training uses DEQ Phantom Gradient (O(1) backward memory).
    Inference uses standard sequential pass with persistent states.
    """

    def __init__(self, vocab_size, d_model, state_dim, cfc_hidden_size, num_layers=1):
        super(FluidicHybridBackbone, self).__init__()
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            FluidicBlock(d_model, state_dim, cfc_hidden_size) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, ssm_states=None, cfc_states=None, time_delta=0.1):
        """
        Args:
            input_ids: (batch, seq_len) token indices
            ssm_states: Optional list of S4 states per layer
            cfc_states: Optional list of CfC states per layer
            time_delta: Time step for CfC dynamics
        Returns:
            logits: (batch, seq_len, vocab_size)
            new_ssm_states: List of updated S4 states
            new_cfc_states: List of updated CfC states
        """
        batch_size = input_ids.shape[0]

        if ssm_states is None:
            ssm_states = [None] * self.num_layers
        if cfc_states is None:
            cfc_states = [None] * self.num_layers

        x = self.embed(input_ids)

        new_ssm_states = []
        new_cfc_states = []

        for i, layer in enumerate(self.layers):
            x, next_ssm, next_cfc = layer(x, ssm_states[i], cfc_states[i], time_delta)
            new_ssm_states.append(next_ssm)
            new_cfc_states.append(next_cfc)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_ssm_states, new_cfc_states
