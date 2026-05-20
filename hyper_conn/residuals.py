"""
Gated Residual Connections

This module provides alternative residual connection mechanisms that use
gating to control information flow, rather than simple addition.

===============================================================================
CONTEXT: RESIDUAL CONNECTIONS IN HYPER-CONNECTIONS
===============================================================================

Standard residual connection:
    x_{l+1} = x_l + F(x_l)

This is the simplest form where F's output is directly added to the input.
The KromHC framework (and mHC variants) extend this with learnable mixing
matrices that control how multiple residual streams interact.

GATED RESIDUALS provide an alternative extension:
    x_{l+1} = g * x_l + (1-g) * F(x_l)

where g is a learned gate that interpolates between preserving the
residual (g=1) and using the branch output (g=0).

===============================================================================
GATING MECHANISMS
===============================================================================

1. GRUGatedResidual:
   Uses a GRU cell to compute the gate, treating the branch output
   as the "input" and the residual as the "hidden state".
   
   This leverages the GRU's learned update mechanism which has been
   proven effective in sequential modeling.

2. GatedResidual:
   Uses a simple linear projection from concatenated inputs to
   compute gate values, then interpolates.
   
   Lighter weight than GRU but still provides adaptive mixing.

===============================================================================
COMPARISON WITH HYPER-CONNECTIONS
===============================================================================

| Mechanism         | Mixing Capacity | Parameters | Computational Cost |
|-------------------|-----------------|------------|-------------------|
| Standard Residual | Fixed (add)     | 0          | O(1)              |
| GatedResidual     | 1D gate         | O(d)       | O(d)              |
| GRUGatedResidual  | GRU dynamics    | O(d²)      | O(d²)             |
| HyperConnections  | n×n mixing      | O(n²C)     | O(n²C)            |
| mHC / KromHC      | n×n DS mixing   | O(n²C)     | O(n²C)            |

Gated residuals provide a middle ground between fixed addition and
full stream mixing. They're useful when:
- Only 1 residual stream is needed (n=1)
- Fine-grained per-element control is preferred over stream mixing
- Computational budget is limited

===============================================================================
USAGE WITH HYPER-CONNECTIONS
===============================================================================

These gated residuals can be used as the `depth_residual_fn` in
HyperConnections variants:

    gated = GatedResidual(dim)
    hc = HyperConnections(
        num_streams,
        dim=dim,
        depth_residual_fn=gated
    )

This replaces the simple addition in the depth connection with
gated mixing, potentially improving gradient flow.
"""

import torch
from torch import nn
from torch.nn import Module

from einops import rearrange, pack, unpack


class GRUGatedResidual(Module):
    """
    GRU-based gated residual connection.
    
    Uses a GRU cell to compute how to combine the branch output (new
    information) with the residual (existing information).
    
    The GRU naturally handles the update vs. reset trade-off:
    - Update gate z: how much of the old state to keep
    - Reset gate r: how much of the old state influences the candidate
    
    COMPUTATION:
    ============
    Treating branch output x as "input" and residual as "hidden state h":
    
        z = σ(W_z @ x + U_z @ h)           # Update gate
        r = σ(W_r @ x + U_r @ h)           # Reset gate
        h̃ = tanh(W @ x + U @ (r ⊙ h))    # Candidate
        h' = (1 - z) ⊙ h + z ⊙ h̃         # New hidden state
    
    The output h' adaptively combines:
    - Old residual (when z ≈ 0)
    - New branch output (when z ≈ 1)
    
    PROPERTIES:
    ===========
    - Learned gating dynamics (not just interpolation)
    - 3 gate matrices: update, reset, candidate
    - Parameter count: O(3 * d²) = O(d²)
    
    COMPARISON WITH LSTM:
    ====================
    GRU is lighter than LSTM (2 gates vs 3) while achieving similar
    performance in practice. For residual connections, the simpler
    structure is often preferred.
    """
    
    def __init__(self, dim):
        """
        Initialize GRU-gated residual.
        
        Args:
            dim: Feature dimension d
        """
        super().__init__()
        # GRUCell: input_size = hidden_size = dim
        # Parameters: 3 * dim * dim weights for input-hidden
        #           + 3 * dim * dim weights for hidden-hidden
        #           + 3 * dim biases
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, residual):
        """
        Apply GRU gating to combine branch output with residual.
        
        Args:
            x: Branch output F(x_l), shape (..., d)
            residual: Residual stream x_l, shape (..., d)
            
        Returns:
            Gated combination, shape (..., d)
        """
        # Pack arbitrary leading dimensions into batch
        x, packed_shape = pack([x], '* d')
        residual, _ = pack([residual], '* d')

        # GRU: treat x as input, residual as hidden state
        output = self.gru(x, residual)

        # Unpack back to original shape
        output, = unpack(output, packed_shape, '* d')
        return output


class GatedResidual(Module):
    """
    Simple linear-gated residual connection.
    
    Computes a learned gate from the concatenation of branch output
    and residual, then interpolates between them.
    
    COMPUTATION:
    ============
        gate = σ(Linear([x, residual]))    # Gate in [0, 1]
        output = lerp(x, residual, gate)   # Interpolation
               = (1 - gate) * x + gate * residual
    
    When gate ≈ 1: output ≈ residual (preserve input)
    When gate ≈ 0: output ≈ x (use branch output)
    
    FINE_GATE OPTION:
    ================
    - fine_gate=False: Single scalar gate (1 parameter output)
    - fine_gate=True: Per-dimension gates (d parameter outputs)
    
    Fine gating allows different dimensions to have different
    mixing ratios, at the cost of more parameters.
    
    PROPERTIES:
    ===========
    - Simple linear projection for gate computation
    - Parameter count: O(2d) or O(2d²) for fine gate
    - Symmetric treatment of x and residual (unlike GRU)
    
    COMPARISON WITH GRU:
    ===================
    - Simpler: just one linear layer vs. full GRU cell
    - Cheaper: O(d) or O(d²) vs O(d²)
    - Less expressive: no memory dynamics
    """
    
    def __init__(self, dim, fine_gate=False):
        """
        Initialize linear-gated residual.
        
        Args:
            dim: Feature dimension d
            fine_gate: If True, compute per-dimension gates
        """
        super().__init__()
        
        # Project concatenated inputs to gate values
        # Input: [x, residual] of size 2d
        # Output: 1 (scalar gate) or d (fine gate)
        self.to_learned_mix = nn.Linear(dim * 2, dim if fine_gate else 1)

    def forward(self, x, residual):
        """
        Apply linear gating to combine branch output with residual.
        
        Args:
            x: Branch output F(x_l), shape (b, n, ...)
            residual: Residual stream x_l, shape (b, n, ...)
            
        Returns:
            Gated combination: (1-g)*x + g*residual
        """
        # Concatenate along last dimension
        x_and_residual, _ = pack([x, residual], 'b n *')

        # Compute gate(s) via linear projection + sigmoid
        mix = self.to_learned_mix(x_and_residual)

        # Linear interpolation: lerp(x, residual, mix) = (1-mix)*x + mix*residual
        # torch.lerp(start, end, weight) = start + weight * (end - start)
        out = x.lerp(residual, mix.sigmoid())
        return out
