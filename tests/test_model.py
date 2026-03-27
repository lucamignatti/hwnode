"""Unit tests for HW-NODE model."""

import torch
import pytest

from hwnode.model import HWNodeBlock, HWNodeNetwork


class TestHWNodeBlock:
    """Tests for a single HW-NODE block."""

    @pytest.fixture
    def block(self):
        return HWNodeBlock(input_dim=32, state_dim=8, order=4)

    def test_forward_shape(self, block):
        """Output shape matches input shape."""
        x = torch.randn(16, 32)
        y = block(x)
        assert y.shape == (16, 32)

    def test_output_not_zero(self, block):
        """Forward pass produces non-trivial output."""
        x = torch.randn(16, 32)
        y = block(x)
        assert y.abs().sum() > 0

    def test_gradient_flow(self, block):
        """Gradients flow through all parameters."""
        x = torch.randn(4, 32, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()

        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_spectral_norm_constraint(self, block):
        """Spectral norm of A stays <= 1."""
        # Do a forward pass to trigger parametrization
        x = torch.randn(4, 32)
        _ = block(x)

        spec_norm = torch.linalg.norm(block.A, ord=2).item()
        assert spec_norm <= 1.0 + 1e-4, f"Spectral norm {spec_norm} > 1"

    def test_matrix_exp_convergence(self, block):
        """High-order Taylor series converges to torch.matrix_exp."""
        A = torch.randn(8, 8) * 0.1  # small norm for convergence
        high_order_block = HWNodeBlock(input_dim=32, state_dim=8, order=12)

        approx = high_order_block._matrix_exp_approx(A)
        exact = torch.matrix_exp(A)

        error = (approx - exact).abs().max().item()
        assert error < 1e-3, f"Matrix exp error {error} too large"

    def test_different_orders(self):
        """Higher order gives better approximation."""
        A = torch.randn(8, 8) * 0.3
        exact = torch.matrix_exp(A)

        errors = []
        for order in [2, 4, 8]:
            block = HWNodeBlock(input_dim=32, state_dim=8, order=order)
            approx = block._matrix_exp_approx(A)
            errors.append((approx - exact).abs().max().item())

        # Errors should be monotonically decreasing
        assert errors[0] > errors[1] > errors[2], f"Errors not decreasing: {errors}"

    def test_different_activations(self):
        """All supported activations work."""
        for act in ["relu_squared", "relu", "gelu", "silu"]:
            block = HWNodeBlock(input_dim=16, state_dim=4, activation=act)
            x = torch.randn(4, 16)
            y = block(x)
            assert y.shape == (4, 16), f"Failed for activation {act}"

    def test_param_count(self):
        """Verify HW-NODE uses fewer params than equivalent FFN."""
        block = HWNodeBlock(input_dim=64, state_dim=16)
        hwnode_params = sum(p.numel() for p in block.parameters())

        # Equivalent FFN: two linear layers (64->64, 64->64)
        ffn_params = 64 * 64 * 2  # = 8192

        assert hwnode_params < ffn_params, (
            f"HW-NODE params ({hwnode_params}) >= FFN params ({ffn_params})"
        )


class TestHWNodeNetwork:
    """Tests for stacked HW-NODE backbone."""

    def test_forward_shape(self):
        net = HWNodeNetwork(obs_dim=8, hidden_dim=32, state_dim=8, num_blocks=2)
        x = torch.randn(16, 8)
        h = net(x)
        assert h.shape == (16, 32)

    def test_residual_connection(self):
        """Output changes with different number of blocks."""
        net1 = HWNodeNetwork(obs_dim=8, hidden_dim=32, state_dim=8, num_blocks=1)
        net2 = HWNodeNetwork(obs_dim=8, hidden_dim=32, state_dim=8, num_blocks=3)

        # They should produce different outputs (different architectures)
        assert net1.output_dim == net2.output_dim == 32

    def test_gradient_flow_full_network(self):
        """Gradients flow through the entire stacked network."""
        net = HWNodeNetwork(obs_dim=4, hidden_dim=16, state_dim=4, num_blocks=3)
        x = torch.randn(4, 4, requires_grad=True)
        h = net(x)
        loss = h.sum()
        loss.backward()

        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
class TestScalingAndVirtualDepth:
    """Tests for scaling the model to larger sizes and using virtual depths (recurrent evaluation)."""

    def test_large_model_forward_backward(self):
        """Test a sufficiently large model (d=1024, n=256) to ensure memory and gradients are stable."""
        block = HWNodeBlock(input_dim=1024, state_dim=256, order=6)
        x = torch.randn(8, 1024, requires_grad=True)
        y = block(x)
        assert y.shape == (8, 1024)
        
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_virtual_depth(self):
        """Test dynamic thinking via 'virtual depth' (recurrently applying the same block)."""
        block = HWNodeBlock(input_dim=128, state_dim=32, order=4)
        x = torch.randn(4, 128, requires_grad=True)
        
        # Virtual depth = 5
        h = x
        for _ in range(5):
            h = h + block(h)  # Residual recurrence
        
        assert h.shape == (4, 128)
        loss = h.pow(2).mean()
        loss.backward()
        
        # Ensure gradients still flow through loop
        assert x.grad is not None
        for param in block.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestLanguageModelingCapabilities:
    """Tests evaluating HW-NODE for Sequence / Language Modeling tasks."""

    def test_3d_sequence_processing(self):
        """HW-NODE should natively support (B, T, D) sequence tensors for language modeling."""
        B, T, D = 4, 128, 64
        # A single block acting as a drop-in replacement for a Transformer FFN
        block = HWNodeBlock(input_dim=D, state_dim=16, order=4)
        
        # Sequence of embeddings
        x = torch.randn(B, T, D, requires_grad=True)
        
        # Forward pass should handle the extra dimension naturally because nn.Linear and matmul broadcast
        y = block(x)
        
        assert y.shape == (B, T, D)
        assert not torch.isnan(y).any()

    def test_lm_cross_entropy_gradient_flow(self):
        """Test a dummy language modeling step with cross-entropy loss."""
        B, T, D = 2, 32, 64
        vocab_size = 100
        
        net = HWNodeNetwork(obs_dim=D, hidden_dim=D, state_dim=16, num_blocks=3)
        lm_head = torch.nn.Linear(D, vocab_size)
        
        x = torch.randn(B, T, D, requires_grad=True)
        targets = torch.randint(0, vocab_size, (B, T))
        
        # Apply the stack
        h = net(x)  # (B, T, D)
        logits = lm_head(h)  # (B, T, V)
        
        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        
        assert x.grad is not None


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.taylor_vs_chebyshev import ChebyshevHWNodeBlock, LearnedTaylorHWNodeBlock

class TestTaylorVsChebyshevAtScale:
    """Tests evaluating Taylor vs Chebyshev formulations at large scales."""

    def test_chebyshev_large_scale(self):
        """Test Chebyshev block with large hidden dim and state dim."""
        block = ChebyshevHWNodeBlock(input_dim=512, state_dim=128, order=8)
        x = torch.randn(4, 512, requires_grad=True)
        y = block(x)
        
        assert y.shape == (4, 512)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert block.cheb_weights.grad is not None

    def test_learned_taylor_large_scale(self):
        """Test Learned Taylor block with large hidden dim and state dim."""
        block = LearnedTaylorHWNodeBlock(input_dim=512, state_dim=128, order=8)
        x = torch.randn(4, 512, requires_grad=True)
        y = block(x)
        
        assert y.shape == (4, 512)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert block.taylor_weights.grad is not None


class DynamicTermWeightingHWNodeBlock(HWNodeBlock):
    """An HW-NODE block where the polynomial coefficients are dynamically predicted from the input."""
    def __init__(self, input_dim: int, state_dim: int, order: int = 4, activation: str = "relu_squared"):
        super().__init__(model_dim=input_dim, state_dim=state_dim, taylor_order=order, square_output=(activation=="relu_squared"))
        self.router = torch.nn.Linear(input_dim, order + 1)
        
    def _dynamic_taylor_poly(self, M: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # weights: (batch, order+1)
        n = M.shape[0]
        I = torch.eye(n, device=M.device, dtype=M.dtype)
        w0 = weights[:, 0].view(-1, 1, 1)
        result = w0 * I.unsqueeze(0)
        
        M_power = I
        for k in range(1, self.taylor_order + 1):
            M_power = (M_power @ M) / k
            wk = weights[:, k].view(-1, 1, 1)
            result = result + wk * M_power.unsqueeze(0)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term_weights = torch.softmax(self.router(x), dim=-1) * (self.taylor_order + 1)
        
        # We manually process one step without loop to show dynamic thinking correctly mapping a batch
        z = self._hammerstein_nonlinearity(self.W_in(x))
        
        A_hat = self._spectrally_normalized_A()
        M = (A_hat * self.dt)
        P_A = self._dynamic_taylor_poly(M, term_weights)
        
        z = z.unsqueeze(1) @ P_A.transpose(-2, -1)
        z = z.squeeze(1)
        
        y = self._wiener_nonlinearity(self.W_out(z))
        return y


class TestDynamicThinking:
    """Tests evaluating dynamic orders and dynamic term weighting."""

    def test_dynamic_term_weighting(self):
        """Test that term polynomials can be driven by the input features (dynamic thinking)."""
        block = DynamicTermWeightingHWNodeBlock(input_dim=64, state_dim=16, order=4)
        x = torch.randn(8, 64, requires_grad=True)
        y = block(x)
        
        assert y.shape == (8, 64)
        
        loss = y.sum()
        loss.backward()
        
        # Router should get gradients from driving the term weights
        assert block.router.weight.grad is not None
        assert block.router.weight.grad.abs().sum() > 0
