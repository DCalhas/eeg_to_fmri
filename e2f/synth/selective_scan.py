"""Pure-PyTorch selective scan for Spec2VolCAMU-Net's VMamba blocks (mamba_ssm's CUDA kernel is not installed).

Vendored from the audit session's ~/eeg_to_fmri/audit/loso_published.py (2026-10-23), which follows
mamba_ssm.ops.selective_scan_interface.selective_scan_ref (Apache-2.0): same recurrence
x_t = exp(delta_t A) x_{t-1} + delta_t B_t u_t, solved exactly within chunks and carried across chunks.
"""
import sys
import torch


def selective_scan_chunked(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, chunk=8):
	"""Same recurrence as selective_scan_ref, x_t = exp(delta_t A) x_{t-1} + delta_t B_t u_t, solved exactly per chunk
	(x_t = sum_{s<=t} exp(S_t - S_s) dBu_s with S the within-chunk cumsum of delta A <= 0, so every exponent is <= 0)
	and carried sequentially across chunks: l/chunk Python steps instead of l."""
	dtype_in = u.dtype
	u, delta = u.float(), delta.float()
	if delta_bias is not None:
		delta = delta + delta_bias[..., None].float()
	if delta_softplus:
		delta = torch.nn.functional.softplus(delta)
	b, d, l = u.shape
	n = A.shape[1]
	B = B.float().repeat_interleave(d // B.shape[1], dim=1)
	C = C.float().repeat_interleave(d // C.shape[1], dim=1)
	dA = torch.einsum("bdl,dn->bdnl", delta, A)
	dBu = torch.einsum("bdl,bdnl,bdl->bdnl", delta, B, u)
	pad = (-l) % chunk
	if pad:
		dA, dBu = torch.nn.functional.pad(dA, (0, pad)), torch.nn.functional.pad(dBu, (0, pad))
	nc = (l + pad) // chunk
	S = dA.view(b, d, n, nc, chunk).cumsum(-1)
	dBu = dBu.view(b, d, n, nc, chunk)
	tril = torch.ones(chunk, chunk, dtype=torch.bool, device=u.device).tril()
	h = u.new_zeros((b, d, n))
	xs = []
	for k in range(nc):
		Sk = S[:, :, :, k]
		W = torch.exp((Sk[..., :, None] - Sk[..., None, :]).masked_fill(~tril, float("-inf")))
		xk = torch.einsum("bdnts,bdns->bdnt", W, dBu[:, :, :, k]) + torch.exp(Sk) * h[..., None]
		h = xk[..., -1]
		xs.append(xk)
	x = torch.cat(xs, dim=-1)[..., :l]
	y = torch.einsum("bdnl,bdnl->bdl", x, C)
	out = y if D is None else y + u * D[:, None]
	if z is not None:
		out = out * torch.nn.functional.silu(z)
	out = out.to(dtype=dtype_in)
	return (out, x[..., -1]) if return_last_state else out


def build():


def patch_vmamba():
	"""Install the chunked scan (with activation checkpointing) as vmunet.vmamba.selective_scan_fn."""
	import importlib
	vm = importlib.import_module("vmunet.vmamba")
	vm.selective_scan_fn = lambda *a, **k: torch.utils.checkpoint.checkpoint(selective_scan_chunked, *a, use_reentrant=False, **k)
