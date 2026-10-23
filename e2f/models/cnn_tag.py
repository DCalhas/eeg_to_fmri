"""PyTorch port of the MLHC 2023 EEG-to-fMRI synthesizer (topographical attention + ResBlocks +
random Fourier features + style prior) and of its classification adaptation.

Faithful to src/models/eeg_to_fmri.py and src/models/classifiers.py (TF 2.9), with the
TF defaults made explicit (Glorot-uniform weights, zero biases, BN eps=1e-3 / momentum 0.99,
LayerNorm eps=1e-3). Tensors are channels-first: EEG (B, 1, C, F, T), fMRI (B, 1, X, Y, Z).
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsemax(z, dim=-1):
	"""Sparsemax (Martins & Astudillo 2016), as tfa.activations.sparsemax."""
	z = z - z.max(dim=dim, keepdim=True).values
	zs, _ = torch.sort(z, dim=dim, descending=True)
	k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
	shape = [1] * z.dim(); shape[dim] = -1
	k = k.view(shape)
	cs = zs.cumsum(dim)
	support = (1 + k * zs) > cs
	k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)
	tau = (cs.gather(dim, k_z - 1) - 1) / k_z.to(z.dtype)
	return torch.clamp(z - tau, min=0)


def _glorot_vector_(t):
	"""Keras GlorotUniform on a 1-D shape: fan_in = fan_out = n -> limit sqrt(3/n)."""
	n = t.numel()
	nn.init.uniform_(t, -math.sqrt(3.0 / n), math.sqrt(3.0 / n))


def _probe_numel(module, in_shape):
	"""Output size of `module` for one input, without touching BatchNorm running statistics."""
	was = module.training
	module.eval()
	with torch.no_grad():
		n = module(torch.zeros(1, 1, *in_shape)).numel()
	module.train(was)
	return n


def _glorot(m):
	if isinstance(m, (nn.Conv3d, nn.Linear)):
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			nn.init.zeros_(m.bias)


class TopographicalAttention(nn.Module):
	"""c = X A^T (B,C,C); W = sparsemax(c); out = W X. X is (B, C, F*T)."""

	def __init__(self, channels, features):
		super().__init__()
		self.A = nn.Parameter(torch.empty(channels, features))
		nn.init.xavier_uniform_(self.A)

	def forward(self, x):
		W = sparsemax(torch.einsum("bcf,mf->bcm", x, self.A), dim=-1)
		self.scores = W
		return torch.bmm(W, x)


class ResBlock3d(nn.Module):
	"""left: conv(k,s,valid) -> maxpool -> BN -> ReLU -> conv(3,same) -> BN -> ReLU;
	right: conv(k,s,valid) -> maxpool -> BN; out = ReLU(left + right)."""

	def __init__(self, in_ch, out_ch, k, s, pool_k=(2, 2, 1), pool_s=(1, 1, 1), dropout=0.0):
		super().__init__()
		bn = lambda: nn.BatchNorm3d(out_ch, eps=1e-3, momentum=0.01)
		self.left = nn.Sequential(nn.Conv3d(in_ch, out_ch, k, s), nn.MaxPool3d(pool_k, pool_s), bn(), nn.ReLU(),
								  nn.Conv3d(out_ch, out_ch, 3, 1, padding="same"), bn(), nn.ReLU())
		self.right = nn.Sequential(nn.Conv3d(in_ch, out_ch, k, s), nn.MaxPool3d(pool_k, pool_s), bn())
		self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

	def forward(self, x):
		return self.dropout(F.relu(self.left(x) + self.right(x)))

	def conv_params(self):
		return [m.weight for m in self.modules() if isinstance(m, nn.Conv3d)] + \
			   [m.bias for m in self.modules() if isinstance(m, nn.Conv3d)]


class RandomFourierFeatures(nn.Module):
	"""cos input: LayerNorm_{gamma=beta=pi/2}((x W / scale) + b), W~N(0,1), b~U(0,2pi).
	Returns the pre-cosine value; cos is applied by the caller (as the TF Sinusoids layer)."""

	def __init__(self, in_dim, out_dim, scale=None, trainable_scale=True):
		super().__init__()
		self.register_buffer("W", torch.randn(in_dim, out_dim))
		self.register_buffer("b", torch.rand(out_dim) * 2 * math.pi)
		s = torch.tensor([math.sqrt(in_dim / 2.0) if scale is None else float(scale)])
		self.scale = nn.Parameter(s, requires_grad=trainable_scale)
		self.ln = nn.LayerNorm(out_dim, eps=1e-3, elementwise_affine=False)

	def forward(self, x):
		# Keras NonNeg constraint projects scale to >= 0 after each step; scale -> 0 would divide by zero there too.
		# Here the projection is applied in the forward pass (same feasible set, avoids inf).
		h = x @ (self.W / self.scale.clamp(min=1e-12)) + self.b
		return self.ln(h) * (math.pi / 2) + math.pi / 2


class EEGEncoder(nn.Module):
	"""attention (optional) -> ResBlocks -> Dense(latent)."""

	def __init__(self, in_shape, latent=(7, 7, 7), n_channels=4, kernels=((7, 37, 2), (7, 7, 2)),
				 strides=((3, 5, 1), (2, 2, 1)), attention=True, dropout=0.0):
		super().__init__()
		C, Fr, T = in_shape
		self.in_shape = in_shape
		self.attention = TopographicalAttention(C, Fr * T) if attention else None
		blocks, ch = [], 1
		for k, s in zip(kernels, strides):
			blocks.append(ResBlock3d(ch, n_channels, k, s, dropout=dropout)); ch = n_channels
		self.blocks = nn.Sequential(*blocks)
		n = _probe_numel(self.blocks, in_shape)
		self.latent = latent
		self.dense = nn.Linear(n, math.prod(latent))
		self.apply(_glorot)

	def forward(self, x):
		if self.attention is not None:
			B, _, C, Fr, T = x.shape
			x = self.attention(x.reshape(B, C, Fr * T)).reshape(B, 1, C, Fr, T)
		return self.dense(self.blocks(x).flatten(1))


class FMRIEncoder(nn.Module):
	"""Encoder of the fMRI autoencoder used only for latent alignment during synthesis training."""

	def __init__(self, in_shape=(64, 64, 30), latent=(7, 7, 7), n_channels=4,
				 kernels=((14, 6, 12), (6, 4, 6), (12, 47, 5)), strides=((2, 1, 1), (1, 1, 1), (1, 1, 1)), dropout=0.5):
		super().__init__()
		blocks, ch = [], 1
		for k, s in zip(kernels, strides):
			blocks.append(ResBlock3d(ch, n_channels, k, s, pool_k=(2, 2, 2), pool_s=(1, 1, 1))); ch = n_channels
		self.blocks = nn.Sequential(*blocks)
		n = _probe_numel(self.blocks, in_shape)
		self.dense = nn.Linear(n, math.prod(latent))
		self.dropout = nn.Dropout(dropout)  # fmri_ae encoder: Dropout(0.5) after the latent Dense (theta dropout=True)
		self.apply(_glorot)

	def forward(self, y):
		return self.dropout(self.dense(self.blocks(y).flatten(1)))


class Decoder(nn.Module):
	"""RFF -> cos -> style W (elementwise) -> Dense(voxels) -> 1x1x1 conv.
	Only the deterministic decoder (the configuration used for the 2023 synthesis/classification) is ported;
	the padded/variational iDCT decoders of eeg_to_fmri.py are not."""

	def __init__(self, latent_dim=343, out_shape=(64, 64, 30), trainable_scale=True):
		super().__init__()
		self.out_shape = out_shape
		self.rff = RandomFourierFeatures(latent_dim, latent_dim, trainable_scale=trainable_scale)
		self.style = nn.Parameter(torch.empty(latent_dim))
		_glorot_vector_(self.style)
		self.dense = nn.Linear(latent_dim, math.prod(out_shape))
		self.outfilter = nn.Conv3d(1, 1, 1)
		_glorot(self.dense); _glorot(self.outfilter)

	def sinusoids(self, z):
		"""Pre-cosine projection (what the contrastive loss acts on)."""
		return self.rff(z)

	def from_sinusoids(self, s):
		v = self.dense(torch.cos(s) * self.style)
		return self.outfilter(v.view(-1, 1, *self.out_shape))

	def forward(self, z):
		return self.from_sinusoids(self.sinusoids(z))


class CNNTAG(nn.Module):
	"""Synthesis model. forward(eeg) -> (fmri_hat, z_eeg); align(fmri) -> z_fmri."""

	def __init__(self, eeg_shape=(64, 134, 10), fmri_shape=(64, 64, 30), latent=(7, 7, 7)):
		super().__init__()
		self.encoder = EEGEncoder(eeg_shape, latent)
		self.fmri_encoder = FMRIEncoder(fmri_shape, latent)
		self.decoder = Decoder(math.prod(latent), fmri_shape)

	def forward(self, eeg):
		z = self.encoder(eeg)
		return self.decoder(z), z

	def align(self, fmri):
		return self.fmri_encoder(fmri)

	def l2_penalty(self, wd):
		"""Keras L2(wd) on ResBlock conv kernels+biases of both encoders: wd * sum(w^2)."""
		ps = [p for m in self.modules() if isinstance(m, ResBlock3d) for p in m.conv_params()]
		return wd * sum((p ** 2).sum() for p in ps)


def legacy_synthesis_loss(fmri_hat, fmri, z_eeg, z_fmri, latent=(7, 7, 7)):
	"""mae_cosine from losses_utils: per-sample voxel MAE + (1 - tensordot(z1, z2, [[1,2,3],[3,2,1]]) / (|Z1||Z2|)).
	The TF 'cosine' contracts spatial axes in reversed order and normalises by whole-batch norms,
	giving a (B,B) term that broadcasts against the (B,) MAE before the mean; reproduced as is."""
	mae = (fmri_hat - fmri).abs().mean(dim=(1, 2, 3, 4))
	z1, z2 = z_eeg.view(-1, *latent), z_fmri.view(-1, *latent)
	cos = 1.0 - torch.einsum("aijk,bkji->ab", z1, z2) / (z1.norm() * z2.norm())
	return (mae[None, :] + cos).mean()


class LegacyClassificationView(nn.Module):
	"""Reproduces pretrained_EEG_to_fMRI(..., trainable=True) + ViewLatentContrastiveClassifier.

	As in the 2023 code: the encoder (no attention, Dropout 0.5 after each block, new final Dense)
	is freshly initialised, the RFF kernel/bias are *re-sampled* (only the scale is copied), and only
	the style vector and the voxel Dense/1x1 conv are copied from the pretrained synthesizer (frozen).
	With `use_pretrained_encoder=True` the encoder convs, Dense and RFF are copied instead (corrected).
	"""

	def __init__(self, synth: CNNTAG, eeg_shape, strides=None, use_pretrained_encoder=False, freeze_encoder=False):
		super().__init__()
		latent = synth.encoder.latent
		if use_pretrained_encoder:
			# the learned EEG->fMRI mapping: the synthesizer's own encoder (attention included) and RFF.
			# Requires inputs in the synthesizer's montage/shape (harmonize EEG to the NODDI montage first).
			if tuple(eeg_shape) != tuple(synth.encoder.in_shape):
				raise ValueError(f"pretrained encoder expects {synth.encoder.in_shape}, got {eeg_shape}; harmonize the montage")
			self.encoder = copy.deepcopy(synth.encoder)
			self.rff = copy.deepcopy(synth.decoder.rff)
			self.rff.scale.requires_grad_(False)
		else:
			if strides is None:
				strides = {132: ((3, 5, 1), (5, 2, 1))}.get(eeg_shape[0], ((3, 5, 1), (2, 2, 1)))
			self.encoder = EEGEncoder(eeg_shape, latent, strides=strides, attention=False, dropout=0.5)
			for m in self.encoder.modules():  # pretrained_ResBlock(trainable=True) / Dense: GlorotUniform biases
				if isinstance(m, (nn.Conv3d, nn.Linear)) and m.bias is not None:
					_glorot_vector_(m.bias)
			self.rff = RandomFourierFeatures(math.prod(latent), math.prod(latent), scale=float(synth.decoder.rff.scale.detach()), trainable_scale=False)
		if freeze_encoder:
			for p in self.encoder.parameters():
				p.requires_grad_(False)
		self.style = nn.Parameter(synth.decoder.style.detach().clone(), requires_grad=False)
		self.dense = synth.decoder.dense
		self.outfilter = synth.decoder.outfilter
		for p in list(self.dense.parameters()) + list(self.outfilter.parameters()):
			p.requires_grad_(False)
		self.out_shape = synth.decoder.out_shape

	def forward(self, eeg):
		s = self.rff(self.encoder(eeg))
		v = self.dense(torch.cos(s) * self.style)
		return self.outfilter(v.view(-1, 1, *self.out_shape)), s


def legacy_contrastive_loss(s1, s2, same, m=0.5):
	"""ContrastiveLoss(m=0.5) on |s1 - s2| where s = (sinusoid pre-activation - pi/2) / ||S||_batch,
	the norm taken over the whole batch tensor without gradient (as s1.numpy() in TF)."""
	a = s1 - math.pi / 2
	b = s2 - math.pi / 2
	a = a / a.detach().norm()
	b = b / b.detach().norm()
	d = (a - b).abs()
	same = same.float().unsqueeze(-1)
	return (same * d + (1 - same) * (d - m).abs()).mean(dim=1)
