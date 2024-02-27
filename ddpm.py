import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import numpy as np
import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from attend import Attend
from data import MedDataset_png, MNIST, MvtecDatasetSR, MvtecDataset
import pandas as pd
import glob
from unet_model import ResUnet
import yaml
import idx2numpy

from sklearn import metrics
from torchvision.transforms import ToPILImage
from torch import Tensor
from torchvision import transforms
import timm

from anomalib.models.components import DynamicBufferModule, FeatureExtractor, KCenterGreedy
from anomalib.models.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler
# from denoising_diffusion_pytorch.version import __version__

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helpers functions

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        self_condition = False,
        cond_img = True,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (False, False, False, True),
        flash_attn = False,
        mode = 'mri'
    ):
        super().__init__()

        # determine dimensions
        self.mode = mode
        self.cond_model = ResUnet(data=self.mode)
        self.channels = channels
        self.self_condition = self_condition
        self.cond_img = cond_img
        input_channels = channels * (1 if self.cond_img else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        
        self.conv_fusion = block_klass(mid_dim*2, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
        
    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, cond_img, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        #if exists(cond_img):
        #    x = torch.cat((cond_img, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        cond_feat = self.cond_model(cond_img.to(torch.float))
        x = torch.cat((x, cond_feat),1)
        x = self.conv_fusion(x) 

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# classifier

class PatchcoreModel(DynamicBufferModule, nn.Module):
    """Patchcore Module."""

    def __init__(
        self,
        input_size,#: tuple[int, int],
        layers,#: list[str],
        backbone,#: str = "wide_resnet50_2",
        pre_trained,#: bool = True,
        num_neighbors#: int = 9,
    ):
        super().__init__()
        self.tiler = None #: Tiler | None = None
        
        self.training = True
        self.backbone = backbone
        self.layers = layers
        self.input_size = input_size
        self.num_neighbors = num_neighbors

        if 'resnet' in self.backbone:
            self.feature_extractor = FeatureExtractor(backbone=self.backbone, pre_trained=pre_trained, layers=self.layers)
        else:
            out_indices = layers
            self.feature_extractor = timm.models.efficientnet_b4(pretrained=True, features_only=True, out_indices=tuple(out_indices))
            self.layers = out_indices
        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        self.register_buffer("memory_bank", Tensor())
        self.memory_bank: Tensor

    def forward(self, input_tensor): # Tensor):# -> Tensor | dict[str, Tensor]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Tensor | dict[str, Tensor]: Embedding for training,
                anomaly map and anomaly score for testing.
        """
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            if 'resnet' not in self.backbone:
                features = {self.layers[layer]: self.feature_pooler(feature) for layer, feature in enumerate(features)}
            else:
                features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            output = embedding
        else:
            # apply nearest neighbor search
            patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
            # reshape to batch dimension
            patch_scores = patch_scores.reshape((batch_size, -1))
            locations = locations.reshape((batch_size, -1))
            # compute anomaly score
            pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
            # reshape to w, h
            patch_scores = patch_scores.reshape((batch_size, 1, width, height))
            #if patch_scores is gpu tensor, then convert it to cpu tensor
            if patch_scores.is_cuda:
                patch_scores = patch_scores.cpu()
            # get anomaly map
            anomaly_map = self.anomaly_map_generator(patch_scores)

            output = {"anomaly_map": anomaly_map, "pred_score": pred_score}

        return output

    def generate_embedding(self, features):#: dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: dict[str:Tensor]:

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding

    def subsample_embedding(self, embedding: Tensor, sampling_ratio: float) -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """

        # Coreset Subsampling
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()
        self.memory_bank = coreset

    @staticmethod
    def euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
        """
        Calculates pair-wise distance between row vectors in x and those in y.

        Replaces torch cdist with p=2, as cdist is not properly exported to onnx and openvino format.
        Resulting matrix is indexed by x vectors in rows and y vectors in columns.

        Args:
            x: input tensor 1
            y: input tensor 2

        Returns:
            Matrix of distances between row vectors in x and y.
        """
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        res = res.clamp_min_(0).sqrt_()
        return res

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int):# -> tuple[Tensor, Tensor]:
        """Nearest Neighbours using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        distances = self.euclidean_dist(embedding, self.memory_bank)
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(self, patch_scores: Tensor, locations: Tensor, embedding: Tensor) -> Tensor:
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores
        Returns:
            Tensor: Image-level anomaly scores
        """

        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = self.memory_bank.shape[0]  # edge case when memory bank is too small
        _, support_samples = self.nearest_neighbors(
            nn_sample, n_neighbors=min(self.num_neighbors, memory_bank_effective_size)
        )
        # 4. Find the distance of the patch features to each of the support samples
        distances = self.euclidean_dist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples])
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # s in the paper
        return score


class Classifier_PatchCore(nn.Module):
    def __init__(self, config, obj, threshold=None):
        super(Classifier_PatchCore, self).__init__()
        self.config = config
        self.mode = self.config['data']
        self.obj = obj
        self.threshold = threshold

        self.backbone = "wide_resnet50_2" #"wide_resnet50_2" seresne

        if 'resnet' in self.backbone:
            layers = ['layer2', 'layer3']
        else:
            layers = [1, 2]
            
        if 'mnist' in self.mode:
            patchcore = PatchcoreModel(input_size = [84, 84], layers = layers,backbone= self.backbone, pre_trained= True, num_neighbors= 9)
        else:
            patchcore = PatchcoreModel(input_size = [224, 224], layers = layers,backbone= self.backbone, pre_trained= True, num_neighbors= 9)
        patchcore.trianing=False

        if 'mnist' in self.mode:
            patchcore.load_state_dict(torch.load(f'patchcore_mnist_{self.obj}_hr.pth'))
        elif 'mvtec' in self.mode:
            pretrained = np.load(f'memory_bank_mvtec_all.npy')
        else:
            pretrained = np.load(f'memory_bank_brats_{obj}_hr.npy')
        patchcore.memory_bank = torch.from_numpy(pretrained).to(device)  

        self.patchcore = patchcore

        if self.threshold == None:
            self.create_testloader()
            self.calc_threshold()

    def create_testloader(self):
        
        if 'mvtec' in self.mode:
            test_files = f'./mvtec/{self.obj}/test/*/*.png'
            test_files = glob.glob(test_files)
            self.test_dataset = MvtecDatasetSR(test_files, train=False, mode=None, denoise=False)
            self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        elif self.mode == 'mnist':
            file_path = self.config['mnist_path']
            label_path = self.config['mnist_labels_path']
            
            np.random.seed(42)
            mri_files = idx2numpy.convert_from_file(file_path)
            mri_labels = idx2numpy.convert_from_file(label_path)
            self.test_dataset = MNIST(self.config, mri_files, mri_labels, train=False, num=[self.obj], max_file=100) 
            self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        else:
            print("Data: Brain")
            np.random.seed(42)
            mri_files = np.array(glob.glob(self.config['mri_files']))
            np.random.shuffle(mri_files)

            #split mri_files into train, validation and test in 70:15:15 ratio
            train_split = int(0.8 * len(mri_files))
            mri_files_test = mri_files[:train_split]

            self.test_dataset = MedDataset_png(self.config, mri_files_test, train=False, tumor=False)
            self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        print("Finished creating testloader.")
        print(len(test_files))
        print("Length of testloader: ", len(self.test_loader))

    def calc_threshold(self):
        scores= []
        inputs = []
        labels = []
        data = next(iter(self.test_loader))
        self.img_size = data[0].shape[-1]

        self.patchcore.training = False
        self.patchcore.feature_extractor.eval()
        self.patchcore.feature_extractor = self.patchcore.feature_extractor.to(device)
        for i, data in enumerate(self.test_loader):

            if len(data) == 3:
                print("Data is 3")
                input, _, cls = data
            else:
                print("Data is 4")
                input, _, cls, _ = data

            if input.shape[1] != 3:
                input = input.repeat(1, 3, 1, 1)
            if ('mvtec' in self.mode) or ('mnist' in self.mode):
                if input.max() > 1.0:
                    input = input/2.0
            else:
                if self.obj == 'flair':
                    mean = self.config['mean_flair']
                    std = self.config['std_flair']
                    mini = (0 - mean) / std
                else:
                    mean = self.config['mean_t1']
                    std = self.config['std_t1']
                    mini = (0 - mean) / std
                #denormalize the input
                input = input - mini
                input = input*std + mean
            
            print(input.shape, self.patchcore.input_size)
            input = F.interpolate(input, size=(self.patchcore.input_size[0], self.patchcore.input_size[0]), mode='bilinear', align_corners=False)
            input = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225])(input)
            input = input.to(device)
            out = self.patchcore(input)
            anomaly_map, pred_score = out["anomaly_map"], out["pred_score"]
            input = F.interpolate(input, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

            inputs.append(input.cpu())
            scores.append(pred_score.cpu().numpy())
            if len(data) > 3:
                labels.append(cls.numpy()+1)
        print("Finished testing.")

        #calculate optimal threshold for anomaly detection using scores and labels
        scores = np.concatenate(scores)
        labels = np.array(labels)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=2)
        diff = tpr - fpr
        diff_max = np.argmax(diff)
        #find the threshold where fprrate is 0
        i = np.where(fpr == 0.)[0][-1]
        i = diff_max
        self.threshold = thresholds[i]
        print("Threshold: ", self.threshold)

    def forward(self, hr):
        if hr.shape[1] != 3:
            hr = hr.repeat(1, 3, 1, 1)
        if ('mvtec' in self.mode) or ('mnist' in self.mode):
            if hr.max() > 1.0:
                hr = hr/2.0
        else:
            if self.obj == 'flair':
                mean = self.config['mean_flair']
                std = self.config['std_flair']
                mini = (0 - mean) / std
            else:
                mean = self.config['mean_t1']
                std = self.config['std_t1']
                mini = (0 - mean) / std
            #denormalize the input
            hr = hr - mini
            hr = hr*std + mean
        hr = F.interpolate(hr, size=(self.patchcore.input_size[0], self.patchcore.input_size[0]), mode='bilinear', align_corners=False)
        hr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225])(hr)
        hr = hr.to(device)
        hr_out = self.patchcore(hr)
        anomaly_map, pred_score = hr_out["anomaly_map"], hr_out["pred_score"]
        anomaly_map = F.interpolate(anomaly_map, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        if pred_score > self.threshold:
            return 1, anomaly_map, pred_score
        return 0, anomaly_map, pred_score    

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        config,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = False,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond
        
        self.config = config
        self.branch_out = self.config['branch_out']
        self.start_intermediate = self.config['start_intermediate']
        self.model = model
        self.classifier_flag = 0
        self.fusion_cnt = 0
        self.classifier_t = []

        self.lst = []
        self.cnt = -1

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def call_classifier(self):
        if self.config['classifier']:
            print("Classifier is being called")
            self.classifier = Classifier_PatchCore(self.config, obj=self.config['classifier_obj'], threshold=None)

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def gaussian_pdf(self, x, mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        return (1/(std * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((x - mean) / std)**2)
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, mask, min_max_val, cond_img, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, device=None):
        if self.config['branch_out']:
            #if branch_out, cond_img is originally a full condition, mask is probability-based mask
            #mask1 = torch.clip(mask, 0.2, 1.0).to(device)
            cond_out = cond_img * mask.to(device)
            self.cond_out = cond_out.to(torch.float32)
            #cond_out = F.interpolate(cond_out, scale_factor=5, mode='bilinear',align_corners=False)
            #mask_out =  F.interpolate(mask, scale_factor=5, mode='nearest')
            # mask = torch.where(mask < 0.9, 0.0, mask) #0.7
            binary_mask = (mask >= 1.0).float()
            if cond_img.is_cuda is not True:
                cond_img = cond_img.to(device)
            cond_out_binary = cond_img * binary_mask.to(device)
            self.cond_out = cond_out_binary.to(torch.float32)
            mask2 = (1. - binary_mask)
            if self.config['data'] == 'mnist':
                mask2 = torch.clip(mask2, 0.5, 1.0).to(device)
            else:
                mask2 = torch.clip(mask2, 0.95, 1.0).to(device)

            cond_in = cond_img * mask2
            cond_in = cond_in.to(torch.float32)
            cond_out = cond_out.to(torch.float32)

            np.save('lr_out.npy', cond_out.cpu().detach().numpy())
            np.save('lr_in.npy', cond_in.cpu().detach().numpy())
            #binarize the mask
            mask = (mask >= 1.0).to(torch.float32)
            x_out, x_in = x[0], x[1]
            #x_out_up =  F.interpolate(x_out, scale_factor=5, mode='bilinear',align_corners=False)
            model_output_out = self.model(x_out, cond_out, t, x_self_cond)
            #model_output_out = F.interpolate(model_output_out, scale_factor=0.2, mode='bilinear',align_corners=False)
            model_output_in = self.model(x_in, cond_in, t, x_self_cond)

            if self.config['mask_x']:
                assert len(torch.unique(mask)) == 2, 'mask should be binary {}'.format(torch.unique(mask))

                model_output_out = model_output_out.cpu()*mask.cpu()

                model_output_out = torch.where(mask.cpu() == 0., torch.tensor(min_max_val[0]), model_output_out)
                model_output_out = model_output_out.to(device) 
                if ('mnist' in self.config['data']) or ('mvtec' in self.config['data']):
                    if ('mri' in self.config['data']): #('transistor' in self.config['mvtec_path']) or ('mri' in self.config['data']):
                        model_output_out = model_output_out
                    else:
                        model_output_out = 0.5*model_output_out.cpu() + 0.5*self.cond_out.cpu() 
                
                mask2 = 1. - mask
                #if self.t == self.config['continue_fusion_timestep']:
                   # model_output_in = model_output_in.cpu()*mask2.cpu()
                model_output_in = model_output_in.to(device)

                if self.t == 0:
                    np.save('pred_out.npy', model_output_out.cpu().detach().numpy())
                    np.save('pred_in.npy', model_output_in.cpu().detach().numpy())

        else:
            model_output = self.model(x, cond_img, t, x_self_cond)
            if not self.branch_out:
                if self.t == self.num_timesteps:
                    if self.config['mask_x']:
                        assert len(torch.unique(mask)) == 2, 'mask should be binary'
 
                        model_output = torch.where(mask.cpu() == 0., torch.tensor(min_max_val[0]), model_output)
                        model_output = model_output.to(device)     

        if self.config['branch_out']:
            maybe_clip_out = partial(torch.clamp, min = min_max_val[0], max = min_max_val[1]) if clip_x_start else identity
            maybe_clip_in = partial(torch.clamp, min = min_max_val[0], max = min_max_val[1]) if clip_x_start else identity
        else:
            maybe_clip = partial(torch.clamp, min = min_max_val[0], max = min_max_val[1]) if clip_x_start else identity          

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            if self.config['branch_out']:
                x_start_out = model_output_out
                x_start_out = maybe_clip_out(x_start_out)
                x_start_out = x_start_out.cuda()
                #print(x_start_out.is_cuda, t.is_cuda, x_out.is_cuda)
                #print(x_start_out.cuda().is_cuda, t.is_cuda, x_out.is_cuda)
                pred_noise_out = self.predict_noise_from_start(x_out, t, x_start_out)

                x_start_in = model_output_in
                x_start_in = maybe_clip_in(x_start_in)
                x_start_in = x_start_in.cuda()
                pred_noise_in = self.predict_noise_from_start(x_in, t, x_start_in)

            else:
                x_start = model_output
                x_start = maybe_clip(x_start)
                x_start = x_start.cuda()
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        if self.config['branch_out']:
            return ModelPrediction(pred_noise_out, x_start_out), ModelPrediction(pred_noise_in, x_start_in)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, mask, min_max_val, cond_img, t, x_self_cond = None, clip_denoised = True, device = None):
        if self.config['branch_out']:
            preds_out, preds_in = self.model_predictions(x, mask, min_max_val, cond_img, t, x_self_cond, device=device)
            x_start_out = preds_out.pred_x_start
            x_start_in = preds_in.pred_x_start
            if clip_denoised:

                x_start_out.clamp_(min_max_val[0],  min_max_val[1])#cond_img.max())      
                x_start_in.clamp_(min_max_val[0], min_max_val[1])

            if (self.t <= self.config['start_timestep']) and (self.config['start_intermediate']):
                self.config['branch_out'] = False
                self.config['mask_x'] = False
                
                #binarize the mask
                mask = (mask >= 1.).to(torch.float32)
                x_start_in_mask = x_start_in * (1. - mask.to(device))
                x_start = x_start_in_mask + x_start_out #torch.where(mask == 0., x_start_in_mask, x_start_out)

                #binarize the mask
                #mask = (mask >= 1.).to(torch.float32)

                x_out = x[0] * mask.to(device)
                x_in = x[1] * (1. - mask.to(device))
                assert torch.any((x_out == 0.)) and torch.any((x_in == 0.)), 'x_out and x_in should be masked'
                self.x_branchout = [x_out, x_in]
                x = torch.where(x_out == 0., x_in, x_out)
                self.x = x
                self.x_start_skip = x_start.clone()
                self.x_start_skip_mask = torch.where(self.x_start_skip == 0., 0., 1)

                if clip_denoised:
                    x_start.clamp_(min_max_val[0], min_max_val[1])
                if self.viz:
                    np.save('pred_in_'+str(self.cnt)+'.npy', x_start_in.cpu().detach().numpy())
                    np.save('pred_out_'+str(self.cnt)+'.npy', x_start_out.cpu().detach().numpy())
                    np.save('pred_concat_'+str(self.cnt)+'.npy', x_start.cpu().detach().numpy())
                model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
                return model_mean, posterior_variance, posterior_log_variance, x_start
            
            model_mean_out, posterior_variance_out, posterior_log_variance_out = self.q_posterior(x_start = x_start_out, x_t = x[0], t = t)
            model_mean_in, posterior_variance_in, posterior_log_variance_in = self.q_posterior(x_start = x_start_in, x_t = x[1], t = t)

            return (model_mean_out, model_mean_in), (posterior_variance_out, posterior_variance_in), (posterior_log_variance_out, posterior_log_variance_in), (x_start_out, x_start_in)
        else:
            preds = self.model_predictions(x, mask, min_max_val, cond_img, t, x_self_cond, device=device)
            x_start = preds.pred_x_start
            #self.x_branchout_fusion = [x_start * mask.to(device), x_start * (1. - mask).to(device)]
        
            if clip_denoised:
                x_start.clamp_(min_max_val[0], min_max_val[1])#(-1., 1.)

            if (self.config['branch_out'] == False) and (self.config['start_intermediate']) and (self.branch_out):
                if self.config['data'] == 'mnist':
                    coeff =  (1/((self.config['continue_fusion_timestep'])*2)) * self.t + 0.5
                elif 'mvtec' in self.config['data']:
                    if 'transistor' not in self.config['mvtec_path']:
                        coeff = 0
                    else:
                        coeff = torch.zeros(x_start.shape[0], 1, x_start.shape[-1], x_start.shape[-1])
                        coeff = torch.where(self.cond_out == 1., 0.5, 0.)
# #(1/((self.config['continue_fusion_timestep'])*2)) * self.t
                else:
                    coeff = 0 #(1/((self.config['continue_fusion_timestep'])*2)) * self.t
                #x_start = coeff * self.cond_out + (1. - coeff) * x_start
                x_start = x_start.clamp_(min_max_val[0], min_max_val[1])

            
            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, mask, min_max_val, cond_img, t: int, x_self_cond = None):
        self.x = x
        if self.config['branch_out']:
            sample_img = x[0]
            b, *_, device = *sample_img.shape, self.device
        else:
            b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, mask = mask, min_max_val = min_max_val, cond_img = cond_img, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True,  device = device)
        if self.config['branch_out']:
            noise = torch.randn_like(x[0]) if t > 0 else 0. # no noise if t == 0
            pred_img_out = model_mean[0] + (0.5 * model_log_variance[0]).exp() * noise
            pred_img_in = model_mean[1] + (0.5 * model_log_variance[1]).exp() * noise
            return [pred_img_out, pred_img_in], x_start
        else:
            noise = torch.randn_like(self.x) if t > 0 else 0. # no noise if t == 0
            pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
            #if (self.t == 0) and (self.branch_out == True) and (self.start_intermediate ==True):
            #    pred_img = pred_img * self.x_start_skip_mask
            return pred_img, x_start

    def branching_out(self, img, x_start, imgs, x_start_lst):

        # print("Branching out at timestep: ", self.t)
        imgs.append([img[0].cpu(), img[1].cpu()])
        if self.t == 0:
            np.save('pred_out2.npy', img[0].cpu().detach().numpy())
            np.save('pred_in2.npy', img[1].cpu().detach().numpy())
        x_start_lst.append([x_start[0].cpu(), x_start[1].cpu()])
        self.branch_cnt = 1

        return img, x_start, imgs, x_start_lst
    
    def fusion(self, img, x_start, imgs, x_start_lst, mask, min_max_val, cond_img, self_cond = None):
        # print("Fusion at timestep: ", self.t)

        if (self.branch_cnt == 1) or (self.branch_out == False):
            imgs.append(img)
            x_start_lst.append(x_start.cpu())

            self.branch_cnt = 0
        else:
            if self.config['classifier']:
                #Predictor to classify if x_start is fake or not 
                if self.classifier_flag == 0:
                    pred_cls, _, _ = self.classifier(x_start)
                    self.pred_cls = pred_cls
                if (self.pred_cls > 0.0) or (self.t ==0):
                    if self.classifier_flag == 0:
                        print("Classified as correct (Anomaly)")
                    
                    print("Continue Fusion at timestep: ", self.t)
                    self.config['branch_out'] = False
                    imgs.append(img)

                    x_start_lst.append(x_start.cpu())

                    self.branch_cnt = 0
                    self.classifier_flag = 1
                    self.fusion_cnt +=1
                    print(f"Pred accepted at time {self.t}")
                    self.classifier_t[self.cnt] = self.t
                    np.save('fusion_time.npy', np.array(self.classifier_t))
                    
                else:
                    print("Classified as incorrect (Normal - Hallucinated)")
                    self.config['branch_out'] = True
                    self.config['mask_x'] = True
                   
                    #if self.fusion_cnt == 1:
                    img, x_start = self.p_sample(self.x_branchout, mask, min_max_val, cond_img, self.t, self_cond)
                    #else:
                    #    print("x_branchout_fusion")
                    #    img, x_start = self.p_sample(self.x_branchout_fusion, mask, min_max_val, cond_img, self.t, self_cond)
                    #np.save('pred_concat2_'+str(self.cnt)+'.npy', img.cpu().detach().numpy())
                    imgs.append(img)
                    x_start_lst.append(x_start.cpu())

                    self.branch_cnt = 1
                    self.fusion_cnt = 0
            else:
                if (self.t < self.config['start_timestep']) and (self.t > self.config['continue_fusion_timestep']) and (self.config['start_intermediate']) and (self.config['use_gt'] == False):
                    self.config['branch_out'] = True #if incorrect, repeat branching out process at the same timestep
                    self.config['mask_x'] = True
                    img, x_start = self.p_sample(self.x_branchout, mask, min_max_val, cond_img, self.t, self_cond)
                    #np.save('pred_concat2_'+str(self.cnt)+'.npy', img.cpu().detach().numpy())
                    imgs.append(img)
                    x_start_lst.append(x_start.cpu())

                    self.branch_cnt = 1
                else:
                    print("Continue Fusion at timestep: ", self.t)
                    self.config['branch_out'] = False
                    imgs.append(img)
                    # if self.t == 1:
                    #     np.save('pred_concat_'+str(self.cnt)+'.npy', x_start.cpu().detach().numpy())
                    x_start_lst.append(x_start.cpu())

                    self.branch_cnt = 0

        return img, x_start, imgs, x_start_lst

    @torch.inference_mode()
    def p_sample_loop(self, cond_img, mask, min_max_val, shape, return_all_timesteps = False, return_all_outputs = False):
        print("PERFORM DDPM!")
        batch, device = shape[0], self.device
        
        torch.manual_seed(42)      
        img = torch.randn(shape, device = device)

        if self.start_intermediate:

            if self.config['use_gt']:
                #t starts in reverse order: e.g. 9 . 8 .... 0
                t = torch.tensor(self.config['use_gt_timestep'], device=device).long() #torch.randint(0, self.num_timesteps, (batch,), device=device).long()
                t = torch.stack([t for _ in range(batch)], dim=0)
                img = self.q_sample(x_start = self.hr, t = t, noise = img)
                self.num_timesteps = self.config['use_gt_timestep']

        imgs = [img]
        confidence_map = []
        x_start_lst = []
        x_start = None
        self.branch_cnt = 0
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps, disable=True):
            self.t = t

            self_cond = x_start if self.self_condition else None
            if self.config['branch_out']:
                if self.t == self.num_timesteps-1:
                    img = [img, img]                
            img, x_start = self.p_sample(img, mask, min_max_val, cond_img, self.t, self_cond)
            if self.config['branch_out']:
                img, x_start, imgs, x_start_lst = self.branching_out(img, x_start, imgs, x_start_lst)
            else:
                img, x_start, imgs, x_start_lst = self.fusion(img, x_start, imgs, x_start_lst, mask, min_max_val, cond_img, self_cond = None)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        if not self.start_intermediate:
            if self.branch_out:
                if isinstance(ret, list):
                    ret = torch.stack(ret, dim=0)
                else:
                    ret = torch.stack((ret, ret), dim=0)

        ret = self.unnormalize(ret)
        if return_all_outputs:
            return ret, x_start_lst, confidence_map
        print("Return shape: ", ret.shape)
        print("Return min {} max {}".format(torch.min(ret), torch.max(ret)))
        return ret

    @torch.inference_mode()
    def ddim_sample(self, cond_img, mask, min_max_val, shape, return_all_timesteps = False):
        print("PERFORM DDIM!")
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        if self.config['branch_out']:
            imgs = [img, img]
        else:
            imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            self.t = time
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None

            if self.config['branch_out']:
                if self.t == self.num_timesteps-1:
                    img = [img, img]
                preds_out, preds_in = self.model_predictions(img, mask, min_max_val, cond_img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)
                x_start_out = preds_out.pred_x_start
                x_start_in = preds_in.pred_x_start

                if time_next < 0:
                    img = [x_start_out, x_start_in]
                    imgs.append(img)
                    continue
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img[0])

                if (self.t <= self.config['start_timestep']) and (self.config['start_intermediate']):
                    self.config['branch_out'] = False
                    self.config['mask_x'] = False
                    x_start = torch.where(x_start_out == 0., x_start_in, x_start_out) #x_start_out + x_start_in
                    x_start = x_start.clamp_(min_max_val[0], min_max_val[1])
                    self.x_start_skip = x_start.clone()

                    #binarize the mask
                    mask = (mask >= 1.).to(torch.float32)

                    x_out = preds_out.pred_noise * mask.to(device)
                    x_in = preds_in.pred_noise * (1. - mask.to(device))
                    assert torch.any((x_out == 0.)) and torch.any((x_in == 0.)), 'x_out and x_in should be masked'
                    pred_noise = torch.where(x_out == 0., x_in, x_out)

                    if self.viz:
                        np.save('pred_concat_'+str(self.cnt)+'.npy', x_start.cpu().detach().numpy())
                    img = x_start * alpha_next.sqrt() + \
                        c * pred_noise + \
                        sigma * noise
                else:
                    img = [x_start_out * alpha_next.sqrt() + c * preds_out.pred_noise + sigma * noise, x_start_in * alpha_next.sqrt() + c * preds_in.pred_noise + sigma * noise]

            else:
                pred_noise, x_start, *_ = self.model_predictions(img, mask, min_max_val, cond_img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

                np.save('pred_noise.npy', pred_noise.cpu().detach().numpy())

                # if (self.t < self.config['continue_fusion_timestep']) and (self.config['start_intermediate']) and (self.branch_out):
                #     if self.config['data'] == 'mnist':
                #         coeff =  (1/((self.config['continue_fusion_timestep'])*2)) * self.t + 0.5
                #     elif 'mvtec' in self.config['data']:
                #         if 'transistor' not in self.config['mvtec_path']:
                #             coeff = 0
                #         else:
                #             coeff = 0 #(1/((self.config['continue_fusion_timestep'])*2)) * self.t
                #     else:
                #         coeff = 0 #(1/((self.config['continue_fusion_timestep'])*2)) * self.t
                #     x_start = coeff * self.x_start_skip + (1. - coeff) * x_start
                if self.branch_out:
                    x_start = x_start.clamp_(self.min_max_val[0], self.min_max_val[1])

                if time_next < 0:
                    img = x_start
                    imgs.append(img)
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)

                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, cond_img, gt, batch_size = 16, return_all_timesteps = False, return_all_outputs = False,  mask = None, ood_confidence_ad=False, min_max_val = None):
        self.cnt += 1
        self.classifier_t.append(self.num_timesteps)
        self.min_max_val = min_max_val
        #print("MIN {} MAX {}".format(min_max_val[0], min_max_val[1]))
        if self.config['ood_AD']:
            if len(torch.unique(mask)) != 1:
                self.viz = True
            else:
                self.viz = False
        else:
            self.viz = False
        print("CNT: ", self.cnt)
        if self.config['branch_out'] == False:
            self.config['branch_out'] = self.branch_out
        
        if self.config['start_intermediate'] == False:
            self.config['start_intermediate'] = self.start_intermediate

        if self.config['start_intermediate']:
            self.start_intermediate = True
            if self.config['use_gt']:
                self.hr = gt
        else:
            self.start_intermediate = False

        if (self.config['ood_AD'] == True) or (self.config['ood_confidence'] == True):
            self.config['mask_cond'] = True
            self.config['mask_x'] = True

        if self.config['branch_out']:
            if len(torch.unique(mask)) == 1:
                if torch.unique(mask) == 1:
                    print("Original reverse process as AD is low")
                    self.config['mask_cond'] = False
                    self.config['mask_x'] = False
                    self.config['branch_out'] = False
                    self.config['start_intermediate'] = False
                
        print("Branch: {} | Start Intermediate: {} | Mask Cond: {} | Mask X: {}".format(self.config['branch_out'], self.config['start_intermediate'], self.config['mask_cond'], self.config['mask_x']))

        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.is_ddim_sampling:
            return sample_fn(cond_img, mask, min_max_val, (batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)
        return sample_fn(cond_img, mask, min_max_val, (batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps, return_all_outputs = return_all_outputs)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond_img, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, cond_img, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, cond_img, train, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        if train:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        else:
            torch.random.manual_seed(42)
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, cond_img, t, *args, **kwargs)

# dataset classes

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer(object):
    def __init__(
        self,
        configs,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 500,
        num_samples = 25,
        results_folder = './results/',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        calculate_fid = False,
        max_grad_norm = 1.,
        save_best_and_latest_only = False
    ):
        super().__init__()
        
        self.config = configs
        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        if self.config['data'] == 'mri':
            # dataset and dataloader (if png, mri_files is a list of flair png files)
            mri_files = self.config['mri_files']
            mri_files = np.array(glob.glob(mri_files))
            #shuffle mri_files
            np.random.seed(42)
            np.random.shuffle(mri_files)
            #split mri_files into train, validation and test in 70:15:15 ratio
            train_split = int(0.8 * len(mri_files))
            mri_files_train = mri_files[:train_split]
            mri_files_test = mri_files[train_split:]
        
            if self.config['train']:  
                self.ds = MedDataset_png(self.config, mri_files_train, train=True) 
                self.ds_test = MedDataset_png(self.config, mri_files_test, train=False)

                dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
                dl_test = DataLoader(self.ds_test, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = cpu_count())

                data = next(iter(dl))
                data2 = next(iter(dl_test))
                print("Train: {} Vali: {}".format(len(dl), len(dl_test)))
                print("Train shape: {} {} Vali shape: {} {}".format(data[0].shape, data[1].shape, data2[0].shape, data2[1].shape))
                print("Min: {} Max: {}".format(data[0].min(), data[0].max()))

        elif self.config['data'] == 'mnist':
            # dataset and dataloader
            mri_files = self.config['mnist_path']
            mri_files = idx2numpy.convert_from_file(mri_files).shape[0]
            mri_files = np.arange(mri_files)
            #shuffle mri_files
            np.random.seed(42)
            np.random.shuffle(mri_files)
            #split mri_files into train, validation and test in 70:15:15 ratio
            train_split = int(0.7 * len(mri_files))
        
            mri_files = idx2numpy.convert_from_file(self.config['mnist_path'])
            mri_files_labels = idx2numpy.convert_from_file(self.config['mnist_labels_path'])
            mri_files_train = mri_files[:train_split]
            mri_labels_train = mri_files_labels[:train_split]
            mri_files_test = mri_files[train_split:]
            mri_labels_test = mri_files_labels[train_split:]
        
            if self.config['train']:  
                self.ds = MNIST(self.config, mri_files_train, mri_labels_train, train=True, num=8, max_file=200) 
                self.ds_test = MNIST(self.config, mri_files_test, mri_labels_test, train=False, num=8, max_file=100)

                dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
                dl_test = DataLoader(self.ds_test, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = cpu_count())

                data = next(iter(dl))
                data2 = next(iter(dl_test))
                print("Train: {} Vali: {}".format(len(dl), len(dl_test)))
                print("Train shape: {} {} Vali shape: {} {}".format(data[0].shape, data[1].shape, data2[0].shape, data2[1].shape))
                print("Min: {} Max: {}".format(data[0].min(), data[0].max()))

        elif self.config['data'] == 'mvtec':
            # dataset and dataloader
            mri_files = self.config['mvtec_path']
            mri_files = np.array(glob.glob(mri_files))
            #shuffle mri_files
            np.random.seed(42)
            np.random.shuffle(mri_files)
            #split mri_files into train, validation and test in 70:15:15 ratio
            train_split = int(0.7 * len(mri_files))
            mri_files_train = mri_files[:train_split]
            mri_files_test = mri_files[train_split:]
            
            if self.config['train']:  
                self.ds = MvtecDataset(mri_files_train, train=True)  
                self.ds_test = MvtecDataset(mri_files_test, train=True)

                dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
                dl_test = DataLoader(self.ds_test, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = cpu_count())

                data = next(iter(dl))
                data2 = next(iter(dl_test))
                print("Train: {} Vali: {}".format(len(dl), len(dl_test)))
                print("Train shape: {} {} Vali shape: {} {}".format(data[0].shape, data[1].shape, data2[0].shape, data2[1].shape))
                print("Min: {} Max: {}".format(data[0].min(), data[0].max()))

        if self.config['train']:
             self.dl = dl
             self.dl_test = dl_test

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
        self.results_folder_string = results_folder+self.config['ProjectName']
        self.results_folder = Path(results_folder+self.config['ProjectName'])
        self.results_folder.mkdir(exist_ok = True)
        # Save the dictionary as a yaml file
        with open(self.results_folder_string+'/config.yaml', 'w') as yaml_file:
            yaml.dump(configs, yaml_file)
        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator except for self.classifier
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        
        self.best_ls = 1e10
        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

        self.df = pd.DataFrame(columns = ['epoch', 'loss'])
        self.df_train = pd.DataFrame(columns = ['epoch', 'loss'])
       
        if self.config['data'] == 'mri': 
            if not self.config['translate_zero']:
                self.max_val = 1.0 #(4096-self.config['mean_flair'])/self.config['std_flair']
                self.min_val = -1.0 #(0-self.config['mean_flair'])/self.config['std_flair']
            else:
                self.min_val2 = (0-self.config['mean_flair'])/self.config['std_flair'] 
                self.min_val = 0.
                self.max_val = (4096-self.config['mean_flair'])/self.config['std_flair']
                self.max_val = self.max_val + torch.abs(torch.tensor(self.min_val2))
        elif self.config['data'] == 'mnist':
            self.min_val = 0.
            self.max_val = 1.0
        elif 'mvtec' in self.config['data']:
            self.min_val = 0.
            self.max_val = 2.0
        self.min_max_val = (self.min_val, self.max_val) 

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
                            }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    #write a function that always returns a multiple of 1000
    def round_num(self, x, num=1000):
        return int(np.ceil(x/num))*num

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        with tqdm(initial = self.step, total = self.train_num_steps, disable = True) as pbar:
            
            while self.step < self.train_num_steps:

                total_loss = 0.

                # for _ in range(self.gradient_accumulate_every):
                for _, data in enumerate(self.dl):
                    hr, lr, _ = data
                    hr = hr.to(device)
                    lr = lr.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(hr,lr, train=True)
                        loss = loss / len(self.dl) #apply a whole dataloader batch #(self.gradient_accumulate_every*len(self.dl))
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()
              
                self.df_train = self.df.append({'epoch': self.step, 'loss': total_loss}, ignore_index=True)
                #save self.df to csv
                self.df_train.to_csv(self.results_folder_string+"/train_loss.csv")

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            lst = []
                            for _, data in enumerate(self.dl_test):
                                hr, lr, _ = data
                                hr = hr.to(device)
                                lr = lr.to(device)
                                #loss = self.model.sample(hr,lr, train=False)
                                out = self.ema.ema_model.sample(lr, batch_size=lr.shape[0], mask = None, return_all_timesteps = False, min_max = self.min_max_val)
                                #lst.append(loss.cpu().detach().numpy())
                                lst.append(torch.nn.MSELoss()(out, hr).cpu().detach().numpy())

                        ls = np.mean(np.array(lst))

                        if self.best_ls > ls:
                            self.best_ls = ls
                            if self.config['data'] == 'mnist':
                                train_phase = self.round_num(self.step, num=100)
                            elif self.config['data'] == 'mri':
                                train_phase = self.round_num(self.step, num=500)
                            elif self.config['data'] == 'mvtec':
                                train_phase = self.round_num(self.step, num=500)
                            self.save("best"+str(train_phase))
                            np.save(self.results_folder_string+"/hr.npy", hr.cpu())
                            np.save(self.results_folder_string+"/lr.npy", lr.cpu())
                            np.save(self.results_folder_string+"pred.npy", out.cpu().detach())

                        self.df = self.df.append({'epoch': self.step, 'loss': ls}, ignore_index=True)
                        self.df.to_csv(self.results_folder_string+"/loss.csv")
                pbar.update(1)

        accelerator.print('training complete')
