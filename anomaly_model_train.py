import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import read_image
from anomalib.pre_processing.transforms import Denormalize
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks

from anomalib.models.components import DynamicBufferModule, FeatureExtractor, KCenterGreedy
from anomalib.models.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from data import MedDataset, MedDataset_png, MNIST, MvtecDatasetSR
import yaml
from medpy.io import load
from medpy.io import header
import glob
import timm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import idx2numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


if __name__ == "__main__":

    mode = 'mvtec'
    with open('config.yaml') as file:
        config_mri = yaml.load(file, Loader=yaml.FullLoader)
    MODEL = "patchcore"  # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
    CONFIG_PATH = './anomalib/' + f"src/anomalib/models/{MODEL}/config.yaml"
    # pass the config file to model, callbacks and datamodule
    config = get_configurable_parameters(config_path=CONFIG_PATH)

    config_data = {
    'mnist_path': './MNIST/raw/train-images-idx3-ubyte',
    'mnist_labels_path': './MNIST/raw/train-labels-idx1-ubyte',
    'mnist_test_path': './MNIST/raw/t10k-images-idx3-ubyte',
    'mnist_labels_test_path': './MNIST/raw/t10k-labels-idx1-ubyte'
    }

    if mode == 'mnist':
    #load mnist
        images = idx2numpy.convert_from_file(config_data['mnist_path'])
        labels = idx2numpy.convert_from_file(config_data['mnist_labels_path'])
        images_test = idx2numpy.convert_from_file(config_data['mnist_test_path'])
        labels_test = idx2numpy.convert_from_file(config_data['mnist_labels_test_path'])


    elif mode == 'mri':
        mri_files = '/home/BRATS_png/normal/*flair.png'
        mri_files = glob.glob(mri_files)
        mri_files_test = '/home/BRATS_png/tumor/*flair.png'
        mri_files_test = glob.glob(mri_files_test)

        np.random.seed(42)
        np.random.shuffle(mri_files)
        np.random.shuffle(mri_files_test)

        mri_files_test = mri_files_test[:100]
        print(len(mri_files), len(mri_files_test))

    if mode == 'mnist':
        train_dataset = MNIST(config_data, images, labels, num=1, train=False, max_file=300)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    elif mode == 'mvtec':
        print("MVTEC data")
        obj = 'pill'
        train_files = './mvtec/{obj}/*/good/*.png'.format(obj=obj)
        exceptions = [] #['bottle', 'bottle2', 'leather', 'zipper']
        #test_files = './mvtec/{obj}/test/{defect}/*.png'.format(obj=obj, defect=defect)

        train_files = glob.glob(train_files)
        if len(exceptions) > 0:
            train_files_filtered = []
            for i in range(len(train_files)):
                if train_files[i].split('/')[-4] in exceptions:
                    continue
                train_files_filtered.append(train_files[i])
        #test_files = glob.glob(test_files)
        
        else:
            train_files_filtered= train_files

        np.random.seed(42)
        np.random.shuffle(train_files)
        np.random.shuffle(train_files_filtered)

        print(len(train_files), len(train_files_filtered))

        train_dataset = MvtecDatasetSR(train_files_filtered, train=True, denoise=False, max_num=1000)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        #test_dataset = MvtecDatasetSR(test_files, train=False, mode=str(defect), denoise=False)
        #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        train_dataset = MedDataset_png(config_mri, mri_files, train=True, tumor=False)
        test_dataset = MedDataset_png(config_mri, mri_files_test, train=False, tumor=True)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(len(train_loader))
    data = next(iter(train_loader))
    print(data[0].shape, data[1].shape)
    print(data[1].min(), data[1].max())
    print(data[0].min(), data[0].max())

    backbone = "wide_resnet50_2" #"wide_resnet50_2"
    if 'resnet' in backbone:
        layers = ['layer2', 'layer3']
    else:
        layers = [1, 2]
    if mode == 'mnist':
        patchcore = PatchcoreModel(input_size = [84, 84], layers = layers,backbone= backbone, pre_trained= True, num_neighbors= 9) #wide_resnet50_2
    elif  mode == 'mvtec':
        patchcore = PatchcoreModel(input_size = [224, 224], layers = layers,backbone= backbone, pre_trained= True, num_neighbors= 9)
    else:
        patchcore = PatchcoreModel(input_size = [224, 224], layers = layers,backbone= backbone, pre_trained= True, num_neighbors= 9) #wide_resnet50_2
    patchcore.training = True


    embeddings = []
    repeat = 1

    for _ in range(repeat):
        for i, data in enumerate(train_loader):
            if len(data) == 3:
                input, _, _ = data
            else:
                input, _, *_ = data
            if input.shape[1] != 3:
                input = input.repeat(1, 3, 1, 1)
            if mode != 'mri':
                if input.max() > 1.0:
                    input = input/2.0
            #normalize input using imagenet stats
            input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            input = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225])(input)
            input = input.to(device)
                
            embedding = patchcore(input)

            embeddings.append(embedding)

    print("All features extracted.")
    embeddings = torch.vstack(embeddings)

    print("Applying core-set subsampling to get the embedding.")
    patchcore.subsample_embedding(embeddings, 0.1)
    print("Done.")
    if obj == '*':
        obj = 'all'
    if mode == 'mnist':
        np.save('memory_bank_mnist_train.npy', patchcore.memory_bank.cpu().numpy()) 
    elif mode == 'mvtec':
        np.save('memory_bank_mvtec_{}.npy'.format(obj), patchcore.memory_bank.cpu().numpy())
    elif mode == 'mri':
        np.save('memory_bank_mri_flair_train.npy', patchcore.memory_bank.cpu().numpy())
