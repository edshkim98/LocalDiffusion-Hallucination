import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from anomalib.models.components import DynamicBufferModule, FeatureExtractor, KCenterGreedy
from anomalib.models.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler

import timm 
from sklearn import metrics
from torchvision import transforms
from data import MedDataset_png, MNIST, MvtecDatasetSR
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import glob
import idx2numpy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# classifier
# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten the tensor for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
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
        self.feature_extractor = self.feature_extractor#.to(device)
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
            if self.obj == 'pill':
                pretrained = np.load(f'/home/seunghki/mnist_az/memory_bank_mvtec_pill_hr.npy')
            else:
                pretrained = np.load(f'/home/seunghki/mnist_az/memory_bank_mvtec_all.npy')
        else:
            pretrained = np.load(f'/home/seunghki/mnist_az/memory_bank_mri_flair2t1.npy')
        patchcore.memory_bank = torch.from_numpy(pretrained)#.to(device)  

        self.patchcore = patchcore
        self.patchcore.memory_bank = self.patchcore.memory_bank.to(device)

        if self.threshold == None:
            self.create_testloader()
            self.calc_threshold()

    def create_testloader(self):
        
        if 'mvtec' in self.mode:
            test_files = f'/home/seunghki/mnist_az/mvtec/{self.obj}/test/*/*.png'
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
            mri_files2 = np.array(glob.glob(self.config['mri_files'].replace('tumor', 'normal')))
            np.random.shuffle(mri_files)
            np.random.shuffle(mri_files2)

            #split mri_files into train, validation and test in 70:15:15 ratio
            train_split = int(0.8 * len(mri_files))
            mri_files_test = mri_files[:train_split]

            self.test_dataset = MedDataset_png(self.config, mri_files2, train=False, tumor=False)
            self.test_dataset2 = MedDataset_png(self.config, mri_files_test, train=False, tumor=True)
            print(len(self.test_dataset), len(self.test_dataset2))
            #concatenate the two datasets
            self.test_dataset = ConcatDataset([self.test_dataset, self.test_dataset2])
            self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        print("Finished creating testloader.")
        if 'mvtec' in self.mode:
            print(len(test_files))
        else:
            print(len(mri_files_test) + len(mri_files2))
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
                input, _, cls = data
            else:
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
                input = input/4096.0
                if len(torch.unique(cls)) == 1:
                    cls = torch.tensor([0])
                else:
                    cls = torch.tensor([1])
            
            print(input.shape, self.patchcore.input_size)
            input = F.interpolate(input, size=(self.patchcore.input_size[0], self.patchcore.input_size[0]), mode='bilinear', align_corners=False)
            input = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225])(input)
            input = input.to(device)
            out = self.patchcore(input)
            anomaly_map, pred_score = out["anomaly_map"], out["pred_score"]
            input = F.interpolate(input, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

            inputs.append(input.cpu().numpy())
            scores.append(pred_score.cpu().numpy())
            if len(data) >= 3:
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
            hr = hr/4096.0
        hr = F.interpolate(hr, size=(self.patchcore.input_size[0], self.patchcore.input_size[0]), mode='bilinear', align_corners=False)
        hr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225])(hr)
        hr_out = self.patchcore(hr)
        anomaly_map, pred_score = hr_out["anomaly_map"], hr_out["pred_score"]
        anomaly_map = F.interpolate(anomaly_map, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        if pred_score > self.threshold:
            return 1, anomaly_map, pred_score
        return 0, anomaly_map, pred_score 