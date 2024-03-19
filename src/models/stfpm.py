from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

from src.models.stfpm_add.timm import FeatureExtractor
from src.models.stfpm_add.anomaly_map import AnomalyMapGenerator



def create_stfpm(strategy, img_shape, parameters):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    st = STFPM(device)
    st = st.to(device)
    return st, device  


class STFPM(nn.Module):
    """
    STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection.

    Args:
        layers (list[str]): Layers used for feature extraction
        input_size (tuple[int, int]): Input size for the model.
        backbone (str, optional): Pre-trained model backbone. Defaults to "resnet18".
    """
    def __init__(
        self, device,
        layers: list[str] = ["layer1", "layer2", "layer3"], 
        input_size: tuple[int, int] = (256,256),
        backbone: str = "resnet18", 
    ) -> None:
        super().__init__()

        self.training = True
        self.device = device
        self.backbone = backbone
        self.teacher = FeatureExtractor(backbone=self.backbone, pre_trained=True, layers=layers)
        self.student = FeatureExtractor(backbone=self.backbone, pre_trained=False, layers=layers, requires_grad=True)

        # teacher model is fixed
        for parameters in self.teacher.parameters():
            parameters.requires_grad = False

        self.teacher.eval()
        self.student.train()

        image_size = input_size
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=image_size)

    def forward(self, images: Tensor) -> Tensor | dict[str, Tensor] | tuple[dict[str, Tensor]]:
        """
        Forward-pass images into the network.

        During the training mode the model extracts the features from the teacher and student networks.
        During the evaluation mode, it returns the predicted anomaly map.

        Args:
          images (Tensor): Batch of images.

        Returns:
          Teacher and student features when in training mode, otherwise the predicted anomaly maps.
        """
        self.teacher.eval()
        if self.training == True:
            self.student.train()
        else:
            self.student.eval()

        #added 70
        with torch.no_grad():
            teacher_features: dict[str, Tensor] = self.teacher(images)

        student_features: dict[str, Tensor] = self.student(images)
        
        if self.training:
            output = teacher_features, student_features
        else:
            output = self.anomaly_map_generator(teacher_features=teacher_features, student_features=student_features)

        return output
        
            





