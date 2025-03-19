import torch
import torch.nn as nn
import torchvision.models as models


class MixedInputModel(nn.Module):
    """
    Neural network that processes both image and tabular data
    
    Parameters:
        num_tabular_features : int
            Number of tabular features
        num_classes : int
            Number of output classes
        tabular_hidden_dims : list
            List of hidden dimensions for tabular MLP
        dropout_rate : float
            Dropout rate
        pretrained : bool
            Whether to use pretrained weights for the image model
    """
    def __init__(self, num_tabular_features, num_classes=2, 
                 tabular_hidden_dims=[64, 32], pretrained=True):
        super(MixedInputModel, self).__init__()
        
        # Image feature extractor (ResNet18)
        # Use weights='IMAGENET1K_V1' instead of pretrained=True (which is deprecated)
        self.image_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        num_image_features = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity()  # Remove final FC layer
        
        # Tabular feature processor (MLP)
        tabular_layers = []
        input_dim = num_tabular_features
        
        for hidden_dim in tabular_hidden_dims:
            tabular_layers.append(nn.Linear(input_dim, hidden_dim))
            tabular_layers.append(nn.BatchNorm1d(hidden_dim))
            tabular_layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        self.tabular_model = nn.Sequential(*tabular_layers)
        
        # Combined classifier
        combined_dim = num_image_features + tabular_hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)  # Changed from dim=0 to dim=1 for proper batch handling
        )
        
    def forward(self, image, tabular):

        # Process image
        image_features = self.image_model(image)
        
        # Process tabular data
        tabular_features = self.tabular_model(tabular)
        
        # Combine features
        combined = torch.cat((image_features, tabular_features), dim=1)
        
        # Classification
        output = self.classifier(combined)
        
        return output
    
    def reset_parameters(self):
        """Reset model parameters"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    