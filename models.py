import torch
import torch.nn as nn
import torchvision.models as models


class BaseModel(nn.Module):
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
                 tabular_hidden_dims=[512, 128], pretrained=True):
        super(BaseModel, self).__init__()
        
        # Image feature extractor 
        self.image_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        num_image_features = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity()  # Remove final FC layer
        
        # Tabular feature processor (MLP)
        tabular_layers = []
        input_dim = num_tabular_features
        
        # First layer
        tabular_layers.append(nn.Linear(input_dim, 512))
        tabular_layers.append(nn.BatchNorm1d(512))
        tabular_layers.append(nn.SiLU())  # Swish activation (SiLU in PyTorch)
        tabular_layers.append(nn.Dropout(p=0.3))
        
        # Second layer
        tabular_layers.append(nn.Linear(512, 128))
        tabular_layers.append(nn.BatchNorm1d(128))
        tabular_layers.append(nn.SiLU())  # Swish activation
            
        self.tabular_model = nn.Sequential(*tabular_layers)
        
        # Combined classifier
        combined_dim = num_image_features + tabular_hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(combined_dim, num_classes),
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


class SexBasedModel(nn.Module):
    """
    Neural network that processes both image and tabular data, specializing in male and female models separately.
    The model uses the sex feature to choose the appropriate image classifier.
    
    Parameters:
        num_tabular_features : int
            Number of tabular features (including sex)
        num_classes : int
            Number of output classes
        tabular_hidden_dims : list
            List of hidden dimensions for tabular MLP
        pretrained : bool
            Whether to use pretrained weights for the image model
    """
    def __init__(self, num_tabular_features, num_classes=2, 
                 tabular_hidden_dims=[512, 128], pretrained=True):
        super(SexBasedModel, self).__init__()
        
        # Male and female image feature extractors
        self.male_image_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.female_image_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        num_image_features = self.male_image_model.fc.in_features
        self.male_image_model.fc = nn.Identity()  # Remove final FC layer
        self.female_image_model.fc = nn.Identity()  # Remove final FC layer
        
        # Tabular feature processor (MLP) - using all features except sex
        # We assume sex is the last feature in the tabular data
        tabular_layers = []
        input_dim = num_tabular_features - 1  # Excluding sex feature
        
        # First layer
        tabular_layers.append(nn.Linear(input_dim, 512))
        tabular_layers.append(nn.BatchNorm1d(512))
        tabular_layers.append(nn.SiLU())  # Swish activation (SiLU in PyTorch)
        tabular_layers.append(nn.Dropout(p=0.3))
        
        # Second layer
        tabular_layers.append(nn.Linear(512, 128))
        tabular_layers.append(nn.BatchNorm1d(128))
        tabular_layers.append(nn.SiLU())  # Swish activation
            
        self.tabular_model = nn.Sequential(*tabular_layers)
        
        # Combined classifier
        combined_dim = num_image_features + tabular_hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(combined_dim, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, image, tabular):
        # Split tabular data - assuming sex is the last column
        # (0 for male, 1 for female in a binary encoding)
        sex = tabular[:, -1:]
        tabular_without_sex = tabular[:, :-1]
        
        # Process tabular data (excluding sex)
        tabular_features = self.tabular_model(tabular_without_sex)
        
        # Process image based on sex
        batch_size = image.size(0)
        image_features = torch.zeros(batch_size, self.male_image_model.fc.in_features).to(image.device)
        
        # For each sample in the batch, choose the appropriate model
        for i in range(batch_size):
            if sex[i, 0] < 0.5:  # Male
                image_features[i] = self.male_image_model(image[i].unsqueeze(0))
            else:  # Female
                image_features[i] = self.female_image_model(image[i].unsqueeze(0))
        
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
                
                
    