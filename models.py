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
    def __init__(self, num_tabular_features, num_classes=1, 
                 tabular_hidden_dims=[512, 128], pretrained=True):
        """
        Initialize the base model architecture.

        This model combines a ResNet-18 for image feature extraction with an MLP for 
        processing tabular data. The features from both sources are concatenated and 
        passed through a classifier to make the final prediction.

        Parameters:
            num_tabular_features : int
                Number of tabular features
            num_classes : int, default=1
                Number of output classes
            tabular_hidden_dims : list, default=[512, 128]
                List of hidden dimensions for tabular MLP
            pretrained : bool, default=True
                Whether to use pretrained weights for the image model
        """
        
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
            nn.Sigmoid()  # Changed from dim=0 to dim=1 for proper batch handling
        )
        
    def forward(self, image, tabular):
        """
        Forward pass through the neural network.

        Parameters:
            image : torch.Tensor
                Input images, shape (batch_size, channels, height, width)
            tabular : torch.Tensor
                Tabular features, shape (batch_size, num_tabular_features)
                
        Returns:
            torch.Tensor
                Model predictions, shape (batch_size, num_classes)
        """

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
    def __init__(self, num_tabular_features, num_classes=1, 
                 tabular_hidden_dims=[512, 128], pretrained=True):
        """
        Initialize the sex-based model architecture with separate pathways for male and female images.

        This model contains two separate ResNet-18 networks for processing images from male and female 
        subjects, and a shared tabular data processor. The model combines the image features with 
        processed tabular features to make the final prediction.

        Parameters:
            num_tabular_features : int
                Number of tabular features (including sex feature)
            num_classes : int, default=1
                Number of output classes
            tabular_hidden_dims : list, default=[512, 128]
                List of hidden dimensions for tabular MLP
            pretrained : bool, default=True
                Whether to use pretrained weights for the image models
        """
        super(SexBasedModel, self).__init__()
        
        # Male and female image feature extractors
        self.male_image_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.female_image_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        self.num_image_features = self.male_image_model.fc.in_features
        self.male_image_model.fc = nn.Identity()  # Remove final FC layer
        self.female_image_model.fc = nn.Identity()  # Remove final FC layer
        
        
        # Tabular feature processor (MLP) - using all features except sex
        # We assume sex is the last feature in the tabular data
        tabular_layers = []
        input_dim = num_tabular_features -1  # Excluding sex feature
        
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
        combined_dim = self.num_image_features + tabular_hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(combined_dim, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, image, tabular):
        """
        Forward pass for the sex-based model.

        The method processes both image and tabular data, routing each image to either
        the male or female image model based on the sex feature from tabular data.

        Parameters:
            image : torch.Tensor
                Batch of images to process (B, C, H, W)
            tabular : torch.Tensor
                Batch of tabular features (B, num_features), with sex as the last column
                
        Returns:
            torch.Tensor
                Model predictions (B, num_classes)
        """
        # Split tabular data - assuming sex is the last column
        # (0 for male, 1 for female in a binary encoding)
        sex = tabular[:, -1:]
        tabular_without_sex = tabular[:, :-1]
        
        # Process tabular data (excluding sex)
        tabular_features = self.tabular_model(tabular_without_sex)
        
        # Process image based on sex
        batch_size = image.size(0)
        image_features = torch.zeros(batch_size, self.num_image_features).to(image.device)
        
        # For each sample in the batch, choose the appropriate model
        for i in range(batch_size):
            if sex[i, 0] < 0.5:  # Male
                image_features[i] = self.male_image_model(image[i])
            else:  # Female
                image_features[i] = self.female_image_model(image[i])
        
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



class BrightnessBasedModel(nn.Module):
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
    def __init__(self, num_tabular_features, num_classes=1, 
                 tabular_hidden_dims=[512, 128], pretrained=True):  
        """
        Neural network that processes both image and tabular data, using different models for
        dark and light images based on brightness.

        Parameters:
            num_tabular_features : int
                Number of tabular features
            num_classes : int
                Number of output classes
            tabular_hidden_dims : list
                List of hidden dimensions for tabular MLP
            pretrained : bool
                Whether to use pretrained weights for the image model
        """
        super(BrightnessBasedModel, self).__init__()
        
        # Dark and Light image feature extractors
        self.dark_image_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.light_image_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        self.num_image_features = self.dark_image_model.fc.in_features
        self.dark_image_model.fc = nn.Identity()  # Remove final FC layer
        self.light_image_model.fc = nn.Identity()  # Remove final FC layer
        
        
        # Tabular feature processor (MLP) 
        # We assume sex is the last feature in the tabular data
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
        combined_dim = self.num_image_features + tabular_hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(combined_dim, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, image, tabular):
        """
        Forward pass for the brightness-based model.

        The method processes both image and tabular data, routing each image to either
        the dark or light image model based on its brightness level.

        Parameters:
            image : torch.Tensor
                Batch of images to process (B, C, H, W)
            tabular : torch.Tensor
                Batch of tabular features (B, num_features)
                
        Returns:
            torch.Tensor
                Model predictions (B, num_classes)
        """

        # Process tabular data 
        tabular_features = self.tabular_model(tabular)
        

        # Calculate brightness of each image
        batch_size = image.size(0)
        image_features = torch.zeros(batch_size, self.num_image_features).to(image.device)

        # Process each image individually
        for i in range(batch_size):
            # Calculate average brightness of the image
            img = image[i]
            brightness = torch.mean(img)
            brightness_threshold = 145  # Adjust this threshold as needed
            
            # Route to appropriate model based on brightness
            if brightness < brightness_threshold:  # Dark image
                image_features[i] = self.dark_image_model(img)
                
            else:  # Light image
                image_features[i] = self.light_image_model(img)
        
        
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
                
    