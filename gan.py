import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorModel(nn.Module):
    def __init__(self, noise_dim, output_dim, hidden_dim=128):
        """
        Initializes the Generator model.

        Parameters:
        - noise_dim (int): Dimensionality of the input noise vector.
        - output_dim (int): Dimensionality of the generated output.
        - hidden_dim (int): Dimensionality of the hidden layers (default: 128).
        """
        super(GeneratorModel, self).__init__()

        self.model = nn.Sequential(
            # Fully connected layer 1
            nn.Linear(noise_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            # Fully connected layer 2
            nn.Linear(hidden_dim, hidden_dim * 2),
            #nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),

            # Fully connected layer 3
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            #nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim * 4, output_dim),
            #nn.Tanh()  
        )

    def forward(self, x):
        """
        Forward pass for the generator.

        Parameters:
        - x (torch.Tensor): Input noise tensor of shape (batch_size, noise_dim).

        Returns:
        - torch.Tensor: Generated output tensor of shape (batch_size, output_dim).
        """
        return self.model(x)
    

class DiscriminatorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        """
        Initializes the Discriminator model.

        Parameters:
        - input_dim (int): Dimensionality of the input data.
        - hidden_dim (int): Dimensionality of the hidden layers (default: 128).
        """
        super(DiscriminatorModel, self).__init__()

        self.model = nn.Sequential(
            # Fully connected layer 1
            nn.Linear(input_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),

            # Fully connected layer 2
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),

            # Fully connected layer 3
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),

            # Fully connected output layer
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        """
        Forward pass for the discriminator.

        Parameters:
        - x (torch.Tensor): Input data tensor of shape (batch_size, input_dim).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, 1), representing probabilities.
        """
        return self.model(x)
    

class GAN(nn.Module):
    def __init__(self, noise_dim, data_dim, hidden_dim=128):
        """
        Initializes the GAN model, including the generator and discriminator.

        Parameters:
        - noise_dim (int): Dimensionality of the input noise vector.
        - data_dim (int): Dimensionality of the data output/input.
        - hidden_dim (int): Dimensionality of the hidden layers for both models (default: 128).
        """
        super(GAN, self).__init__() 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = GeneratorModel(noise_dim, data_dim, hidden_dim).to(device)
        self.discriminator = DiscriminatorModel(data_dim, hidden_dim).to(device)

    def forward(self, noise):
        """
        Forward pass for the GAN.

        Parameters:
        - noise (torch.Tensor): Input noise tensor for the generator.

        Returns:
        - torch.Tensor: Discriminator output for generated data.
        """
        generated_data = self.generator(noise)
        classification = self.discriminator(generated_data)
        return classification