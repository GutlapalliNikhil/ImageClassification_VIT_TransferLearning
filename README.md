It is remarkable that it required a week to gain a comprehensive understanding of Transformer models and their intricate workings. However, once familiarized, employing the model itself can be accomplished within a day or even less.

# Project Demo: Pretrained ViT Model on Custom Dataset

This repository contains a demonstration of using a pretrained Vision Transformer (ViT) model, trained on the ImageNet dataset, on a custom dataset. The code is implemented in PyTorch and includes training and evaluation of the model.

### Dataset and Data Loading

The custom dataset used in this project consists of images of pizza, steak, and sushi. The dataset is downloaded from a specified source and then split into training and test sets. The images are resized to a fixed size of 224x224 pixels and converted to tensors using torchvision transforms. The dataset is loaded into PyTorch data loaders for efficient training and evaluation.

### Pretrained Model

The pretrained ViT model is downloaded using the torchvision.models.vit_b_16 function with the default weights specified by torchvision.models.ViT_B_16_Weights.DEFAULT. The model is then loaded onto the GPU (if available) for faster computation. The model's head (classifier) is modified to match the number of classes in the custom dataset.

### Model Training

The model is trained using the loaded dataset and the Adam optimizer with a learning rate of 1e-3. The loss function used is cross-entropy loss. The training is performed for a specified number of epochs. During each epoch, the model is trained on the training dataset, and the loss and accuracy are recorded. After each epoch, the model is evaluated on the test dataset to calculate the test loss and accuracy. The training progress is displayed using a progress bar.

### Results and Visualization

The training and evaluation results, including train loss, train accuracy, test loss, and test accuracy, are stored in a dictionary. The results are then plotted using matplotlib to visualize the changes in loss and accuracy over the epochs. The resulting plots are saved as an image file named "loss_accuracy_plot.png".

Please note that this README provides an overview of the code. For detailed information and instructions on running the code, refer to the code comments and documentation within the code files.

If you have any further questions, feel free to reach out.
