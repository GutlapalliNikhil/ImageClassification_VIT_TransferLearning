import torch
import torchvision

from torch import nn
from torchvision import transforms

import helper

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


################

#Dataset and Dataloading

################

image_path = helper.download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")

train_dir = image_path / "train"
test_dir = image_path / "test"

IMG_SIZE = 224
BATCH_SIZE = 4

manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(train_dir, transform=manual_transforms)
test_data = datasets.ImageFolder(test_dir, transform=manual_transforms)

class_names = train_data.classes

# Turn images into data loaders
train_dataloader = DataLoader(
      train_data,
      batch_size=BATCH_SIZE,
      shuffle=True,
      pin_memory=True,
)

test_dataloader = DataLoader(
      test_data,
      batch_size=BATCH_SIZE,
      shuffle=False,
      pin_memory=True,
)


##########

# Downloading the pretrained model 

##########

vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
vit_model = torchvision.models.vit_b_16(weights = vit_weights).cuda()

for parameter in vit_model.parameters():
    parameter.requires_grad = False

vit_model.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)


############

# Model Training

############

optimizer = torch.optim.Adam(params=vit_model.parameters(), lr=1e-3)

loss_fn = torch.nn.CrossEntropyLoss()

vit_model.to(device)

epochs = 5

results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
}

for epoch in tqdm(range(epochs)):

	vit_model.train()

	train_loss, train_acc = 0, 0

	for batch, (X, y) in enumerate(train_dataloader):

		X, y = X.to(device), y.to(device)

		y_pred = vit_model(X)

		loss = loss_fn(y_pred, y)

		train_loss += loss.item()

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
		train_acc += (y_pred_class == y).sum().item()/len(y_pred)

	train_loss = train_loss / len(train_dataloader)

	train_acc = train_acc / len(train_dataloader)

	vit_model.eval() 

	test_loss, test_acc = 0, 0

	with torch.inference_mode():

		for batch, (X, y) in enumerate(test_dataloader):

			X, y = X.to(device), y.to(device)

			test_pred_logits = vit_model(X)

			loss = loss_fn(test_pred_logits, y)

			test_loss += loss.item()

			test_pred_labels = test_pred_logits.argmax(dim=1)

			test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

	test_loss = test_loss / len(test_dataloader)

	test_acc = test_acc / len(test_dataloader)

	print(f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
	)

	results["train_loss"].append(train_loss)
	results["train_acc"].append(train_acc)
	results["test_loss"].append(test_loss)
	results["test_acc"].append(test_acc)

loss = results["train_loss"]
test_loss = results["test_loss"]

accuracy = results["train_acc"]
test_accuracy = results["test_acc"]

epochs = range(len(results["train_loss"]))

plt.figure(figsize=(15, 7))

# Plot loss
import matplotlib.pyplot as plt

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, label="train_loss")
plt.plot(epochs, test_loss, label="test_loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, label="train_accuracy")
plt.plot(epochs, test_accuracy, label="test_accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.legend()

# Save the plots as image files
plt.savefig('loss_accuracy_plot.png')
