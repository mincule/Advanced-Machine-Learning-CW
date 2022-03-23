import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchvision import transforms


class LitCNN(pl.LightningModule):
    def __init__(self, num_classes=10, dims=(3, 32, 32), learning_rate=2e-4):
        super().__init__()

        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = num_classes
        self.dims = dims
        channels, width, height = self.dims
        self.transform = transforms.ToTensor()

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Conv2d(channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, self.num_classes),
        )

        self.accuracy = Accuracy()
    
    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer