import torch
import torchmetrics
from torch.utils.data import Dataset, DataLoader 
from torch.optim.adam import Adam
import torchmetrics
import segmentation_models_pytorch as smp
import lightning as L


import UNets


"""Parameters:

    in_channels: number of input channels. For HAADF image this is 1
    classes: number of classes. Set to 4 for point defects (no_defect, vacancy, interstitial, substitution)
    model: model pytorch will train. Can add different architectures to UNets.py via segmentation_models_pytorch
    lossFunc: loss function from torch.nn()
    inputDim: Expecting a square image of side length inputDim
    batchSize: Batch size

Data Architecture:

    x: tensor of input images dim batchSize x in_channels x inputDim x inputDim
    y: tensor of labels dim batchSize x classes x inputDim x inputDim 
    loss: class balanced loss computed with lossFunc 

"""
in_channels = 1
classes = 4
model = UNets.efficientnetb5(in_channels=in_channels, classes=classes)

inputDim = 1024
batchSize = 32
epochs = 1

weights = torch.tensor([1,1,1,1], dtype=torch.float32)
criterion = torch.nn.BCELoss(weight=weights, reduction='mean')
optimizer = Adam(model.parameters(), lr=1e-3)

trainDataset = ""
testDataset = ""

trainloader = DataLoader(trainDataset, num_workers=4, shuffle=True, batch_size=batchSize)
testloader = DataLoader(testDataset, num_workers=4, shuffle=False, batch_size=batchSize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LitDefectSegmentation(L.LightningModule):
    def __init__(self, model):
        super().__ini__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, labels = batch
        x = x.view
        x = x.view(x.size(0), in_channels, inputDim, inputDim).to(device)
        labels = labels.view(x.size(0), classes, inputDim, inputDim).to(device)
        y = self.model(x)
        loss = criterion(y, labels).mean()
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        x = x.view
        x = x.view(x.size(0), in_channels, inputDim, inputDim).to(device)
        labels = labels.view(x.size(0), classes, inputDim, inputDim).to(device)
        y = self.model(x)
        loss = criterion(y, labels).mean()
        self.log("test_loss", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

defectSegmentation = LitDefectSegmentation(model)

trainer = L.Trainer(limit_train_batches= batchSize, max_epochs=1)
trainer.fit(model=defectSegmentation, train_dataloaders=trainloader)
trainer.test(model=defectSegmentation, dataloaders=testloader)

