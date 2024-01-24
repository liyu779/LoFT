from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import MetricCallback

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from lightly.models.modules import heads

from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD

from lightly.models.utils import activate_requires_grad, deactivate_requires_grad
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from lightly.utils.scheduler import CosineWarmupScheduler
from typing import Any, Dict, Tuple
import torch.nn.functional as F

def pgd_attack(model, images, labels, eps=8. / 255., alpha=2. / 255., iters=20, advFlag=None, forceEval=True, randomInit=True):
    # images = images.to(device)
    # labels = labels.to(device)
    # loss = F.cross_entropy()
    # init
    if randomInit:
        delta = torch.rand_like(images) * eps * 2 - eps
    else:
        delta = torch.zeros_like(images)
    delta = torch.nn.Parameter(delta, requires_grad=True)
    # delta.requires_grad_(True)
    # images.requires_grad_(True)
  
    for i in range(iters):
        if advFlag is None:
            if forceEval:
                model.eval()
            outputs = model(images + delta)
        else:
            if forceEval:
                model.eval()
            outputs = model(images + delta, advFlag)

        model.zero_grad()
        cost = F.cross_entropy(outputs, labels)
        # cost.requires_grad_(True)
        # cost.backward()
        
        delta_grad = torch.autograd.grad(cost, [delta])[0]

        delta.data = delta.data + alpha * delta_grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

    model.zero_grad()

    return (images + delta).detach()


class BYOLModel(LightningModule):
    def __init__(self,topk: Tuple[int, ...] = (1, 5),
        freeze_model: bool = False,):
        super().__init__()
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        # create a byol model based on ResNet
        self.classification_head = Linear(512, 10)
        self.criterion = CrossEntropyLoss()


        self.topk = topk
        self.freeze_model = freeze_model


    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone(images).flatten(start_dim=1)
        # features = self.projection_head(features)
        return self.classification_head(features)
    
    def shared_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets = batch[0], batch[1]
        # images = pgd_attack(self,images, targets, eps=8. / 255.,
        #                               alpha=2. / 255, iters=20, forceEval=True).data
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_labels, targets, k=self.topk)
        return loss, topk

    def training_step(self, batch, batch_idx) -> Tensor:

        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx,) -> Tensor:
        with torch.inference_mode(False):
            loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
            batch_size = len(batch[1])
            log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
            self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
            return loss

    def configure_optimizers(self):
        parameters = list(self.classification_head.parameters())
        # if not self.freeze_model:
        #     parameters = self.parameters()

        optimizer = SGD(
            parameters,
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
    
        
    def on_fit_start(self) -> None:
        # Freeze model weights.
        if self.freeze_model:
            deactivate_requires_grad(self.backbone)
            deactivate_requires_grad(self.projection_head)

    def on_fit_end(self) -> None:
        # Unfreeze model weights.
        if self.freeze_model:
            activate_requires_grad(self.backbone)
            activate_requires_grad(self.projection_head)
    

print("Running linear evaluation...")
path_to_train = "/home/model-server/code/cifar10/train/"
path_to_test = "/home/model-server/code/cifar10/test/"
batch_size = 256
# Setup training data. 
train_transform = T.Compose(
    [
        # T.RandomResizedCrop(32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
    ]
)
train_dataset = LightlyDataset(input_dir=str(path_to_train), transform=train_transform)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
)

# Setup validation data.
val_transform = T.Compose(
    [
        # T.Resize(32),
        # T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
    ]
)
val_dataset = LightlyDataset(input_dir=str(path_to_test), transform=val_transform)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    persistent_workers=True,
)

# Train linear classifier.
metric_callback = MetricCallback()
trainer = Trainer(
    max_epochs=25,
    accelerator="gpu",
    devices=1,
    callbacks=[
        LearningRateMonitor(),
        DeviceStatsMonitor(),
        metric_callback,
    ],
    logger=TensorBoardLogger(save_dir=str('./'), name="linear_eval"),
    precision='16-mixed',
    strategy="auto",
    # num_sanity_val_steps=0,
    inference_mode=False
)


model = BYOLModel()
checkpoint = torch.load('/home/model-server/code/aug/benchmark_logs/cifar10/version_8/BYOL/checkpoints/epoch=199byol0.8721.ckpt')

model.load_state_dict(checkpoint['state_dict'],strict= False)

trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
for metric in ["val_top1", "val_top5"]:
        print(f"max linear {metric}: {max(metric_callback.val_metrics[metric])}")