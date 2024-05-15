import io
import torch
import torch.nn as nn
import seaborn as sns
from torchmetrics.classification import BinaryF1Score, BinaryConfusionMatrix, BinaryAccuracy, BinaryRecall, \
    BinaryPrecision, MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, \
    MulticlassAccuracy
from torchmetrics import MetricCollection
import PIL.Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import lightning as L
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger

from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from utils import loadData, set_reproducibility

class Classificator(L.LightningModule):
    def __init__(self, CNN, class_labels, config, num_classes, sync_dist=False):
        super().__init__()
        self.CNN = CNN
        self.class_labels = class_labels
        self.learning_rate = config["lr"]
        self.sync_dist= sync_dist
        if config["loss"] == "BCEwLogits":
            def bcewl(y_est, y_true):
                return F.binary_cross_entropy_with_logits(y_est,
                                                          F.one_hot(y_true, num_classes).type(torch.float))
            self.loss = bcewl
        elif config["loss"] == "CrossEntropyLoss":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss {config['loss']} not implemented")

        if num_classes == 2:
            self.val_confusion_matrix = BinaryConfusionMatrix()
            self.test_confusion_matrix = BinaryConfusionMatrix()
            metrics = MetricCollection([
                BinaryPrecision(),
                BinaryRecall(),
                BinaryF1Score(),
                BinaryAccuracy()
            ])
        else:
            self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes)
            self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes)
            metrics = MetricCollection([
                MulticlassPrecision(num_classes),
                MulticlassRecall(num_classes),
                MulticlassF1Score(num_classes),
                MulticlassAccuracy(num_classes),
            ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix="test_")


    def training_step(self, batch):
        images, labels = batch
        output = self.CNN(images)
        _, preds = torch.max(output, dim=1)
        loss = self.loss(output, labels)
        self.log("Training loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.train_metrics.update(preds, labels)
        return loss
    
    def on_train_epoch_end(self):
        performance = self.train_metrics.compute()
        self.log_dict(performance, sync_dist=self.sync_dist)
        self.train_metrics.reset()

    def validation_step(self, batch):
        images, labels = batch
        output = self.CNN(images)
        loss = self.loss(output, labels)
        _, preds = torch.max(output, dim=1)
        self.val_confusion_matrix.update(preds, labels)
        self.log("Validation loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.valid_metrics.update(preds, labels)
        return loss
    
    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        self.log_dict(output, sync_dist=self.sync_dist)
        self.valid_metrics.reset()
        cm = self.val_confusion_matrix.compute()
        image = self.transform_confusion_matrix(cm)
        self.logger.experiment.add_image("Confusion matrix validation results", image)
        self.val_confusion_matrix.reset()
    
    def test_step(self, batch):
        images, labels = batch
        output = self.CNN(images)
        loss = self.loss(output, labels)
        _, preds = torch.max(output, dim=1)
        self.test_metrics.update(preds, labels)
        self.test_confusion_matrix.update(preds, labels)
        self.log("Test loss", loss, prog_bar=True, sync_dist=self.sync_dist)

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.log_dict(output, sync_dist=self.sync_dist)
        self.test_metrics.reset()
        cm = self.test_confusion_matrix.compute()
        image = self.transform_confusion_matrix(cm)
        self.logger.experiment.add_image("Confusion matrix test results", image)
        self.test_confusion_matrix.reset()

    # Takes a tensor and plot confusion matrix from it and then return as tensor
    def transform_confusion_matrix(self, cm):
        cm = cm.cpu().numpy()
        fig = plt.figure()
        sns.heatmap(cm, annot=True, xticklabels=self.class_labels, yticklabels=self.class_labels, fmt="g", cbar=False)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        plt.close()
        buf.close()
        return image


    def configure_optimizers(self):
        
      optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      def learning_rate_fn(epoch):
          return 0.95 ** epoch
      scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_fn)

      return {'optimizer': optimizer, 
              'lr_scheduler':scheduler,
              }

# inspired by https://www.sciencedirect.com/science/article/pii/S0208521622000742#s0010
class ConvNet(nn.Module):
    def __init__(self, num_classes=2, dropout = 0.5):
        super().__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout2d(p=0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.25),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(in_features=4608, out_features=1024),
            nn.LeakyReLU(0.02),
            nn.BatchNorm1d(num_features=1024),
            nn.Dropout(dropout),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.02),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(dropout),
            nn.Linear(in_features=512, out_features=num_classes),
        )

    def forward(self, x):
        return self.convlayer(x)
    
# train method for Hyperparameter tuning
def train_func(config):
    
    if config["reproducibility_active"]:
        set_reproducibility()
    #set_reproducibility(config["seed"]) # https://discuss.ray.io/t/reproducibility-of-ray-tune-with-seeds/6812/4
    
    project_data = config.get('project_data_dir', "/Project")
    data_set_dir = config.get('data_set_dir', project_data + "/chest_xray")
    lightning_logs = config.get('lightning_logs', project_data + "/lightning_logs")

    # TODO
    train_loader, val_loader, _ = loadData(dataDir=data_set_dir,
                                           numWorkers=7,
                                           batchSize=config["batch_size"],
                                           useSampler=config.get('use_sampler', False),
                                           )

    cnn = ConvNet(dropout= config["dropout"])
    # TODO
    model = Classificator(cnn, ["Normal", "Pneumonia"], config, 2, sync_dist=True)
    logger = TensorBoardLogger(lightning_logs, name="simple_CNN/tuning")
    early_stopping = L.pytorch.callbacks.EarlyStopping(monitor='Validation loss', patience=10, min_delta=1e-6)
    callbacks = [early_stopping, RayTrainReportCallback()]

    trainer = L.Trainer(
        max_epochs= config["epochs"],
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=callbacks,
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        logger = logger
    )

    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    trainer = prepare_trainer(trainer)
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)
    
# tuning method for Hyperparam tuning
def tuning(ray_trainer, search_space, num_samples=10, num_epochs=5):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_BinaryAccuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()