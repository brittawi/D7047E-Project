import io
import torch
import torch.nn as nn
import seaborn as sns
from torchmetrics.classification import BinaryF1Score, BinaryConfusionMatrix, BinaryAccuracy, BinaryRecall, BinaryPrecision
from torchmetrics import MetricCollection
import PIL.Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import lightning as L
import torch.nn.functional as F

class Classificator(L.LightningModule):
    def __init__(self, CNN, class_labels, config, num_classes):
        super().__init__()
        self.CNN = CNN
        self.class_labels = class_labels
        self.learning_rate = config["lr"]
        if config["loss"] == "BCEwLogits":
            def bcewl(y_est, y_true):
                return F.binary_cross_entropy_with_logits(y_est,
                                                          F.one_hot(y_true, num_classes).type(torch.float))
            self.loss = bcewl
        elif config["loss"] == "CrossEntropyLoss":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss {config['loss']} not implemented")

        self.val_confusion_matrix = BinaryConfusionMatrix()
        self.test_confusion_matrix = BinaryConfusionMatrix()
        metrics = MetricCollection([
            BinaryPrecision(),
            BinaryRecall(),
            BinaryF1Score(),
            BinaryAccuracy()
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix="test_")


    def training_step(self, batch):
        images, labels = batch
        output = self.CNN(images)
        _, preds = torch.max(output, dim=1)
        loss = self.loss(output, labels)
        self.log("Traning loss", loss, prog_bar=True)
        self.train_metrics.update(preds, labels)
        return loss
    
    def on_train_epoch_end(self):
        performance = self.train_metrics.compute()
        self.log_dict(performance)
        self.train_metrics.reset()

    def validation_step(self, batch):
        images, labels = batch
        output = self.CNN(images)
        loss = self.loss(output, labels)
        _, preds = torch.max(output, dim=1)
        self.val_confusion_matrix.update(preds, labels)
        self.log("Validation loss", loss, prog_bar=True)
        self.valid_metrics.update(preds, labels)
        return loss
    
    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        self.log_dict(output)
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
        self.log("Test loss", loss, prog_bar=True)
        self.log_dict(self.test_metrics.compute())

    def on_test_end(self):
        cm = self.test_confusion_matrix.compute()
        image = self.transform_confusion_matrix(cm)
        self.logger.experiment.add_image("Confusion matrix test results", image)

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