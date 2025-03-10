import torchvision
import torch
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import utility
import os 
import re
from PIL import Image
from torchvision.models.resnet import ResNet50_Weights
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

## Net architecture
class smokeDetectionNet(torch.nn.Module):
    def __init__(self):
        super(smokeDetectionNet, self).__init__()
        
        # self.resNetModel = torchvision.models.resnet50(pretrained=True)
        self.resNetModel = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        fc_features = self.resNetModel.fc.in_features  
        print("model.fc.in_features: ", fc_features)
        for param in self.resNetModel.parameters():
            param.requires_grad = False

        # Unfreeze the final residual block (layer4)
        for param in self.resNetModel.layer4.parameters():
            param.requires_grad = True

        self.resNetModel.fc = torch.nn.Sequential(torch.nn.Linear(2048,128),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(128,1),    # when using BCEWithLogitsLoss
                                        )
        for param in self.resNetModel.fc.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        return self.resNetModel(x)



#### Dataset Reader from path and label
class PathDatasetReader(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Custom dataset that mimics the behavior of torchvision.datasets.ImageFolder.
        
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of labels corresponding to each image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(PathDatasetReader, self).__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform 

        # Directly use labels as indices, but also provide class_names and class_to_idx for compatibility
        self.class_names = ['Normal', 'Smoke']  # This assumes 0='Normal', 1='Smoke'
        self.class_to_idx = {'Normal': 0, 'Smoke': 1}  # Direct mapping

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label at the specified index.
        
        Args:
            idx (int): Index
        
        Returns:
            tuple: (sample, target) where sample is the transformed image and target is the class label.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

                
        # Convert label to tensor, ensure it's float32 for BCEWithLogitsLoss
        label = torch.tensor(label, dtype=torch.float32)

        return image, label, image_path 
    
    def print_class_distribution(self):
        num_normal = self.labels.count(0)
        num_smoke = self.labels.count(1)
        print(f"Number of normal data: {num_normal}, Number of smoke data: {num_smoke}")



    

#########################################################################################
## Shuffle data loader function
class ShuffleDataLoader_Testing():
    def __init__(self, testPath, testLabel, batchSize):
        super().__init__()

        self.testPath = testPath
        self.testLabel = testLabel

        self.batchSize = batchSize
    
    def dataloaderReturn(self):

        data_transforms = {
            'test': torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224, 224)),
                # torchvision.transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            ]),
            }

        print("DataLoader for testing ...")
        test_data = PathDatasetReader(self.testPath, self.testLabel, transform=data_transforms['test'])
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batchSize, num_workers=4, pin_memory=True, shuffle=True)


        loaders = {
            'test': test_loader
        }
        
        test_data.print_class_distribution()

        class_names = test_data.class_names
        print("List of the class: ", class_names)
        print(test_data.class_to_idx)

        return loaders



#########################################################################################
## Shuffle Testing 
class smokeDetectionShuffleTesting(torch.nn.Module):
    def __init__(self, loader, net , cost, runPlatform ):
        super().__init__()

        self.loader = loader
        self.device = torch.device(runPlatform)
        self.net = net.to(self.device, non_blocking=True)
        self.loss = self.createCost(cost).to(self.device, non_blocking=True)

        # metrics
        self.precision_metric = BinaryPrecision().to(self.device, non_blocking=True)
        self.recall_metric =  BinaryRecall().to(self.device, non_blocking=True)
        self.f1score_metric = BinaryF1Score().to(self.device, non_blocking=True)
        self.auroc_metric = BinaryAUROC().to(self.device, non_blocking=True)
    
    # loss function
    def createCost(self,cost):
        support_cost = {
                'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
                'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss()
            }
        return support_cost[cost]
    


    def modelShuffleTesting(self):
        
        test_target_tensor = torch.tensor([], device=self.device)
        test_pred_tensor = torch.tensor([], device=self.device)
        test_probs_tensor = torch.tensor([], device=self.device)
        test_acc = 0.0
        test_loss = 0.0
        saveResultsList = []  

        self.net.eval()
        # for all the shuffled data 
        print("Calculating the metrics for shuffled data ...")
        with torch.no_grad():
            for batch_idx, (data, target, paths) in enumerate(self.loader['test']):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                target = target.unsqueeze(1)
                output = self.net(data)
                probs = torch.sigmoid(output)
                preds = torch.round(probs)
                loss = self.loss(output, target)
                test_acc = test_acc + torch.sum(preds == target.data)
                test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
                test_target_tensor = torch.cat((test_target_tensor, target), 0)
                test_pred_tensor = torch.cat((test_pred_tensor, preds), 0)
                test_probs_tensor = torch.cat((test_probs_tensor, probs), 0)                # use probs to calculate AUROC

                # store file paths, targets and predictions
                for imgPathBatch, targetBatch, predbatch in zip(paths, target, preds):
                    saveResultsList.append([imgPathBatch, int(targetBatch.cpu().numpy().item()), int(predbatch.cpu().numpy().item())])

            
        test_precision = self.precision_metric(test_pred_tensor, test_target_tensor)
        test_recall = self.recall_metric(test_pred_tensor, test_target_tensor)
        test_f1score = self.f1score_metric(test_pred_tensor, test_target_tensor)
        test_augroc = self.auroc_metric(test_probs_tensor, test_target_tensor)
        test_acc = test_acc/len(self.loader['test'].dataset)
        TP_test, FP_test, TN_test, FN_test = utility.confusion(test_pred_tensor, test_target_tensor)

    

        if (FP_test + TN_test) == 0:
            FP_rate_test = np.nan
        else:
            FP_rate_test = FP_test / (FP_test + TN_test)
        if (FN_test + TP_test) == 0:
            FN_rate_test = np.nan
        else:
            FN_rate_test = FN_test / (FN_test + TP_test)
        print("TP_test: ", TP_test, ";   FP_test: ", FP_test, ";   TN_test: ", TN_test, ";   FN_test: ", FN_test)

        table = PrettyTable(["Items", "Test"])
        table.add_row(["Accuracy", test_acc.cpu().numpy()])     
        table.add_row(["Loss", test_loss.cpu().numpy()])
        table.add_row(["Precision", test_precision.cpu().numpy()])
        table.add_row(["Recall", test_recall.cpu().numpy()])
        table.add_row(["F1score", test_f1score.cpu().numpy()])
        table.add_row(["AUROC", test_augroc.cpu().numpy()])
        table.add_row(["FP_rate", FP_rate_test])
        table.add_row(["FN_rate", FN_rate_test])

        print(table)
        
        saveResultsDataFrame = pd.DataFrame(saveResultsList, columns=['ImagePath', 'Target', 'Prediction'])

        return test_acc.cpu().numpy(), test_loss.cpu().numpy(), test_precision.cpu().numpy(), test_recall.cpu().numpy(), test_f1score.cpu().numpy(), test_augroc.cpu().numpy(), FP_rate_test, FN_rate_test, saveResultsDataFrame






if __name__ == "__main__":
    print("good to go ...")


