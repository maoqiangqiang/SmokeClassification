import torchvision
import torch
from torchmetrics.classification import BinaryAUROC
import numpy as np
from prettytable import PrettyTable
import utility
import time 
from PIL import Image
from torchvision.models.resnet import ResNet50_Weights
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


#  enforce deterministic behavior in PyTorch
import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)



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

        return image, label
    
    def print_class_distribution(self):
        num_normal = self.labels.count(0)
        num_smoke = self.labels.count(1)
        print(f"Number of normal data: {num_normal}, Number of smoke data: {num_smoke}")




#########################################################################################
## data loader function
class smokeDetectionDataLoader():
    def __init__(self, trainPath, trainLabel, validPath, validLabel, batchSize):
        super().__init__()

        self.trainPath = trainPath
        self.trainLabel = trainLabel
        self.validPath = validPath
        self.validLabel = validLabel
        self.batchSize = batchSize
        
    def dataloaderReturn(self):

        data_transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224,224)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(20),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # 'test': torchvision.transforms.Compose([
            #     torchvision.transforms.Resize(size=(224, 224)),
            #     torchvision.transforms.ToTensor(),
            #     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # ]),
            }

        print("DataLoader for training ...")
        train_data = PathDatasetReader(self.trainPath, self.trainLabel, transform=data_transforms['train'])
        valid_data = PathDatasetReader(self.validPath, self.validLabel, transform=data_transforms['valid'])

        ### for distributed data paralle 
        train_sampler = torch.utils.data.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.DistributedSampler(valid_data)


        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batchSize, num_workers=4, pin_memory=True, shuffle=(train_sampler is None), sampler = train_sampler)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=self.batchSize, num_workers=4, pin_memory=True, shuffle=(valid_sampler is None), sampler = valid_sampler)
        samplers = {
            'train': train_sampler,
            'valid': valid_sampler
        }

        loaders = {
            'train': train_loader,
            'valid': valid_loader
        }

        train_data.print_class_distribution()
        valid_data.print_class_distribution()

        class_names = train_data.class_names
        print("List of the class: ", class_names)
        print(train_data.class_to_idx)

        return samplers, loaders






#########################################################################################
## model class
class smokeDetectionModel(torch.nn.Module):
    def __init__(self, samplers, loader, net, cost, opt, runPlatform, rank):
        super().__init__()

        self.samplers = samplers
        self.loader = loader
        self.device = torch.device(runPlatform)
        self.net = net.to(self.device, non_blocking=True)
        self.loss = self.createCost(cost).to(self.device, non_blocking=True)
        self.optimizer = self.createOptimizer(opt)
        self.rank = rank

        # metrics
        self.auroc_metric = BinaryAUROC().to(self.device, non_blocking=True)
    
    # loss function 
    def createCost(self,cost):
        support_cost = {
                'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
                'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss()
            }
        return support_cost[cost]

    # optimizer
    def createOptimizer(self,opt,**rests):
        support_optim = {
            'SGD': torch.optim.SGD(list(self.net.module.resNetModel.layer4.parameters()) + list(self.net.module.resNetModel.fc.parameters()), lr=0.01, **rests),
            'Adam': torch.optim.Adam(list(self.net.module.resNetModel.layer4.parameters()) + list(self.net.module.resNetModel.fc.parameters()), lr=0.01, **rests),
            'RMSprop':torch.optim.RMSprop(list(self.net.module.resNetModel.layer4.parameters()) + list(self.net.module.resNetModel.fc.parameters()), lr=0.01, **rests),
            'Adagrad': torch.optim.Adagrad(list(self.net.module.resNetModel.layer4.parameters()) + list(self.net.module.resNetModel.fc.parameters()), lr=0.01, **rests),
            'AdamW': torch.optim.AdamW(list(self.net.module.resNetModel.layer4.parameters()) + list(self.net.module.resNetModel.fc.parameters()), lr=0.01, **rests)
        }
        return support_optim[opt]
    
    # @utility.time_function
    def modelTraining(self, num_epochs):
        n_epochs = num_epochs
        # valid_loss_min = np.Inf

        train_accuracy_list = []
        train_loss_list = []
        valid_accuracy_list = []
        valid_loss_list = []
        train_precision_list = []
        train_recall_list = []
        train_f1score_list = []
        train_auroc_list = []
        valid_precision_list = []
        valid_recall_list = []
        valid_f1score_list = []
        valid_auroc_list = []

        # Initialize the scheduler
        scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=1e-5)
        for epoch in range(1, (n_epochs+1)):
            print(f"Epoch: {epoch} at rank: {self.rank}")

            self.samplers['train'].set_epoch(epoch)
            self.samplers['valid'].set_epoch(epoch)

            start_epoch = time.perf_counter()

            train_loss, valid_loss = 0.0, 0.0
            train_acc, valid_acc = 0.0, 0.0
            TP_train, FP_train, TN_train, FN_train = 0, 0, 0, 0
            TP_valid, FP_valid, TN_valid, FN_valid = 0, 0, 0, 0

            # Lists to store predictions and targets for AUROC calculation
            train_pred_list, train_target_list = [], []
            valid_pred_list, valid_target_list = [], []


            self.net.train()
            # Reset the gradients to None
            self.optimizer.zero_grad(set_to_none=True)

            # for batch_idx, (data, target) in enumerate(train_dataloader):
            for batch_idx, (data, target) in enumerate(self.loader['train']):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                target = target.unsqueeze(1)
                # print("target shape: ", target.shape)
                if batch_idx == 0:
                    print("train data is_cuda: ", data.is_cuda)
                output = self.net(data)                           
                ########### FOR BCEWithLogitsLoss
                probs = torch.sigmoid(output) 
                preds = torch.round(probs)     
                loss = self.loss(output, target)                  

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Calculate batch accuracy and loss
                train_acc = train_acc + torch.sum(preds == target.data)
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))    #  the average train_loss for the finished batches

                # Track TP, FP, TN, FN for the training set
                TP, FP, TN, FN = utility.confusion(preds, target)
                TP_train += TP
                FP_train += FP
                TN_train += TN
                FN_train += FN
        
                # Collect predictions (probs) and targets for AUROC calculation
                train_pred_list.append(probs.detach())
                train_target_list.append(target.detach())


            # Average loss and accuracy over dataset size
            train_acc = train_acc.float() / len(self.loader['train'].dataset)
            torch.distributed.reduce(train_acc, dst=0, op=torch.distributed.ReduceOp.SUM)
            train_loss = train_loss.float() / len(self.loader['train'])
            torch.distributed.reduce(train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)


            # Reduce TP, FP, TN, FN across ranks
            TP_train = torch.tensor(TP_train, device=self.device)
            FP_train = torch.tensor(FP_train, device=self.device)
            TN_train = torch.tensor(TN_train, device=self.device)
            FN_train = torch.tensor(FN_train, device=self.device)
            torch.distributed.all_reduce(TP_train, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(FP_train, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(TN_train, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(FN_train, op=torch.distributed.ReduceOp.SUM)

            scheduler.step()

            end_epoch = time.perf_counter()
            print("From Rank:{}, Epoch training time: {:.4f}".format(self.rank, end_epoch - start_epoch))


            self.net.eval()
            with torch.no_grad():

                for batch_idx, (data, target) in enumerate(self.loader['valid']):
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    target = target.unsqueeze(1)
                    if batch_idx == 0:
                        print("valid data is_cuda: ", data.is_cuda)
                    output = self.net(data)
                    probs = torch.sigmoid(output)
                    preds = torch.round(probs)
                    loss = self.loss(output, target)
          
                    valid_acc = valid_acc + torch.sum(preds == target.data)
                    valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                    # Track TP, FP, TN, FN per batch
                    TP, FP, TN, FN = utility.confusion(preds, target)
                    TP_valid += TP
                    FP_valid += FP
                    TN_valid += TN
                    FN_valid += FN

                    # Collect predictions and targets for AUROC calculation
                    valid_pred_list.append(probs.detach())
                    valid_target_list.append(target.detach())
            

            # Average loss and accuracy over dataset size
            valid_acc = valid_acc.float() / len(self.loader['valid'].dataset)
            torch.distributed.reduce(valid_acc, dst=0, op=torch.distributed.ReduceOp.SUM)
            valid_loss = valid_loss.float() / len(self.loader['valid'])
            torch.distributed.reduce(valid_loss, dst=0, op=torch.distributed.ReduceOp.SUM)

            # Reduce TP, FP, TN, FN across ranks for validation
            TP_valid = torch.tensor(TP_valid, device=self.device)
            FP_valid = torch.tensor(FP_valid, device=self.device)
            TN_valid = torch.tensor(TN_valid, device=self.device)
            FN_valid = torch.tensor(FN_valid, device=self.device)
            torch.distributed.all_reduce(TP_valid, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(FP_valid, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(TN_valid, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(FN_valid, op=torch.distributed.ReduceOp.SUM)



            # Concatenate local predictions and targets for AUROC calculation
            local_train_preds = torch.cat(train_pred_list, dim=0)
            local_train_targets = torch.cat(train_target_list, dim=0)
            local_valid_preds = torch.cat(valid_pred_list, dim=0)
            local_valid_targets = torch.cat(valid_target_list, dim=0)

            # Prepare buffers to gather predictions and targets from all ranks
            train_preds_gathered = [torch.zeros_like(local_train_preds) for _ in range(torch.distributed.get_world_size())]
            train_targets_gathered = [torch.zeros_like(local_train_targets) for _ in range(torch.distributed.get_world_size())]
            valid_preds_gathered = [torch.zeros_like(local_valid_preds) for _ in range(torch.distributed.get_world_size())]
            valid_targets_gathered = [torch.zeros_like(local_valid_targets) for _ in range(torch.distributed.get_world_size())]

            # Gather predictions and targets across ranks
            torch.distributed.all_gather(train_preds_gathered, local_train_preds)
            torch.distributed.all_gather(train_targets_gathered, local_train_targets)
            torch.distributed.all_gather(valid_preds_gathered, local_valid_preds)
            torch.distributed.all_gather(valid_targets_gathered, local_valid_targets)

            # Concatenate gathered lists to form the complete predictions and targets
            train_preds_all = torch.cat(train_preds_gathered, dim=0)
            train_targets_all = torch.cat(train_targets_gathered, dim=0)
            valid_preds_all = torch.cat(valid_preds_gathered, dim=0)
            valid_targets_all = torch.cat(valid_targets_gathered, dim=0)

            # Calculate AUROC on the complete dataset (all ranks combined)
            train_auroc = self.auroc_metric(train_preds_all, train_targets_all)
            valid_auroc = self.auroc_metric(valid_preds_all, valid_targets_all)

            # Calculate other metrics based on aggregated values
            train_precision = TP_train / (TP_train + FP_train) if TP_train + FP_train > 0 else np.nan
            train_precision = torch.tensor(train_precision) if not isinstance(train_precision, torch.Tensor) else train_precision
            train_recall = TP_train / (TP_train + FN_train) if TP_train + FN_train > 0 else np.nan
            train_recall = torch.tensor(train_recall) if not isinstance(train_recall, torch.Tensor) else train_recall
            train_f1score = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else np.nan
            train_f1score = torch.tensor(train_f1score) if not isinstance(train_f1score, torch.Tensor) else train_f1score

            valid_precision = TP_valid / (TP_valid + FP_valid) if TP_valid + FP_valid > 0 else np.nan
            valid_precision = torch.tensor(valid_precision) if not isinstance(valid_precision, torch.Tensor) else valid_precision
            valid_recall = TP_valid / (TP_valid + FN_valid) if TP_valid + FN_valid > 0 else np.nan
            valid_recall = torch.tensor(valid_recall) if not isinstance(valid_recall, torch.Tensor) else valid_recall
            valid_f1score = 2 * (valid_precision * valid_recall) / (valid_precision + valid_recall) if (valid_precision + valid_recall) > 0 else np.nan
            valid_f1score = torch.tensor(valid_f1score) if not isinstance(valid_f1score, torch.Tensor) else valid_f1score

            # False positive and false negative rates
            FP_rate_train = FP_train / (FP_train + TN_train) if (FP_train + TN_train) > 0 else np.nan
            FN_rate_train = FN_train / (FN_train + TP_train) if (FN_train + TP_train) > 0 else np.nan
            FP_rate_valid = FP_valid / (FP_valid + TN_valid) if (FP_valid + TN_valid) > 0 else np.nan
            FN_rate_valid = FN_valid / (FN_valid + TP_valid) if (FN_valid + TP_valid) > 0 else np.nan


            # Append metrics to lists for each epoch
            if self.rank == 0:
                train_accuracy_list.append(train_acc.cpu())
                train_loss_list.append(train_loss.cpu())
                valid_accuracy_list.append(valid_acc.cpu())
                valid_loss_list.append(valid_loss.cpu())
                train_precision_list.append(train_precision.cpu())
                train_recall_list.append(train_recall.cpu())
                train_f1score_list.append(train_f1score.cpu())
                train_auroc_list.append(train_auroc.cpu())
                valid_precision_list.append(valid_precision.cpu())
                valid_recall_list.append(valid_recall.cpu())
                valid_f1score_list.append(valid_f1score.cpu())
                valid_auroc_list.append(valid_auroc.cpu())

                # PrettyTable output for current epoch
                table = PrettyTable(["Epoch","Items", "Training", "Validation"])
                table.add_row([epoch, "Accuracy", train_acc.cpu().numpy(), valid_acc.cpu().numpy()])
                table.add_row([epoch, "Loss", train_loss.cpu().numpy(), valid_loss.cpu().numpy()])
                table.add_row([epoch, "Precision", train_precision.cpu().numpy(), valid_precision.cpu().numpy()])
                table.add_row([epoch, "Recall", train_recall.cpu().numpy(), valid_recall.cpu().numpy()])
                table.add_row([epoch, "F1score", train_f1score.cpu().numpy(), valid_f1score.cpu().numpy()])
                table.add_row([epoch, "AUROC", train_auroc.cpu().numpy(), valid_auroc.cpu().numpy()])
                table.add_row([epoch, "FP_rate", FP_rate_train.cpu().numpy(), FP_rate_valid.cpu().numpy()])
                table.add_row([epoch, "FN_rate", FN_rate_train.cpu().numpy(), FN_rate_valid.cpu().numpy()])
                print(table)


        if self.rank == 0:
            return self.net, train_accuracy_list, train_loss_list, valid_accuracy_list, valid_loss_list, \
                train_precision_list, train_recall_list, train_f1score_list, train_auroc_list, valid_precision_list, \
                    valid_recall_list, valid_f1score_list, valid_auroc_list
        else:
            return None





  

if __name__ == "__main__":
    print("good to go ...")







