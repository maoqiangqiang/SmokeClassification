import torch
import numpy as np
import pandas as pd
import time 
import os 
import argparse
import sys
sys.path.append('./src/')

from trainingFunc_ResNet_DDP import smokeDetectionNet, smokeDetectionDataLoader, smokeDetectionModel
import utility
import distDataParallel 




if __name__ == '__main__':
    torch.manual_seed(970425)
    np.random.seed(970425)

    total_start = time.time()

    # parse arguments
    parser = argparse.ArgumentParser(description='DistDataParallel Exp')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--num_epochs', default=100, type=int, help='')
    parser.add_argument('--deviceArg', default='cuda', type=str, help='')


    args = parser.parse_args()

    # experiment case 
    scenarioNum = 10
    CSVFolder = "FinalExp"
    caseFolder = "rawSynGen1"
    dataCase = "HTJDSF"


    rank, current_device =  distDataParallel.setup(args.dist_backend, args.init_method, args.world_size)
    print("The rank is {} and the current device is {}".format(rank, current_device))
            
    device = torch.device(args.deviceArg)



    ## These ten video scenarios are rotated as the test set across ten runs.
    for i in range(scenarioNum):

        i = i+1
        print("---------------------------------------------")
        print(f"\n Starting Run {i}...")
        # model net architecture
        model_init = smokeDetectionNet()
        total_params = sum(p.numel() for p in model_init.parameters())
        print('Total number of parameters: ', total_params)
        total_trained_params = sum(p.numel() for p in model_init.parameters() if p.requires_grad)
        print('Total number of trainable parameters: ', total_trained_params)
        lay4_tot_params = sum(p.numel() for p in model_init.resNetModel.layer4.parameters())
        print('Total number of parameters in the layer4: ', lay4_tot_params)
        lay4_trained_params = sum(p.numel() for p in model_init.resNetModel.layer4.parameters() if p.requires_grad)
        print('Total number of trainable parameters in the layer4: ', lay4_trained_params)
        fc_tot_params = sum(p.numel() for p in model_init.resNetModel.fc.parameters())
        print('Total number of parameters in the fc layer: ', fc_tot_params)
        fc_trained_params = sum(p.numel() for p in model_init.resNetModel.fc.parameters() if p.requires_grad)
        print('Total number of trainable parameters in the fc layer: ', fc_trained_params)

        utility.count_parameters(model_init)





        ## model ddp 
        model_init.to(device)
        model_init = torch.nn.parallel.DistributedDataParallel(model_init, device_ids=[rank])
        print('From Rank: {}, ==> Preparing data..'.format(rank))

        



        ###################################################### Datasets ######################################################
        ## Smoke dataset
        SmokeDatasetBaseDir = './data/'
        TrainSmokeImgPathFile = './datasetCSVPrep/'+CSVFolder+'/trainSmokeImgPath_' + str(i)+'.csv'
        TestSmokeImgPathFile = './datasetCSVPrep/'+CSVFolder+'/testSmokeImgPath_' + str(i)+'.csv'
        # read the csv file and save the image path
        TrainSmokeImgPath_pd = pd.read_csv(TrainSmokeImgPathFile, header=0)
        TestSmokeImgPath_pd = pd.read_csv(TestSmokeImgPathFile, header=0)
        TrainSmokeImgPath = TrainSmokeImgPath_pd['imagePath'].tolist()
        TrainSmokeImgFullPath = [SmokeDatasetBaseDir + path for path in TrainSmokeImgPath]
        TestSmokeImgPath = TestSmokeImgPath_pd['imagePath'].tolist()
        TestSmokeImgFullPath = [SmokeDatasetBaseDir + path for path in TestSmokeImgPath]
        TrainSmokeLabel = TrainSmokeImgPath_pd['label'].tolist()
        TestSmokeLabel = TestSmokeImgPath_pd['label'].tolist()
        
        ## Normal dataset
        NormalDatasetBaseDir = './data/'
        TrainNormalImgPathFile = '../datasetCSVPrep/'+CSVFolder+'/trainNormalImgPath_' + str(i)+'.csv'
        TestNormalImgPathFile = '../datasetCSVPrep/'+CSVFolder+'/testNormalImgPath_' + str(i)+'.csv'
        # read the csv file and save the image path
        TrainNormalImgPath_pd = pd.read_csv(TrainNormalImgPathFile, header=0)
        TestNormalImgPath_pd = pd.read_csv(TestNormalImgPathFile, header=0)
        TrainNormalImgPath = TrainNormalImgPath_pd['imagePath'].tolist()
        TrainNormalImgFullPath = [NormalDatasetBaseDir + path for path in TrainNormalImgPath]
        TestNormalImgPath = TestNormalImgPath_pd['imagePath'].tolist()
        TestNormalImgFullPath = [NormalDatasetBaseDir + path for path in TestNormalImgPath]
        TrainNormalLabel = TrainNormalImgPath_pd['label'].tolist()
        TestNormalLabel = TestNormalImgPath_pd['label'].tolist()

        # combine smoke and normal as training set and testing set
        trainPath = TrainSmokeImgFullPath + TrainNormalImgFullPath
        trainLabel = TrainSmokeLabel + TrainNormalLabel
        testPath = TestSmokeImgFullPath + TestNormalImgFullPath
        testLabel = TestSmokeLabel + TestNormalLabel
        print('Total number of training images: ', len(trainPath))
        print('Total number of testing images: ', len(testPath))


        ###################################################### Dataloader ######################################################
        samplers, dataloader = smokeDetectionDataLoader(trainPath, trainLabel, testPath, testLabel, 128).dataloaderReturn()
        # print(dataloader['train'].dataset.targets)
        # profiler.start()

        ###################################################### Model ######################################################
        model = smokeDetectionModel(samplers, dataloader, model_init, 'BCEWithLogitsLoss', 'AdamW', device, rank)
        if rank == 0:
            model_trained, train_accuracy_list, train_loss_list, valid_accuracy_list, valid_loss_list,\
                train_precision_list, train_recall_list, train_f1score_list, train_auroc_list,\
                    valid_precision_list, valid_recall_list, valid_f1score_list, valid_auroc_list = model.modelTraining(args.num_epochs)
        else:
            _ = model.modelTraining(args.num_epochs)
        
        # profiler.stop()
        total_time = time.time() - total_start
        print('Total time at rank {}: {}'.format(rank, total_time))

        
        # Synchronize before saving (optional)
        torch.distributed.barrier()

        if rank == 0:
            trainedModelDir = './trainedModels/'+caseFolder+ "/"+dataCase+'/'
            trainedModelAddr = trainedModelDir + "ResNetTransfer_" + str(args.num_epochs) + "_" + args.deviceArg  + "_" +str(i) + ".pt"
            torch.save(model_trained.module, trainedModelAddr)


        # Delete the DDP model instance to free up resources
        del model_init
        torch.cuda.empty_cache()  # Clears cache to prevent memory build-up
        # Synchronize all ranks before moving to the next iteration
        torch.distributed.barrier()
    
    distDataParallel.cleanup()