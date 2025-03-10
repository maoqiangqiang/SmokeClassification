import torch 
import time 
import numpy as np
import pandas as pd
import os

import sys
sys.path.append('./src/')
from trainingFunc_ResNet import ShuffleDataLoader_Testing, smokeDetectionShuffleTesting 

if __name__ == '__main__':
    
    total_start = time.perf_counter()
    # parse arguments
    num_epochs = int(sys.argv[1])
    deviceRun = str(sys.argv[2])      # 'cuda' or 'cpu'
    saveFlag = bool(int(sys.argv[3]))

    LoopResults_pd = pd.DataFrame()

    deviceArg = 'cuda'     # for trainedModel


    # experiment case 
    scenarioNum = 10
    CSVFolder = "FinalExp"
    caseFolder = "rawSynGen1"
    dataCase = "HTJDSF"




    iList = np.arange(1, scenarioNum+1, 1)
    LoopResults_pd['i'] = iList

    accList = []
    lossList = []
    precisionList = []
    recallList = []
    f1scoreList = []
    augrocList = []
    FPrateList = []
    FNrateList = []



    # # saveFileAddr = './out/HT/'
    saveFileAddr = './out/TestPostprocs/Shuffle/' + caseFolder + "/"+dataCase+'/'
    os.makedirs(saveFileAddr, exist_ok=True)

    for i in range(scenarioNum):
        
        i = i+1
        print("---------------------------------------------")
        print(f"\n Starting Run {i}...")

        # load saved model
        print("[INFO] loading model...")
        trainedModelDir = './trainedModels/' + caseFolder + "/"+dataCase+'/'
        model_load = torch.load(trainedModelDir + 'ResNetTransfer_'+str(num_epochs)+"_"+deviceArg+"_"+str(i)+'.pt', weights_only=False, map_location=torch.device('cpu'))

        # data loader AND datasets
        SmokeDatasetBaseDir = './data/'
        TestSmokeImgPathFile = './datasetCSVPrep/' + CSVFolder +'/testSmokeImgPath_' + str(i)+'.csv'
        TestSmokeImgPath_pd = pd.read_csv(TestSmokeImgPathFile, header=0)
        TestSmokeImgPath = TestSmokeImgPath_pd['imagePath'].tolist()
        TestSmokeImgFullPath = [SmokeDatasetBaseDir + path for path in TestSmokeImgPath]
        TestSmokeLabel = TestSmokeImgPath_pd['label'].tolist()
        
        ## Normal dataset
        NormalDatasetBaseDir = './data/'
        TestNormalImgPathFile = './datasetCSVPrep/' + CSVFolder +'/testNormalImgPath_' + str(i)+'.csv'
        TestNormalImgPath_pd = pd.read_csv(TestNormalImgPathFile, header=0)
        TestNormalImgPath = TestNormalImgPath_pd['imagePath'].tolist()
        TestNormalImgFullPath = [NormalDatasetBaseDir + path for path in TestNormalImgPath]
        TestNormalLabel = TestNormalImgPath_pd['label'].tolist()

        ## combine testing smoke and normal as testing set
        testPath = TestSmokeImgFullPath + TestNormalImgFullPath
        testLabel = TestSmokeLabel + TestNormalLabel
        print("Total number of testing images: ", len(testPath))

        ## the torch.cumsum function, used in the AUROC metric, lacks a deterministic implementation on CUDA, causing a runtime error.
        torch.use_deterministic_algorithms(True, warn_only=True)
        # test loader
        test_loader = ShuffleDataLoader_Testing(testPath, testLabel, 128).dataloaderReturn()

        # model
        model = smokeDetectionShuffleTesting(test_loader, model_load, "BCEWithLogitsLoss", deviceRun)
        # test 
        test_acc, test_loss, test_precision, test_recall, test_f1score, test_auroc, test_FPrate, test_FNrate, saveResultsDataFrame = model.modelShuffleTesting()
        
        saveResultsDataFrame.to_csv(saveFileAddr + 'Scenario_TargetPredict_ShuffleTesting_'+str(num_epochs)+"_"+deviceArg+'_'+str(i)+'.csv', index=False)                                 




        accList.append(test_acc)
        lossList.append(test_loss)
        precisionList.append(test_precision)
        recallList.append(test_recall)
        f1scoreList.append(test_f1score)
        augrocList.append(test_auroc)
        FPrateList.append(test_FPrate)
        FNrateList.append(test_FNrate)

        total_time = time.perf_counter() - total_start
        print('Total time: ', total_time)

    if saveFlag:
        LoopResults_pd['acc'] = accList
        LoopResults_pd['loss'] = lossList
        LoopResults_pd['precision'] = precisionList
        LoopResults_pd['recall'] = recallList
        LoopResults_pd['f1score'] = f1scoreList
        LoopResults_pd['augroc'] = augrocList
        LoopResults_pd['FPrate'] = FPrateList
        LoopResults_pd['FNrate'] = FNrateList

        # add a average row to the dataframe to calculate the average of n loop results
        mean_row = LoopResults_pd.drop(columns=['i']).mean()
        mean_row['i'] = 'Average'  # Add a label for the average row
        # Use pd.concat to append the mean row as a DataFrame
        LoopResults_pd = pd.concat([LoopResults_pd, pd.DataFrame([mean_row])], ignore_index=True)

        LoopResults_pd.to_csv(saveFileAddr + 'Scenario_ResNet_Results_ShuffleTesting'+str(num_epochs)+"_"+deviceArg+'.csv', index=False)
        print(f"Results saved to {saveFileAddr + 'Scenario_ResNet_Results_ShuffleTesting'+str(num_epochs)+'_'+deviceArg+'.csv'}")
