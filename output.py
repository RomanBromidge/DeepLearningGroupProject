# Script to view outputs (list for each checkpoint that shows if the segments where
# correctly classified)

import argparse, torch
import numpy as np
import pickle

dataset = pickle.load(open("UrbanSound8K_test.pkl", 'rb'))

# Code created for viewing the output files
#output = torch.load("output-MLMC.pkl")

#outputView = output["output"]
#print(f"Printed output: {outputView}")

# Script to compare outputs (list for each checkpoint that shows if the segments where
# correctly classified)
# This code was created for the Qualitative Results section of the report
def classFromIdx (index):
    item = dataset[index]
    itemClass = item["class"]
    print (itemClass)

def compareA ():
    outputMC = torch.load("output-MC.pkl")
    outputMC = outputMC["output"]
    outputLMC = torch.load("output-LMC.pkl")
    outputLMC = outputLMC["output"]

    for i in range(len(outputMC)):
        for k in range (0,3):
            if outputMC[i] == outputLMC[i]:
                print (i)

def compareB ():
    outputMC = torch.load("output-MC.pkl")
    outputMC = outputMC["output"]
    outputLMC = torch.load("output-LMC.pkl")
    outputLMC = outputLMC["output"]

    for i in range(len(outputMC)):
        if outputMC[i] != outputLMC[i]:
            print (i)

def compareC ():
    outputMC = torch.load("output-MC.pkl")
    outputMC = outputMC["output"]
    outputLMC = torch.load("output-LMC.pkl")
    outputLMC = outputLMC["output"]
    outputTSCNN = torch.load("output-TSCNN.pkl")
    outputTSCNN = outputTSCNN["output"]

    for i in range(len(outputMC)):
        if (outputMC[i] == False and outputLMC[i] == False and outputTSCNN[i] == True):
                print (i)

def compareD ():
    outputMC = torch.load("output-MC.pkl")
    outputMC = outputMC["output"]

    outputLMC = torch.load("output-LMC.pkl")
    outputLMC = outputLMC["output"]

    outputMLMC = torch.load("output-MLMC.pkl")
    outputMLMC = outputMLMC["output"]

    outputTSCNN = torch.load("output-TSCNN.pkl")
    outputTSCNN = outputTSCNN["output"]

    for i in range(len(outputMC)):
        if (outputMC[i] == False and outputLMC[i] == False and outputMLMC[i] == False and outputTSCNN[i] == False):
                print (i)

classFromIdx(555)
classFromIdx(623)
classFromIdx(666)
classFromIdx(712)
classFromIdx(781)
#compareD()
