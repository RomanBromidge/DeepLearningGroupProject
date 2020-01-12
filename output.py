import argparse, torch
import numpy as np

output = torch.load("output-MC.pkl")

outputView = output["output"]
print(f"Printed output: {outputView}")
