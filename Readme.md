# Dataset
The train and test sets of the dataset are saved in UrbanSound8K_train.pkl, and UrbanSound8K_test.pkl.
The dataset is structured as a list of dictionaries. Each dict in the list corresponds to a different audio segment from an audio file. The dicts contain the following keys:

•	filename: contains a unique name of the audio file. This is useful for matching audio segments to the audio file that they are coming from, and compute global scores by averaging the segments scores that have the same filename
•	class: class name
•	classID: class number  [0…9]
•	features: all the features to be used for training. This is a dictionary which contains:
•	logmelspec
•	mfcc
•	chroma
•	spectral_contrast
•	Tonnetz


# Dataloader
In dataset.py, the body of a PyTorch dataloader can be found to load UrbanSound8K dataset. You first have to edit this file to load the different inputs (LMC, MC, and MLMC features) for training your convolutional networks. The code already loads the labels, and the unique identifiers of the files that the audio segments belong to. You have to modify the commented lines.
Then to use it, include the following lines in your code:

from dataset import UrbanSound8KDataset

train_loader = torch.utils.data.DataLoader(
     UrbanSound8KDataset(‘UrbanSound8K_train.pkl’, mode),
     batch_size=32, shuffle=True,
     num_workers=8, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    UrbanSound8KDataset(‘UrbanSound8K_test.pkl’, mode),
    batch_size=32, shuffle=False,
    num_workers=8, pin_memory=True)

for i, (input, target, filename) in enumerate(train_loader):
#training code

for i, (input, target, filename) in enumerate(val_loader):
#validation code

In the code above, input is a batch of 32 log-mel spectrograms, target are their corresponding labels, and filename are the names of the audio files each segment belongs to (useful for testing). The variable mode should take one of the values: ‘LMC’, ‘MC’, ‘MLMC’.
