import os
import glob
import hyperspy.api as hs
import pandas as pd
import torch.nn.functional
import numpy as np
import torchvision.transforms.v2 as v2

class STEMDataset:

    # init tells the dataset 1. what files exist 2. the batch size
    def __init__(self, path, transform=None):
        # np array of h5 file paths
        self.path = path
        self.transform = transform
        self.lookup_table = self.build_lookup_table()
    
    def build_lookup_table(self):
        # for each file in files the sample keys and filename are
        # stored in a pd dataframe which automatically assigns 
        # a dataset index for each sample using the default
        # RangeIndex. The dataset index can be used to obtain the 
        # filepath and sample key for an arbitrary sample across any file.
        # This in turn allows random sampling by providing random integers
        # in range of total_samples

        # could also add extraction of space group label at this stage

        files = glob.glob(os.path.join(self.path, '*.dm3')) + \
        glob.glob(os.path.join(self.path, '*.dm4'))

        if not files:
            print("No .dm3 or .dm4 files found.")
            return
        
        df = pd.DataFrame(files, columns = ['file'])

        return df
        
    def get_lookup_table(self):
        return self.lookup_table
    
    # normalize the input numpy array between 0 and 1
    def norm(self, x):
        x = (x - x.min())/(x.max() - x.min())
        return x
    
    def __getitem__(self, index):
        # using index in range of total samples, get the file and 
        # the key from lookup_table
        # use h5 to pull the cbed patterns and label from the file
        # return the cbed patterns and label

        # pandas uses index from 0 to length - 1. It appears pytorch does the same

        df = self.get_lookup_table()
        filepaths = df['file']
        data_obj = hs.load(filepaths.iloc[index])

            # Get calibration
        if data_obj.data.ndim > 2:
            cal = data_obj.axes_manager[2].scale
        else:
            cal = data_obj.axes_manager[1].scale

        # Handle stack vs single image
        if data_obj.data.ndim == 3:
            # print(f"  > Summing stack ({data_obj.data.shape[0]} images)")
            image = np.sum(data_obj.data, axis=0)
        else:
            image = data_obj.data

        # apply transform if provided
        if self.transform:
            image = self.transform(image)
            
        return (image, cal)


    def __len__(self):
        df = self.get_lookup_table
        return df.size()
