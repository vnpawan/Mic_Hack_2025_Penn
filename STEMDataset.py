import pandas as pd
import torch.nn.functional
import numpy as np
import torchvision.transforms.v2 as v2

class STEMDataset:

    # init tells the dataset 1. what files exist 2. the batch size
    def __init__(self, files, transform=None):
        # np array of h5 file paths
        self.files = files
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
        keysDf = pd.DataFrame(columns = ['file','h5Keys'])

        for f in self.h5_files:
           with h5py.File(f, 'r') as h5f:
                newKeys = pd.DataFrame({'file': f,'h5Keys': list(h5f.keys())})
                keysDf = pd.concat([keysDf, newKeys], axis = 0, ignore_index = True)
        return keysDf
        
            
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

        f  = self.lookup_table.file[index]
        key = self.lookup_table.h5Keys[index]

        with h5py.File(f, 'r') as h5f:
            group = h5f[key]
            cbed_stack = group['cbed_stack'][()]
            normedCBED = np.empty((0,512,512))
            for cbed in zip(cbed_stack):
                cbed = np.array(cbed)
                cbed = cbed**0.25
                cbed = self.norm(cbed)
                normedCBED = np.append(normedCBED, cbed, axis=0)
                               
            cbed_stack = torch.FloatTensor(normedCBED)

            if self.transform:
                cbed_stack = self.transform(cbed_stack)

            space_group = group.attrs['space_group']
            space_group = np.frombuffer(space_group, dtype=np.uint8)[0]
            space_group_tensor = torch.tensor(space_group, dtype=torch.long)  
            space_group = torch.nn.functional.one_hot(space_group_tensor, num_classes = 230)
            

        return (cbed_stack, space_group)


    def __len__(self):
        return self.total_samples
