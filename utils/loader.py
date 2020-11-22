import torch
from torch.utils.data import DataSet
import os
import os.path
import numpy as np

class ModelNet40(DataSet):
    def __init__(self, path, npoints, test=True):
        self.path = path
        self.npoints = npoints
        self.test = test
        self.input_pairs = self.create_input_list(path, test)
        self.class_num = 40

    def __getitem__(self, idx):
        path, label = self.input_pairs[idx]
        points = self.off_vertex_parser(path)[:][:1024]
        points = torch.unsqueeze(torch.from_numpy(points).float(), 0)
        label = torch.Tensor(label).long()
        return points, label
        

    def __len__(self):
        return len(self.input_pairs)

    def create_input_list(self, path, test):
        input_pairs = []
        gt_key = os.listdir(path)
        for idx, obj in enumerate(gt_key):
            if test:
                path_to_files = osp.join(path, obj, 'test')
            else:
                path_to_files = osp.join(ath, obj, 'train')
            files = os.listdir(path_to_files)
            filepaths = [(os.path.join(path_to_files, file), idx) for file in files]
            input_pairs = input_pairs + filepaths
            
            return input_pairs

    def off_vertex_parser(self, path_to_off_file):
        # Read the OFF file
        with open(path_to_off_file, 'r') as f:
            contents = f.readlines()
        # Find the number of vertices contained
        # (Handle mangled header lines in .off files)
        if contents[0].strip().lower() != 'off':
            num_vertices = int(contents[0].strip()[4:].split(' ')[0])
            start_line = 1
        else:
            num_vertices = int(contents[1].strip().split(' ')[0])
            start_line = 2
        # Convert all the vertex lines to a list of lists
        vertex_list = [map(float, contents[i].strip().split(' ')) for i in range(start_line, start_line+num_vertices)]
        # Return the vertices as a 3 x N numpy array
        return np.array(vertex_list)


