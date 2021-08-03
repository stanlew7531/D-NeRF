import torch
import numpy as np
import random
from PartExtractor import DNerfManager

class DeformationField(object):
    def __init__(self, DNerfManager, times, samples_ranges, samples_n, additional_samples = None):
        self.manager = DNerfManager

        x_samples = torch.Tensor(np.linspace(samples_ranges[0,0].item(),samples_ranges[0,1].item(),samples_n[0].item()))
        y_samples = torch.Tensor(np.linspace(samples_ranges[1,0].item(),samples_ranges[1,1].item(),samples_n[1].item()))
        z_samples = torch.Tensor(np.linspace(samples_ranges[2,0].item(),samples_ranges[2,1].item(),samples_n[2].item()))

        grid_x, grid_y, grid_z = torch.meshgrid(x_samples, y_samples, z_samples)

        sample_locations = torch.zeros((samples_n[0].item(),samples_n[1].item(),samples_n[2].item(),3),dtype=torch.float32)
        sample_locations[...,0] = grid_x
        sample_locations[...,1] = grid_y
        sample_locations[...,2] = grid_z

        sample_locations = sample_locations.flatten(0,2).unsqueeze(0)
        if(additional_samples is not None):
            sample_locations = torch.cat((sample_locations, additional_samples), dim = 1)
        #print(sample_locations.shape)
        #print(sample_locations)
        # samples contains the time dependant 
        # after init_samples call, will be of shape [N_samples, N_times, 4, 4]
        self.samples = None
        self.init_samples(sample_locations, times)
        

    def init_samples(self, sample_locations = None, times = None):
        self.num_samples = sample_locations.shape[1] if sample_locations is not None else 0
        self.num_times = times.shape[0] if times is not None else 0
        self.times = times
        self.samples = torch.eye(4).repeat(self.num_samples, self.num_times, 1, 1)

        if sample_locations is not None and times is not None:
            deformation_results = self.manager.get_point_deformations(sample_locations.cuda(), times.cuda())
            self.samples[..., 0:3, 3]  = sample_locations.transpose(0,1).repeat(1, self.num_times, 1) - deformation_results

        self.estimate_sample_transforms()

    def estimate_sample_transforms(self, epsilon = 1e-2, n = int(1e1)):
        pertubations_tfs = torch.eye(4).repeat(n, 1, 1)
        pertubations_tfs[..., 0:3, 3] = ((2 * torch.rand((n,3))) - 1) * epsilon
        
        # doing this in a for loop because #lazy and I don't feel like dealing with batching logic on laptops
        for i_t in range(self.num_times):
            for i_sample in range(self.num_samples):
                # t==0 is canonical pose - don't have to register its samples' rotations
                if(i_t is not 0):
                    # get the original location and where it propogates to at time[i_t]
                    original_location = self.samples[i_sample, 0, :, :]
                    deformed_location = self.samples[i_sample, i_t, :, :]
                    #print(original_location)
                    #print(deformed_location)
                    # apply the pertubations to the original location
                    original_location_pertubations = torch.matmul(pertubations_tfs, original_location)[:, 0:3, 3]
                    # determine where those pertubations propogate to at time[i_t]
                    deformation_results = self.manager.get_point_deformations(original_location_pertubations.unsqueeze(0).cuda(), self.times[i_t].unsqueeze(0).cuda()).squeeze()
                    deformed_location_pertubations = torch.clone(original_location_pertubations)
                    deformed_location_pertubations = deformed_location_pertubations - deformation_results
                    # find least squares rotation matrix via SVD
                    rotation_matrix = self.find_rotation_SVD(original_location_pertubations, deformed_location_pertubations)
                    # fill out the sample's rotation transform & continue
                    self.samples[i_sample, i_t, 0:3, 0:3] = rotation_matrix


    def find_rotation_SVD(self, p, q):
        p_bar = torch.mean(p, 0)
        q_bar = torch.mean(q, 0)
        n = p.shape[0]
        w = torch.eye(n)
        x = p - p_bar
        y = q - q_bar
        s = torch.transpose(x,0,1) @ w @ y
        U, S, V = torch.svd(s)
        Vh = torch.transpose(V,-2,-1)
        Uh = torch.transpose(U,-2, -1)
        det = torch.linalg.det(V @ Uh)
        R = torch.eye(V.shape[1])
        R[-1, -1] = det
        R = V @ R @ Uh
        return R