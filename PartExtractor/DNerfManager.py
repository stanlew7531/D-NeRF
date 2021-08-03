import torch
import numpy as np
from run_dnerf import config_parser, create_nerf
from run_dnerf_helpers import to8b
from scipy.spatial.transform import Rotation as R

class DNerfManager(object):
    def __init__(self, config_file):
        # set cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # parse configuration
        parser = config_parser()
        self.args = parser.parse_args(f'--config {config_file}')
        # create D-NeRF model
        _, self.render_kwargs_test, _, _, _ = create_nerf(self.args)

        # set render params
        self.hwf = [400, 400, 555.555]
        self.render_kwargs_test.update({'near' : 2., 'far' : 6.})

        self.camera_intrinsic = np.eye(3)
        self.camera_intrinsic[0,0] = self.hwf[2]
        self.camera_intrinsic[1,1] = self.hwf[2]
        self.camera_intrinsic[0,2] = self.hwf[0] / 2
        self.camera_intrinsic[1,2] = self.hwf[1] / 2

        # extract relevant info
        self.model = self.render_kwargs_test['network_fn']
        self.network_query_fn = self.render_kwargs_test['network_query_fn']
        self.displacement_query_fn = self.render_kwargs_test['displacement_query_fn']

    # points -> N x 3
    # times  -> N x 1
    def get_point_deformations(self, points, times):
        with torch.no_grad():
            results = []
            for time in times:
                times_in = torch.zeros(1,1, dtype=torch.float32).to(times)
                times_in[0,0] = time
                result = self.displacement_query_fn(points, times_in, self.model)
                results += [result.unsqueeze(0)]
            results = torch.cat(results,0)
            results = results.cpu().detach()
        # swap axes is just to keep the sample, time, point indexing scheme
        # vs the time, sample, point indexing the above ^^ loop generates
        return torch.transpose(results, 0, 1)