import torch
import numpy as np

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

def collate_fn(x_list):
    x = x_list[0]
    for k, v in x.items():
        if isinstance(v, np.ndarray):
            x[k] = torch.from_numpy(v)
        elif isinstance(v, dict):
            for k2, v2 in x[k].items():
                if isinstance(v2, np.ndarray):
                    x[k][k2] = torch.from_numpy(v2)
    return x

def data_to_device(data, device=torch.device("cuda:0")):
    for k, v in data.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    data[k][k2] = v2.to(device)
        elif isinstance(v, torch.Tensor):
            data[k] = v.to(device)
    return data