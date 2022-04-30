import torch
import numba
import numpy as np


@numba.njit(cache=True)
def numba_sample_node(weight: np.ndarray, k: int):
    mask = np.zeros(weight.shape, dtype=np.bool_)
    close_set = {-1}
    for row in range(weight.shape[1]):
        idx = np.argsort(weight[:, row])[::-1]
        count = 0
        while True:
            for i in idx:
                if len(close_set) < weight.shape[0] + 1:  # not covered the whole graph
                    if i in close_set:
                        pass
                    else:
                        close_set.add(i)
                        mask[i, row] = True
                        count += 1
                else:
                    mask[i, row] = True
                    count += 1
                if count >= k:
                    break
            if count >= k:
                break

    return mask


def sample_heuristic(weight: torch.Tensor, k: int):
    if k < 0:
        k += weight.shape[0]
        k = max(k, 1)  # in case only 1 node

    if k >= weight.shape[0]:
        return torch.ones_like(weight, dtype=torch.float, device=weight.device)

    mask = numba_sample_node(weight.cpu().numpy(), k)
    return torch.from_numpy(mask).to(torch.float).to(weight.device)


if __name__ == '__main__':
    np.random.seed(42)
    weight = np.random.rand(10, 6)
    k = 9
    mask = numba_sample_node(weight, k)
