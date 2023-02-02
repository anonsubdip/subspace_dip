import torch 
import torch as Tensor

def gramschmidt(
        ortho_bases: Tensor, 
        randn_projs: Tensor
    ):
    """
    This methods implements the Gram-Schmidt process, which takes a finite, 
    linearly independent set of vectors S = {ortho_bases_1, ...,
        ortho_bases_{k-num_rand_projs}, ... randn_projs_{k} }, and generates
    an orthogonal set S' = {u1, ..., uk} that spans the same k-dimensional 
    subspace as S. Here, we assume that randn_projs are linearly independent.
    See https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#The_Gram%E2%80%93Schmidt_process.

    """
    
    assert ortho_bases.ndim == randn_projs.ndim
    assert ortho_bases.shape[0] == randn_projs.shape[0]
    
    num_rand_projs = randn_projs.shape[1]
    for i in range(num_rand_projs):
        randn_projs[:, i] -= ( 
                (randn_projs[:, i, None] * ortho_bases).sum(dim=0) * ortho_bases 
            ).sum(dim=-1) 
        randn_projs[:, i] /= torch.norm(randn_projs[:, i], 2)
        ortho_bases = torch.cat( 
                (ortho_bases, randn_projs[:, i, None]), 
                dim=-1
            )
    return ortho_bases

def stats_to_writer(stats, writer, step):
    for key, value in stats.items():
        writer.add_scalar(key, value, step)