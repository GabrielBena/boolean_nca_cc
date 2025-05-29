import jax
import functools
from mpi4py import MPI
import mpi4jax
import jax.numpy as jnp
import numpy as np
# from mpi4jax import Allreduce # Redundant if mpi4jax.allreduce is used


def only_rank(target_rank: int):
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    def decorator(fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            if my_rank == target_rank:
                return fn(*args, **kwargs)
            else:
                return None
        return decorated
    return decorator


def tree_all_reduce(tree, comm, **kwargs):
    token = jax.lax.create_token()

    def reduce_leaf_func(leaf):
        nonlocal token
        # Assuming mpi4jax.allreduce is the intended function here
        res, token = mpi4jax.allreduce(leaf, token=token, comm=comm, **kwargs) 
        return res
    return jax.tree_util.tree_map(reduce_leaf_func, tree)


def tree_bcast(tree, root=0, comm=None):
    if comm is None:
        comm = MPI.COMM_WORLD

    def bcast_leaf(leaf):
        is_jax_array = isinstance(leaf, jnp.ndarray)
        np_leaf = np.array(leaf) if is_jax_array else leaf
        comm.Bcast(np_leaf, root=root)
        return jnp.array(np_leaf) if is_jax_array else np_leaf

    return jax.tree_util.tree_map(bcast_leaf, tree)
