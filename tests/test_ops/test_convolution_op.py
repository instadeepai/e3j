import jax
import jax.numpy as np
import jax.random as random
import numpy.testing as testing
import pytest
from jax import Array

import e3j
from e3j.ops.coef import Coef4D
from e3j.ops.convolution import ConvolutionParams, convolution
from e3j.utils.sparse import narrow_index_dtype

e3j.config(debug_level=0)

pytestmark = pytest.mark.e3j_ops

np.set_printoptions(
    precision=4,
    suppress=True,
    formatter={"bool": lambda t: "1" if t else "0"},
    linewidth=90,
)


def pack_coef4d(val, idx4, val_dtype="float32", idx_dtype="int32"):
    """Pack (nnz,) values + (4, nnz) COO indices into an opaque JAX array."""
    return Coef4D(val, idx4.T, val_dtype=val_dtype, idx_dtype=idx_dtype).pack_jax()


def assert_allclose(expect, result, rtol=5e-6, atol=5e-6, debug: int = 1):
    try:
        testing.assert_allclose(expect, result, rtol=rtol, atol=atol)
    except AssertionError as err:
        if debug >= 1:
            print("expect == result\n", abs(expect - result) < atol)
        if debug >= 2:
            print("expect\n", expect)
            print("result\n", result)
        raise err


def convolution_reference(idx, val, x, y, s, s_index, sender, receiver, num_out):
    """Unfused gather -> TP -> mix -> scatter, pure JAX (COO adjacency)."""
    num_nodes = x.shape[0]
    num_edges = sender.shape[0]
    channels_x = x.shape[-1]

    x_e = x[sender]

    z_e = np.zeros((num_edges, num_out, channels_x), dtype=x.dtype)
    cxy = val[:, None] * x_e[:, idx[1], :] * y[:, idx[2], None]
    z_e = z_e.at[:, idx[0], :].add(cxy)

    z = np.zeros((num_nodes, num_out, channels_x), dtype=x.dtype)
    z = z.at[receiver].add(z_e * s[:, s_index, :])
    return z


def make_graph(num_nodes, num_edges, key):
    """Generate COO edge lists and derive CSR for the kernel.

    Returns (sender, receiver) sorted by receiver.
    The kernel gets the CSR (sender, receiver_ptr) made by
    internally via `GraphCSR`.
    """
    k1, k2 = random.split(key)
    sender = random.randint(k1, (num_edges,), 0, num_nodes)
    receiver = random.randint(k2, (num_edges,), 0, num_nodes)

    perm = np.argsort(receiver)

    sender_sorted = sender[perm]
    receiver_sorted = receiver[perm]

    return sender_sorted, receiver_sorted


class _TestConvolutionOp:
    num_idx: int
    num_x: int
    num_y: int
    num_out: int
    num_scalars: int
    channels_x: int
    num_nodes: int = 8
    num_edges: int = 24

    def closure(self):
        nx, ny, nz = self.num_x, self.num_y, self.num_out
        ns = self.num_scalars
        nnz = self.num_idx

        keys = iter(random.split(random.key(42), 5))

        def make_idx(dim, n, key):
            idx_all = np.arange(dim)
            idx_rdm = random.randint(key, (n - dim,), 0, dim - 1)
            return np.concat((idx_all, idx_rdm))

        indices = [
            make_idx(nz, nnz, next(keys)),
            make_idx(nx, nnz, next(keys)),
            make_idx(ny, nnz, next(keys)),
        ]

        sigma = np.argsort(indices[0])
        idx = np.stack(indices)[:, sigma]
        idx = idx.astype(narrow_index_dtype((nz, nx, ny)))
        val = random.normal(next(keys), (nnz,))

        s_index = np.concat(
            (np.arange(ns), random.randint(next(keys), (nz - ns,), 0, ns))
        )
        s_index = np.sort(s_index)
        params = ConvolutionParams(num_out=nz, num_scalars=ns)

        return idx, val, s_index, params

    def graph(self):
        return make_graph(
            self.num_nodes,
            self.num_edges,
            random.key(99),
        )

    def inputs(self):
        n, ne = self.num_nodes, self.num_edges
        nx, ny, ns = self.num_x, self.num_y, self.num_scalars
        cx = self.channels_x

        keys = iter(random.split(random.key(123), 3))
        x = random.normal(next(keys), (n, nx, cx))
        y = random.normal(next(keys), (ne, ny))
        s = random.normal(next(keys), (ne, ns, cx))
        return x, y, s

    @property
    def fwd_ref(self):
        idx, val, s_index, params = self.closure()
        sender, receiver = self.graph()
        return lambda x, y, s: convolution_reference(
            idx,
            val,
            x,
            y,
            s,
            s_index,
            sender,
            receiver,
            params.num_out,
        )

    @property
    def fwd_op(self):
        idx, val, s_index, params = self.closure()
        sender, receiver = self.graph()
        idx4 = np.stack([idx[0], idx[1], s_index[idx[0]], idx[2]])
        coef = pack_coef4d(val, idx4)
        return lambda x, y, s: convolution(
            coef,
            x,
            y,
            s,
            sender,
            receiver,
            params,
        )

    @property
    def bwd_ref(self):
        return jax.grad(
            lambda x, y, s: np.sum(self.fwd_ref(x, y, s)),
            argnums=(0, 1, 2),
        )

    @property
    def bwd_op(self):
        return jax.grad(
            lambda x, y, s: np.sum(self.fwd_op(x, y, s)),
            argnums=(0, 1, 2),
        )

    def test_forward(self):
        x, y, s = self.inputs()
        expect = self.fwd_ref(x, y, s)
        result = self.fwd_op(x, y, s)
        assert_allclose(expect, result, rtol=1e-5, atol=1e-5)

    def test_backward(self):
        x, y, s = self.inputs()
        for expect, result in zip(self.bwd_ref(x, y, s), self.bwd_op(x, y, s)):
            assert_allclose(expect, result, atol=5e-5, rtol=5e-5)


class TestConvolution32(_TestConvolutionOp):
    """N=1, basic."""

    num_idx = 200
    num_x = 9
    num_y = 16
    num_out = 9
    num_scalars = 3
    channels_x = 32


class TestConvolution128(_TestConvolutionOp):
    """N=4, larger irreps."""

    num_idx = 400
    num_x = 16
    num_y = 25
    num_out = 16
    num_scalars = 4
    channels_x = 128


class TestConvolution256(_TestConvolutionOp):
    """Strided path (channels > blockDim.x * N)."""

    num_idx = 400
    num_x = 16
    num_y = 25
    num_out = 16
    num_scalars = 4
    channels_x = 256
