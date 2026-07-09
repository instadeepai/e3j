import jax
import jax.numpy as np
from jax import Array

from e3j import utils
from e3j.core.scalar_mixing import ScalarMixing
from e3j.core.tensor_product import TensorProduct
from e3j.ops.coef import Coef4D
from e3j.ops.convolution import CUDAConvolutionParams, convolution
from e3j.spaces import O3Space
from e3j.utils import options
from e3j.utils.cache import cache
from e3j.utils.options import Layout
from e3j.utils.sparse import sparse_bcoo


class Convolution:
    r"""Equivariant message-passing convolution.

    Computes the aggregated message on receiver nodes given by:

    .. math::

        m_b = \frac 1 N \sum_a (x_a \otimes y_{ab}) \odot s_{ab}

    where the sum runs over neighbors $a$ of the receiver node $b$,
    $\otimes$ denotes a tensor product operation, $\odot$ denotes
    a scalar mixing, and $N$ the average number of neighbors.

    The plain JAX implementation consists of the following operations:

    1. Gather node features by senders,
    2. Compute the tensor product of sender features with edge features
       (typically harmonic embeddings),
    3. Mix tensor product outputs with edge scalars (typically MLP of RBF encodings),
    4. Scatter-add messages on receiver nodes.


    Optionally, the sum of messages is rescaled by `avg_num_neighbors`.

    Note:
        When using the CUDA convolution kernel, edges must be sorted by receiver
        index for now (at graph construction time). The adjacency relationship
        is encoded in a sparse CSR format to avoid atomic (memory-locked, non-
        deterministic) operations.
    """

    def __init__(
        self,
        source: tuple[O3Space, O3Space],
        target: O3Space | None = None,
        layout: Layout = Layout.TRAILING_CHANNELS,
        avg_num_neighbors: float | None = None,
        normalization: str | options.TPNormalization = "SQRT_DIM_OUT",
        config: utils.Config | None = None,
    ):
        """
        Initialize a Convolution block from parameters.

        Args:
            source: Representations of the two tensor-product inputs (node and
                edge features). The third source space (mixing scalars) is
                inferred from the former two in the `.source` attribute.
            target: Output representation, inferred by default. Passing a target
                argument enforces a filter on the output irreducible blocks.
            layout: Specifies the channel axis, `TRAILING_CHANNELS` is faster.
            avg_num_neighbors: If given, messages are divided by this factor.
            normalization: Normalization of the tensor product's Clebsch-Gordan
                coefficients, see :class:`e3j.utils.options.TPNormalization`.
            config: Global :class:`e3j.utils.config.Config` (optional) pointing
                to the implementation path. The best available option should be
                automatically selected based on the environment.
        """
        # Tensor product block
        otimes = TensorProduct(
            (source[0], source[1]),
            target,
            normalization=normalization,
            layout=layout,
        )
        # Statically rescale coefficients by num_neighbors to skip
        # the node-wise message rescaling overhead.
        if avg_num_neighbors is not None:
            with jax.ensure_compile_time_eval():
                otimes.coef = sparse_bcoo(
                    values=otimes.values / avg_num_neighbors,
                    indices=otimes.indices,
                    shape=otimes.coef.shape,
                )

        # Scalar mixing block
        mix = ScalarMixing(otimes.target, layout=layout)

        # Source (trilinear) and target
        self.source = (*otimes.source, mix.num_irreps * O3Space("0e"))
        self.target = otimes.target

        # Optional rescaling factor
        self.avg_num_neighbors = avg_num_neighbors
        self.config = utils.config.state() if config is None else config
        self.layout = layout
        self.normalization = otimes.normalization

        self._otimes = otimes
        self._mix = mix

    @cache
    def coef(self) -> Coef4D:
        """Return packed Coef4D coefficients for the forward pass."""
        with jax.ensure_compile_time_eval():
            mix, otimes = self._mix, self._otimes
            idx = otimes.indices.T
            mix_idx = np.array(mix.mix_indices, dtype=idx.dtype)
            coef4D = Coef4D(
                otimes.values,
                np.stack([idx[0], idx[1], idx[2], mix_idx[idx[0]]], axis=-1),
                val_dtype=np.float32,
                idx_dtype=idx.dtype,
            )
            return coef4D

    def _unfused_eval(
        self,
        node_features: Array,
        edge_features: Array,
        edge_scalars: Array,
        senders: Array,
        receivers: Array,
    ) -> Array:
        """Compute receiver messages with a naive plain-JAX implementation."""
        sender_features = node_features[senders]
        messages = self._otimes(sender_features, edge_features)
        messages = self._mix(edge_scalars, messages)
        receiver_feats = (
            np.zeros(
                (node_features.shape[0], *messages.shape[1:]), dtype=messages.dtype
            )
            .at[receivers]
            .add(messages)
        )
        return receiver_feats

    def _fused_eval(
        self,
        node_features: Array,
        edge_features: Array,
        edge_scalars: Array,
        senders: Array,
        receivers: Array,
    ) -> Array:
        """Apply the :func:`e3j.ops.convolution.convolution` primitive.

        This path dispatches to CUDA convolution kernel(s). The backward
        rule calls a dedicated `convolution_bwd()` primitive and kernel,
        while higher order derivatives are implemented in terms of
        `convolution()` and `convolution_bwd()`.

        Note:
            Edges (and edge features) must be sorted by receivers.
        """
        with jax.ensure_compile_time_eval():
            coef4D_packed = self.coef.pack_jax()

        params = CUDAConvolutionParams(
            num_out=self.target.dim,
            num_scalars=self._mix.num_irreps,
        )

        return convolution(
            coef4D_packed,
            node_features,
            edge_features,
            edge_scalars,
            senders,
            receivers,
            params,
        )

    def __call__(
        self,
        node_features: Array,
        edge_features: Array,
        edge_scalars: Array,
        senders: Array,
        receivers: Array,
    ) -> Array:
        """Return sum of messages on receiver nodes.

        Leading axes should match the number of nodes `n0` or number of edges `n1`
        respectively. Feature dimensions `num_x`, `num_y`, `num_scalars` must match
        the dimensions of the `source` attribute.

        Node features and edge scalars may carry a trailing channel axis of size
        `num_channels`, however edge features are expected to not carry one and are
        broadcast with node features along the channel axis.

        Args:
            node_features: array of shape `(num_nodes, num_x, num_channels)`
            edge_features: array of shape `(num_edges, num_y)`
            edge_scalars: array of shape `(num_edges, num_scalars, num_channels)`
            senders: index vector of length num_edges, in bounds [0, num_nodes)
            receivers: index vector of length num_edges, in bounds [0, num_nodes).
                Expected to be monotonically increasing when using the CUDA
                convolution kernel (for sparse CSR adjacency representation).
        """
        match self.config.convolution:
            case options.Convolution.UNFUSED:
                return self._unfused_eval(
                    node_features,
                    edge_features,
                    edge_scalars,
                    senders,
                    receivers,
                )
            case options.Convolution.FUSED_CUDA:
                return self._fused_eval(
                    node_features,
                    edge_features,
                    edge_scalars,
                    senders,
                    receivers,
                )
            case _:
                raise RuntimeError("Unknow convolution implementation.")
