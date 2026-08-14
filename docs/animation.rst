Message passing on TPU
======================

Animation of E3J's fused message-passing convolution forward kernel,
`e3j/pallas_ops/convolution/mosaic_tpu/fwd.py`, which computes:

.. math::

    {\bf m}_b = \frac 1 N \sum_a
        ({\bf x}_a \otimes {\bf y}_{ab}) \odot {\bf w}_{ab}

.. raw:: html

   <video src="_static/streaming_message_passing.mp4"
          controls muted loop playsinline
          style="width:100%;max-width:960px;border-radius:6px;"></video>

The kernel fuses the whole convolution into one pass over receiver-sorted edges.
Per block of edges, the sender features ${\bf x}_a$ are gathered into VMEM, where the
Clebsch-Gordan paths, unrolled at trace time, build the whole block's messages. The block
is then reduced eight edges at a time, each edge accumulating on its own sublane, and a
change of receiver sums the sublanes into one row and writes it out as ${\bf m}_b$ — one
write to HBM per receiver.

The vector registers of a TPU hold 8x128 tiles, and VMEM lays arrays out to match.
Channels are therefore padded to a multiple of 128 lanes, and the harmonics and output
axes to 8 sublanes. The edge block used is 128 and the reduction is made in chunks of 8
edges at a time to fill the TPU's vector registers.

.. note::

    The kernel actually expects *sender*-sorted edges, so that the backward pass,
    which operates on the transposed graph, can be more efficient. Graphs are
    assumed symmetric, so senders and receivers are morally interchangeable.
