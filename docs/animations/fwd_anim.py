"""
Animates the design of
``src/e3j/pallas_ops/convolution/mosaic_tpu/fwd.py``:

  1. The equivariant message-passing convolution (what we compute).
  2. The streaming loop: async DMA sender gathers into VMEM,
     tensor-product compute, a reduction that walks the block in
     sublane chunks, ONE flush per receiver group.

Sender nodes are written ``a``, receivers ``b``, and the edge from
one to the other ``ab``, as in :class:`e3j.core.Convolution`.

Render:

    uv run --with manim manim -qh docs/animations/fwd_anim.py StreamingMessagePassing
"""

import re

import numpy as np
from manim import *

config.background_color = "#161821"

# Node identity colors (also used for "messages destined for node b").
NODE_COLORS = [BLUE_C, GREEN_C, YELLOW_C, RED_C, PURPLE_B]

# (sender, receiver), already sorted by receiver.
EDGES_SORTED = [(1, 0), (3, 0), (2, 1), (4, 1), (0, 2), (3, 2), (1, 3), (0, 4)]
EB = 4  # edge_block_size in the toy example
CHUNK = 2  # sublane chunk the reduction walks; 8 (num_sublanes) on TPU

GREY_STROKE = "#5a607a"
TXT = "#e8e8f0"
DIM = "#9aa0b8"


def sym(spec, font_size=32, color=TXT):
    """Formula with ``_ab`` subscripts, via Pango markup rather than LaTeX."""
    markup = re.sub(r"_([a-z]+)", r'<span size="x-small" rise="-2000">\1</span>', spec)
    return MarkupText(markup, font_size=font_size, color=color)


def feat_row(color, w=1.35, h=0.32, opacity=0.6):
    return Rectangle(
        width=w,
        height=h,
        stroke_color=color,
        stroke_width=2,
        fill_color=color,
        fill_opacity=opacity,
    )


def empty_row(w=1.35, h=0.32):
    return Rectangle(
        width=w,
        height=h,
        stroke_color=GREY_STROKE,
        stroke_width=2,
        fill_opacity=0.0,
    )


def edge_card(s, r):
    rect = RoundedRectangle(
        corner_radius=0.08,
        width=0.92,
        height=0.6,
        stroke_color=NODE_COLORS[r],
        stroke_width=2.5,
        fill_color=NODE_COLORS[r],
        fill_opacity=0.16,
    )
    label = Text(f"{s}→{r}", font_size=26, color=TXT)
    label.move_to(rect)
    g = VGroup(rect, label)
    g.sender, g.receiver = s, r
    return g


def column(rows, gap=0.12):
    g = VGroup(*rows)
    g.arrange(DOWN, buff=gap)
    return g


class StreamingMessagePassing(Scene):

    def snapshot(self, name):
        """Hook: a named frame worth grabbing. See ``make_thumbnail.py``."""

    def clear_all(self, keep=()):
        keep = set(id(m) for m in keep)
        movers = [m for m in self.mobjects if id(m) not in keep]
        if movers:
            self.play(*[FadeOut(m) for m in movers], run_time=0.7)

    # ------------------------------------------------------------------
    def construct(self):
        self.chapter_graph()
        cards = self.make_edge_tape()
        self.chapter_streaming(cards)

    # ------------------------------------------------------------------
    def make_graph(self, center=ORIGIN, radius=2.1):
        angles = [90, 162, 234, 306, 18]
        nodes = VGroup()
        for i, a in enumerate(angles):
            p = center + radius * np.array(
                [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a)), 0]
            )
            circ = Circle(
                radius=0.33,
                stroke_color=NODE_COLORS[i],
                stroke_width=3,
                fill_color=NODE_COLORS[i],
                fill_opacity=0.25,
            ).move_to(p)
            lab = Text(str(i), font_size=28, color=TXT).move_to(p)
            nodes.add(VGroup(circ, lab))
        arrows = VGroup()
        for s, r in EDGES_SORTED:
            a = Arrow(
                nodes[s].get_center(),
                nodes[r].get_center(),
                buff=0.42,
                stroke_width=2.5,
                color=GREY_STROKE,
                max_tip_length_to_length_ratio=0.08,
            )
            arrows.add(a)
        return nodes, arrows

    def chapter_graph(self):
        head = Text(
            "1.  Equivariant message-passing convolution", font_size=34, color=DIM
        ).to_corner(UL)
        self.play(FadeIn(head))

        nodes, arrows = self.make_graph(center=2.9 * LEFT + 0.2 * DOWN)
        self.play(
            LaggedStart(*[FadeIn(n, scale=0.6) for n in nodes], lag_ratio=0.12),
            run_time=1.2,
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.07), run_time=1.4
        )

        # Each node carries a feature block.
        feat_note = sym("every node a carries features  x_a", font_size=26)
        feat_note.move_to(3.4 * RIGHT + 2.6 * UP)
        self.play(FadeIn(feat_note, shift=0.3 * UP))

        # Demo: one message travelling along edge 3 -> 0.
        a, b = 3, 0
        edge_idx = EDGES_SORTED.index((3, 0))
        msg = feat_row(NODE_COLORS[a], w=0.5, h=0.3)
        msg.move_to(nodes[a].get_center())
        formula = sym("m_ab = ( x_a ⊗ Y_ab ) ⊙ w_ab", font_size=32)
        formula.move_to(3.4 * RIGHT + 1.1 * UP)
        f_caption = Text("sender feats × harmonics × weights", font_size=22, color=DIM)
        f_caption.next_to(formula, DOWN, buff=0.25)

        self.play(Indicate(nodes[a], color=NODE_COLORS[a]), FadeIn(msg))
        self.play(FadeIn(formula), FadeIn(f_caption))
        path = Line(nodes[a].get_center(), nodes[b].get_center())
        self.play(
            MoveAlongPath(msg, path),
            arrows[edge_idx].animate.set_color(NODE_COLORS[b]),
            run_time=1.6,
        )
        self.play(
            msg.animate.set_fill(NODE_COLORS[b]).set_stroke(NODE_COLORS[b]),
            Flash(nodes[b], color=NODE_COLORS[b], flash_radius=0.55),
        )
        self.play(FadeOut(msg))

        # out_b = Σ_{a→b} m_ab  (Σ condition rendered under the sigma).
        of_lhs = sym("out_b =", font_size=32)
        of_sigma = Text("Σ", font_size=42, color=TXT)
        of_cond = Text("a → b", font_size=16, color=TXT)
        of_cond.next_to(of_sigma, DOWN, buff=0.02)
        of_rhs = sym("m_ab", font_size=32)
        out_formula = VGroup(of_lhs, VGroup(of_sigma, of_cond), of_rhs).arrange(
            RIGHT, buff=0.14
        )
        out_formula.move_to(3.4 * RIGHT + 0.5 * DOWN)
        o_caption = Text(
            "receivers sum their incoming messages", font_size=22, color=DIM
        )
        o_caption.next_to(out_formula, DOWN, buff=0.25)
        self.play(FadeIn(out_formula), FadeIn(o_caption))

        # Pulse all messages along their edges to make "sum" concrete.
        pulses, recolor, dots = [], [], []
        for i, (sender, receiver) in enumerate(EDGES_SORTED):
            dot = Dot(color=NODE_COLORS[receiver], radius=0.09).move_to(
                nodes[sender].get_center()
            )
            dots.append(dot)
            pulses.append(
                MoveAlongPath(
                    dot,
                    Line(nodes[sender].get_center(), nodes[receiver].get_center()),
                )
            )
            recolor.append(arrows[i].animate.set_color(NODE_COLORS[receiver]))
            self.add(dot)
        self.play(LaggedStart(*pulses, lag_ratio=0.05), *recolor, run_time=1.8)
        self.play(*[FadeOut(d) for d in dots], run_time=0.5)
        scale_note = Text(
            "real scale:  ~10⁴ nodes,  ~10⁵ edges",
            font_size=24,
            color=YELLOW_C,
        )
        scale_note.move_to(3.4 * RIGHT + 2.0 * DOWN)
        self.play(FadeIn(scale_note, shift=0.3 * UP))
        self.wait(2)
        self.clear_all()

    # ------------------------------------------------------------------
    def make_edge_tape(self):
        # host-sorted edges, laid out as the bottom "edge stream" tape
        cards = VGroup(*[edge_card(s, r) for s, r in EDGES_SORTED])
        cards.arrange(RIGHT, buff=0.18)
        cards.scale(0.95).move_to(3.3 * DOWN)
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.7) for c in cards], lag_ratio=0.06),
            run_time=1.0,
        )
        return cards

    # ------------------------------------------------------------------
    def chapter_streaming(self, cards):
        head = Text("2.  The streaming loop", font_size=34, color=DIM).to_corner(UL)
        tape_lbl = Text("edge stream", font_size=18, color=DIM).next_to(
            cards, DOWN, buff=0.08
        )
        self.play(FadeIn(head), FadeIn(tape_lbl))

        # ---------- memory boxes -------------------------------------
        hbm_box = Rectangle(
            width=4.6, height=5.0, stroke_color=GREY_STROKE, stroke_width=2.5
        ).move_to(4.4 * LEFT + 0.4 * UP)
        hbm_lbl = Text("HBM  — large, slow", font_size=26, color=TXT)
        hbm_lbl.next_to(hbm_box.get_top(), DOWN, buff=0.18)

        nf_rows = column([feat_row(NODE_COLORS[i]) for i in range(5)])
        nf_rows.move_to(hbm_box.get_center() + 1.15 * LEFT + 0.2 * UP)
        nf_lbl = Text("node feats", font_size=22, color=DIM)
        nf_lbl.next_to(nf_rows, UP, buff=0.18)
        nf_idx = VGroup(
            *[
                Text(str(i), font_size=20, color=NODE_COLORS[i]).next_to(
                    nf_rows[i], LEFT, buff=0.15
                )
                for i in range(5)
            ]
        )

        out_rows = column([empty_row() for _ in range(5)])
        out_rows.move_to(hbm_box.get_center() + 1.15 * RIGHT + 0.2 * UP)
        out_lbl = Text("out", font_size=22, color=DIM)
        out_lbl.next_to(out_rows, UP, buff=0.18)
        out_idx = VGroup(
            *[
                Text(str(i), font_size=20, color=NODE_COLORS[i]).next_to(
                    out_rows[i], RIGHT, buff=0.15
                )
                for i in range(5)
            ]
        )

        vmem_box = Rectangle(
            width=6.2, height=4.4, stroke_color=GREY_STROKE, stroke_width=2.5
        ).move_to(3.1 * RIGHT + 0.8 * UP)
        vmem_lbl = Text("VMEM  — small, fast", font_size=24, color=TXT)
        vmem_lbl.next_to(vmem_box.get_top(), DOWN, buff=0.18)

        stage_rows = column([empty_row() for _ in range(EB)])
        stage_rows.move_to(vmem_box.get_center() + 1.85 * LEFT + 0.3 * UP)
        stage_lbl = Text("node feats", font_size=20, color=DIM)
        stage_lbl.next_to(stage_rows, UP, buff=0.15)

        msg_rows = column([empty_row() for _ in range(EB)])
        msg_rows.move_to(vmem_box.get_center() + 1.85 * RIGHT + 0.3 * UP)
        msg_lbl = Text("messages", font_size=20, color=DIM)
        msg_lbl.next_to(msg_rows, UP * 0.95, buff=0.15)

        tp_arrow = Arrow(
            stage_rows.get_right(),
            msg_rows.get_left(),
            buff=0.15,
            color=TXT,
            stroke_width=3,
        )
        tp_lbl = sym("⊗ Y_ab ⊙ w_ab", font_size=24)
        tp_lbl.next_to(tp_arrow, UP, buff=0.1)

        # one accumulator row per chunk sublane; summed only on the way to HBM
        acc_rows = column(
            [feat_row(GREY_STROKE, opacity=0.0) for _ in range(CHUNK)], gap=0.09
        )
        for row in acc_rows:
            row.set_stroke(GREY_STROKE, width=2.5)
        acc_rows.move_to(vmem_box.get_center() + 1.85 * LEFT + 1.35 * DOWN)
        acc_lbl = Text("accumulator", font_size=22, color=DIM)
        acc_lbl.next_to(acc_rows, RIGHT, buff=0.25)

        # cur_receiver lives in SMEM — drawn OUTSIDE the VMEM box, flush with its right edge.
        curr_box = Rectangle(width=0.7, height=0.5, stroke_color=TXT, stroke_width=2)
        curr_box.move_to(1.85 * DOWN).align_to(vmem_box, RIGHT)
        curr_val = Text("–1", font_size=24, color=TXT).move_to(curr_box)
        curr_lbl = Text("receiver (SMEM)", font_size=17, color=DIM)
        curr_lbl.next_to(curr_box, DOWN, buff=0.1)

        self.play(
            Create(hbm_box),
            Create(vmem_box),
            FadeIn(hbm_lbl),
            FadeIn(vmem_lbl),
            run_time=1.0,
        )
        self.play(
            FadeIn(nf_rows),
            FadeIn(nf_lbl),
            FadeIn(nf_idx),
            FadeIn(out_rows),
            FadeIn(out_lbl),
            FadeIn(out_idx),
            FadeIn(stage_rows),
            FadeIn(stage_lbl),
            FadeIn(msg_rows),
            FadeIn(msg_lbl),
            GrowArrow(tp_arrow),
            FadeIn(tp_lbl),
            FadeIn(acc_rows),
            FadeIn(acc_lbl),
            FadeIn(curr_box),
            FadeIn(curr_val),
            FadeIn(curr_lbl),
            run_time=1.4,
        )
        self.wait(1.0)

        caption = Text(" ", font_size=24, color=TXT).move_to(2.45 * DOWN)

        def say(msg, color=TXT, t=0.6):
            nonlocal caption
            new = Text(msg, font_size=24, color=color)
            new.move_to(2.45 * DOWN)
            self.play(FadeOut(caption), FadeIn(new), run_time=t)
            caption = new

        # ---------- block window -------------------------------------
        window = SurroundingRectangle(
            VGroup(*cards[0:EB]), color=WHITE, stroke_width=3, buff=0.08
        )
        block_lbl = VGroup(
            Text("block", font_size=17, color=WHITE),
            Text(f"{EB} edges here", font_size=15, color=DIM),
            Text("128 on TPU", font_size=15, color=DIM),
        ).arrange(DOWN, buff=0.06)
        block_lbl.move_to(5.6 * LEFT + 3.25 * DOWN)
        say("the pipeline streams one block of edges at a time")
        self.play(Create(window), FadeIn(block_lbl))
        self.wait(1.6)

        block1 = EDGES_SORTED[0:EB]

        # ---------- gather senders (async DMA) -----------------------
        def dma_arrow(s, k):
            return Arrow(
                nf_rows[s].get_right(),
                stage_rows[k].get_left(),
                buff=0.1,
                color=NODE_COLORS[s],
                stroke_width=3,
                max_tip_length_to_length_ratio=0.06,
            )

        say("copy the needed sender node features to VMEM", color=GREEN_C)
        self.wait(1.5)
        async_arrows = [dma_arrow(s, k) for k, (s, r) in enumerate(block1)]
        ghosts = [nf_rows[s].copy().set_fill(opacity=0.35) for s, _ in block1]
        self.play(
            LaggedStart(*[GrowArrow(a) for a in async_arrows], lag_ratio=0.06),
            run_time=0.7,
        )
        self.play(
            *[g.animate.move_to(stage_rows[k]) for k, g in enumerate(ghosts)],
            run_time=0.8,
        )
        for k, (s, r) in enumerate(block1):
            self.remove(ghosts[k])
            stage_rows[k].set_fill(NODE_COLORS[s], opacity=0.55)
            stage_rows[k].set_stroke(NODE_COLORS[s])
        self.play(*[FadeOut(a) for a in async_arrows])
        self.wait(0.4)

        # ---------- compute ------------------------------------------
        say("tensor product: staged row × harmonics × weights")
        self.wait(1.5)
        msg_fills = []
        tp_anims = []
        for k, (s, r) in enumerate(block1):
            ghost = stage_rows[k].copy()
            self.add(ghost)
            msg_fills.append(ghost)
            msg_rows[k].set_stroke(NODE_COLORS[r])
            tp_anims.append(
                ghost.animate.move_to(msg_rows[k])
                .set_fill(NODE_COLORS[r], opacity=0.55)
                .set_stroke(NODE_COLORS[r])
            )
        self.play(
            Indicate(tp_lbl, color=YELLOW_C),
            LaggedStart(*tp_anims, lag_ratio=0.12),
            run_time=1.6,
        )
        self.wait(1.0)
        say("one message per edge - colored by its receiver", color=DIM)
        self.wait(3.0)

        # ---------- the flush walk ------------------------------------
        say("the block is reduced one chunk of two edges at a time")
        self.wait(1.5)

        chunk_win = SurroundingRectangle(
            VGroup(*cards[0:CHUNK]), color=YELLOW_C, stroke_width=2.5, buff=0.03
        )
        # legend kept clear of the caption line, so it can stay up throughout
        chunk_lbl = VGroup(
            Text("chunk", font_size=17, color=YELLOW_C),
            Text(f"{CHUNK} edges here", font_size=15, color=DIM),
            Text("8 on TPU", font_size=15, color=DIM),
        ).arrange(DOWN, buff=0.06)
        chunk_lbl.move_to(5.6 * RIGHT + 3.25 * DOWN)

        def move_chunk(start, fast=False):
            target = VGroup(*cards[start : start + CHUNK])
            self.play(
                chunk_win.animate.move_to(target.get_center()),
                run_time=0.25 if fast else 0.4,
            )

        def set_curr(v, color):
            nonlocal curr_val
            new = Text(str(v), font_size=24, color=color).move_to(curr_box)
            self.play(Transform(curr_val, new), run_time=0.3)

        def accumulate(rows, r, msg_fills, first_card, fast=False):
            """Add the chunk rows of receiver `r`, each edge on its own sublane."""
            rt = 0.5 if fast else 0.8
            ghosts = [msg_fills[k].copy() for k in rows]
            self.play(
                *[
                    Indicate(
                        cards[first_card + k], color=NODE_COLORS[r], scale_factor=1.15
                    )
                    for k in rows
                ],
                *[
                    g.animate.move_to(acc_rows[k % CHUNK]).stretch_to_fit_width(
                        acc_rows[k % CHUNK].width
                    )
                    for g, k in zip(ghosts, rows)
                ],
                run_time=rt,
            )
            for ghost in ghosts:
                self.remove(ghost)
            for k in rows:
                acc_rows[k % CHUNK].set_fill(NODE_COLORS[r], opacity=0.7)
                acc_rows[k % CHUNK].set_stroke(NODE_COLORS[r])

        def flush(old_r, fast=False):
            """Sum the sublanes into one staged row, then write that row to HBM."""
            rt = 0.45 if fast else 0.8
            ghosts = [row.copy() for row in acc_rows]
            self.play(
                *[g.animate.move_to(acc_rows.get_center()) for g in ghosts],
                run_time=0.3 if fast else 0.45,
            )
            for ghost in ghosts:
                self.remove(ghost)
            staged = feat_row(NODE_COLORS[old_r], w=acc_rows.width, opacity=0.7)
            staged.move_to(acc_rows)
            self.add(staged)
            self.play(
                staged.animate.move_to(out_rows[old_r]).stretch_to_fit_width(
                    out_rows[old_r].width
                ),
                run_time=rt,
            )
            self.remove(staged)
            out_rows[old_r].set_fill(NODE_COLORS[old_r], opacity=0.7)
            out_rows[old_r].set_stroke(NODE_COLORS[old_r])
            if old_r == 0:
                self.snapshot("first-flush")
            self.play(
                Flash(out_rows[old_r], color=NODE_COLORS[old_r], flash_radius=0.6),
                *[
                    row.animate.set_fill(opacity=0).set_stroke(GREY_STROKE)
                    for row in acc_rows
                ],
                run_time=0.5 if fast else 0.7,
            )

        # chunk 0 (edges 0-1): one receiver, so the whole chunk lands in one add
        self.play(FadeIn(chunk_win), FadeIn(chunk_lbl))
        set_curr(0, NODE_COLORS[0])
        say("one receiver in the chunk → both edges at once", t=0.45)
        self.wait(1.5)
        accumulate([0, 1], 0, msg_fills, 0)
        self.wait(1.0)

        # chunk 1 (edges 2-3): its first receiver differs -> flush, then one add
        move_chunk(CHUNK)
        say("new receiver → sum the rows, write out[0]", color=YELLOW_C, t=0.5)
        self.wait(1.2)
        flush(0)
        set_curr(1, NODE_COLORS[1])
        self.wait(0.6)
        accumulate([2, 3], 1, msg_fills, 0)

        # ---------- block 2, fast ------------------------------------
        say("next block - accumulator carries across the boundary", t=0.5)
        block2 = EDGES_SORTED[EB:]
        self.play(
            window.animate.move_to(VGroup(*cards[EB:]).get_center()),
            FadeOut(chunk_win),
            run_time=0.6,
        )

        # fast DMA + compute
        async_arrows = [dma_arrow(s, k) for k, (s, r) in enumerate(block2)]
        ghosts = [nf_rows[s].copy().set_fill(opacity=0.35) for s, _ in block2]
        self.play(
            LaggedStart(*[GrowArrow(a) for a in async_arrows], lag_ratio=0.05),
            run_time=0.5,
        )
        self.play(
            *[g.animate.move_to(stage_rows[k]) for k, g in enumerate(ghosts)],
            run_time=0.5,
        )
        for k, (s, r) in enumerate(block2):
            self.remove(ghosts[k])
            stage_rows[k].set_fill(NODE_COLORS[s], opacity=0.55)
            stage_rows[k].set_stroke(NODE_COLORS[s])
        self.play(*[FadeOut(a) for a in async_arrows], run_time=0.3)

        for f in msg_fills:
            self.remove(f)
        msg_fills2 = []
        tp_anims = []
        for k, (s, r) in enumerate(block2):
            ghost = stage_rows[k].copy()
            self.add(ghost)
            msg_fills2.append(ghost)
            tp_anims.append(
                ghost.animate.move_to(msg_rows[k])
                .set_fill(NODE_COLORS[r], opacity=0.55)
                .set_stroke(NODE_COLORS[r])
            )
        self.play(LaggedStart(*tp_anims, lag_ratio=0.08), run_time=0.9)

        chunk_win.move_to(VGroup(*cards[EB : EB + CHUNK]).get_center())
        self.play(FadeIn(chunk_win), run_time=0.3)

        # chunk 2 (edges 4-5): flush the carried receiver, then one add
        flush(1, fast=True)
        set_curr(2, NODE_COLORS[2])
        accumulate([0, 1], 2, msg_fills2, EB, fast=True)

        # chunk 3 (edges 6-7): the receiver changes mid-chunk, so the mask splits it
        move_chunk(EB + CHUNK, fast=True)
        say("two receivers → the edges are added separately", color=YELLOW_C, t=0.5)
        self.wait(1.5)
        flush(2, fast=True)
        set_curr(3, NODE_COLORS[3])
        accumulate([2], 3, msg_fills2, EB, fast=True)
        flush(3, fast=True)
        set_curr(4, NODE_COLORS[4])
        accumulate([3], 4, msg_fills2, EB, fast=True)

        # final flush
        say(
            "edges exhausted → one final flush for the last group",
            color=YELLOW_C,
            t=0.5,
        )
        self.wait(1.0)
        flush(4, fast=True)
        self.wait(1.0)
        self.play(
            FadeOut(chunk_win), FadeOut(chunk_lbl), FadeOut(window), FadeOut(block_lbl)
        )

        done1 = Text(
            "5 receivers  →  5 HBM writes",
            font_size=28,
            color=GREEN_C,
        )
        done1.move_to(2.5 * DOWN)
        self.play(FadeOut(caption), FadeIn(done1))
        self.play(
            LaggedStart(
                *[Indicate(r, color=NODE_COLORS[i]) for i, r in enumerate(out_rows)],
                lag_ratio=0.15,
            ),
            run_time=1.6,
        )
        self.wait(2.0)
        self.clear_all()
