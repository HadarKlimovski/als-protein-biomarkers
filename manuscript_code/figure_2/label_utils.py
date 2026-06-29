"""Post-pass overlap resolver for matplotlib Text artists.

After adjust_text, iteratively nudge overlapping label pairs apart along the
axis with the smaller overlap dimension (so labels move the shortest distance
needed to clear). Operates in data coordinates.
"""
from __future__ import annotations
import math
import numpy as np


def _data_per_pixel(ax):
    inv = ax.transData.inverted()
    p0 = inv.transform((0, 0))
    p1 = inv.transform((1, 1))
    return abs(p1[0] - p0[0]), abs(p1[1] - p0[1])


def resolve_label_overlaps(ax, texts, max_iter=150, step_px=4, y_only=False,
                           xlim=None, ylim=None):
    """Iteratively push overlapping label pairs apart.

    step_px is the per-iteration push in display pixels (converted to data
    units per axis). When y_only=True the labels are nudged vertically
    regardless of which overlap dimension is smaller (so rail-aligned labels
    stay on the rail). xlim/ylim, if given, clamp positions so labels never
    drift outside the visible plot.
    """
    if not texts:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dxp, dyp = _data_per_pixel(ax)
    sx, sy = step_px * dxp, step_px * dyp
    xmin, xmax = (xlim if xlim is not None else (-1e18, 1e18))
    ymin, ymax = (ylim if ylim is not None else (-1e18, 1e18))
    def _clamp_all():
        for t in texts:
            x, y = t.get_position()
            t.set_position((min(max(x, xmin), xmax), min(max(y, ymin), ymax)))
    for _ in range(max_iter):
        moved = False
        bboxes = [t.get_window_extent(renderer=renderer) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bi, bj = bboxes[i], bboxes[j]
                if not bi.overlaps(bj):
                    continue
                overlap_w = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                overlap_h = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                cxi = (bi.x0 + bi.x1) / 2
                cyi = (bi.y0 + bi.y1) / 2
                cxj = (bj.x0 + bj.x1) / 2
                cyj = (bj.y0 + bj.y1) / 2
                xi, yi = texts[i].get_position()
                xj, yj = texts[j].get_position()
                if (not y_only) and overlap_w <= overlap_h:
                    if cxi >= cxj:
                        texts[i].set_position((xi + sx, yi))
                        texts[j].set_position((xj - sx, yj))
                    else:
                        texts[i].set_position((xi - sx, yi))
                        texts[j].set_position((xj + sx, yj))
                else:
                    if cyi >= cyj:
                        texts[i].set_position((xi, yi + sy))
                        texts[j].set_position((xj, yj - sy))
                    else:
                        texts[i].set_position((xi, yi - sy))
                        texts[j].set_position((xj, yj + sy))
                moved = True
        if not moved:
            break
        _clamp_all()
        fig.canvas.draw()
    _clamp_all()


def place_edge_labels(ax, items, xlim, ylim, fontsize,
                      x_offset_frac=0.03, min_y_gap_frac=0.0,
                      leader_color="0.55", leader_lw=0.4,
                      fontweight_overrides=None,
                      anchor="auto"):
    """Place labels at left/right rails with leader lines to their dots.

    For "overflow" labels — those whose dot lies inside the rail label's text
    body (which happens after the rail is capped to keep text inside the
    axes) — the label is placed directly above the dot with a short vertical
    leader, avoiding text-cutting and label-on-dot overlap.

    items: iterable of (x_dot, y_dot, name) tuples.
      - x_dot > 0  → label goes to the RIGHT
      - x_dot <= 0 → label goes to the LEFT
    """
    x0, x1 = xlim
    y0, y1 = ylim

    # Measure each label's width and height in DATA units via the renderer.
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def _dims_data(text):
        t = ax.text(x0, y0, text, fontsize=fontsize)
        bb = t.get_window_extent(renderer=renderer)
        t.remove()
        p_origin = inv.transform((0, 0))
        p_w      = inv.transform((bb.width, 0))
        p_h      = inv.transform((0, bb.height))
        return abs(p_w[0] - p_origin[0]), abs(p_h[1] - p_origin[1])

    widths  = {it[2]: _dims_data(it[2])[0] for it in items}
    heights = {it[2]: _dims_data(it[2])[1] for it in items}
    max_text_w = max(widths.values(), default=0.0)
    avg_text_h = (sum(heights.values()) / max(1, len(heights))) if heights else 0.0

    x_offset = x_offset_frac * (x1 - x0)
    # Vertical gap: at least average label height × 2.0 so consecutive labels
    # have unambiguous separation. The renderer-measured text bbox is tight
    # against the glyphs, so a 1.0× gap leaves them touching; 1.4× still
    # blurs at 4pt; 2.0× guarantees a visible whitespace strip.
    min_gap = max(min_y_gap_frac * (y1 - y0), avg_text_h * 2.0)
    pad = 0.01 * (x1 - x0)

    # anchor controls how labels are packed along the rail:
    #   "top"    — all labels stacked at the top of their rail (volcano style)
    #   "bottom" — all stacked at the bottom
    #   "auto"   — labels with dot above ylim midpoint pack top, below pack
    #              bottom (scatter style — labels stay close to their dots)
    y_mid = 0.5 * (y0 + y1)
    if anchor == "top":
        in_top = lambda y: True
    elif anchor == "bottom":
        in_top = lambda y: False
    else:
        in_top = lambda y: y >= y_mid

    # Sort labels so leader lines fan out without crossing each other.
    # Labels stacked vertically on a rail produce a non-crossing fan iff
    # they are ordered along the rail by the angle from the rail's anchor
    # corner to each dot. (Tuple-sorts by y-then-x can still cross when
    # dots vary in both axes.) For each side+anchor combo, the anchor is
    # the corner of the rail closest to the first slot; sorting items by
    # angle in the direction the stack grows guarantees non-crossing.
    right_top_items = [it for it in items if it[0] >  0 and     in_top(it[1])]
    right_bot_items = [it for it in items if it[0] >  0 and not in_top(it[1])]
    left_top_items  = [it for it in items if it[0] <= 0 and     in_top(it[1])]
    left_bot_items  = [it for it in items if it[0] <= 0 and not in_top(it[1])]
    # We can't compute angles yet because rail_right / rail_left are derived
    # from the full item set below — finalize sorting after rail positions
    # are known. Build temporary lists that include all candidates.

    # Rail positions: pushed comfortably past the dot cluster so the leader
    # line has room to elbow out. Capped so the text body (extending outward
    # from the anchor) stays inside the axes.
    right_all = right_top_items + right_bot_items
    left_all  = left_top_items + left_bot_items
    rail_right_cluster = (max(it[0] for it in right_all) + x_offset * 1.5) if right_all else x1 - x_offset
    rail_right_cap     = x1 - max_text_w - pad
    rail_right         = min(rail_right_cluster, rail_right_cap)

    rail_left_cluster  = (min(it[0] for it in left_all) - x_offset * 1.5) if left_all else x0 + x_offset
    rail_left_cap      = x0 + max_text_w + pad
    rail_left          = max(rail_left_cluster, rail_left_cap)

    # Angular sort from the rail's anchor corner. For each side+anchor:
    #   * the anchor is the corner of the rail nearest the first stacked slot
    #   * items are sorted so the leader angle changes monotonically along
    #     the rail (fan order = no crossings)
    def _ang(ax_, ay_):
        return lambda it: math.atan2(it[1] - ay_, it[0] - ax_)
    # Right rail top: anchor=(rail_right, y1). Angles in (−π, −π/2).
    # Topmost slot = most negative angle (most leftward dot) → sort ASC.
    right_top = sorted(right_top_items, key=_ang(rail_right, y1))
    # Right rail bot: anchor=(rail_right, y0). Angles in (π/2, π).
    # Bottommost slot (placed first) = most positive (most leftward) → DESC.
    right_bot = sorted(right_bot_items, key=_ang(rail_right, y0), reverse=True)
    # Left rail top: anchor=(rail_left, y1). Angles in (−π/2, 0).
    # Topmost slot = least negative (most rightward dot) → DESC.
    left_top  = sorted(left_top_items,  key=_ang(rail_left,  y1), reverse=True)
    # Left rail bot: anchor=(rail_left, y0). Angles in (0, π/2).
    # Bottommost slot (placed first) = smallest angle (most rightward) → ASC.
    left_bot  = sorted(left_bot_items,  key=_ang(rail_left,  y0))

    def _is_overflow_right(x_dot, name):
        # Dot lies inside the would-be text body (or close enough that the
        # leader stub would overlap the dot).
        return x_dot > rail_right - pad

    def _is_overflow_left(x_dot, name):
        return x_dot < rail_left + pad

    def _stack(group, anchor):
        """Pack labels tightly at the rail's anchor (top or bottom) at
        min_gap spacing. ``anchor`` ∈ {"top", "bottom"} controls which axis
        boundary the stack starts from. Labels are placed in the order
        passed (already sorted: top→ largest dot y first when anchor=top;
        bottom → smallest dot y first when anchor=bottom).
        """
        if not group:
            return []
        n = len(group)
        if anchor == "top":
            top = y1 - min_gap * 0.5
            return [top - i * min_gap for i in range(n)]
        # bottom
        bot = y0 + min_gap * 0.5
        return [bot + i * min_gap for i in range(n)]

    overrides = fontweight_overrides or {}
    annotations = []
    all_text_artists = []

    for side_items, side, ha, anchor, is_overflow in [
        (right_top, "right", "left",  "top",    _is_overflow_right),
        (right_bot, "right", "left",  "bottom", _is_overflow_right),
        (left_top,  "left",  "right", "top",    _is_overflow_left),
        (left_bot,  "left",  "right", "bottom", _is_overflow_left),
    ]:
        rail_items     = [it for it in side_items if not is_overflow(it[0], it[2])]
        overflow_items = [it for it in side_items if     is_overflow(it[0], it[2])]

        ys_adj = _stack(rail_items, anchor)
        x_label = rail_right if side == "right" else rail_left

        for (x_dot, y_dot, name), y_lbl in zip(rail_items, ys_adj):
            fw = overrides.get(name, "normal")
            # Straight diagonal leader from label anchor to dot. Because all
            # rail labels share x=x_label, no two leaders converge at the
            # rail — they fan outward toward each dot's true position.
            ann = ax.annotate(
                name, xy=(x_dot, y_dot), xytext=(x_label, y_lbl),
                ha=ha, va="center", fontsize=fontsize, fontweight=fw,
                arrowprops=dict(arrowstyle="-", color=leader_color, lw=leader_lw,
                                shrinkA=2.5, shrinkB=2.5),
            )
            annotations.append(ann)
            all_text_artists.append(ann)

        # Overflow labels — place above (top anchor) or below (bottom
        # anchor) the dot with a short vertical leader.
        for x_dot, y_dot, name in overflow_items:
            fw = overrides.get(name, "normal")
            if anchor == "top":
                y_lbl = min(y_dot + heights[name] * 1.6, y1 - min_gap * 0.5)
                va = "bottom"
            else:
                y_lbl = max(y_dot - heights[name] * 1.6, y0 + min_gap * 0.5)
                va = "top"
            ann = ax.annotate(
                name, xy=(x_dot, y_dot), xytext=(x_dot, y_lbl),
                ha="center", va=va, fontsize=fontsize, fontweight=fw,
                arrowprops=dict(arrowstyle="-", color=leader_color, lw=leader_lw,
                                shrinkA=2.5, shrinkB=1.5),
            )
            annotations.append(ann)
            all_text_artists.append(ann)

    # The greedy stack above already honors min_gap, so no nudge pass is
    # needed for rail labels — an iterative post-pass tends to drift labels
    # toward the boundaries when neighbors stay within sub-pixel of touching.
    return annotations
