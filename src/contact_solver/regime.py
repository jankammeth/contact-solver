"""Active-set regime classification for the two-robot problem (G4 campaign).

Classifies the active set of a solved two-robot instance into one of the
candidate regimes

    "0"     unconstrained (no contact)
    "1"     single isolated contact
    "2"     two isolated contacts
    "arc"   one boundary arc
    other   e.g. "3pts", "arc+1", "2arcs", ... -> counterexample flags

using three independent detectors:

    1. active_width_classifier   (SCP trajectory, structure-agnostic):
       near an isolated tangency d(t) - R ~ c (t - t*)^2, so the active
       width  w(eps) = |{t : d <= R + eps}|  scales like sqrt(eps); on a
       boundary arc w(eps) plateaus at the arc length L > 0. The fitted
       log-log slope per cluster separates the two.

    2. dual_mass_classifier      (SCP collision duals): the dual at grid
       node t_k estimates mu([t_k - h/2, t_k + h/2]). An atom
       concentrates its mass at ~1 node (per-node dual ~ mu*,
       independent of h); an arc spreads it (per-node dual ~ Lambda h).
       Discriminator: max-node share of cluster mass.

    3. sci_refinement_classifier (SCI): re-solve at decreasing detection
       tolerance delta; a contact cluster whose count grows under
       refinement is an arc; a stable count is isolated contacts.

Each detector returns per-cluster labels plus the diagnostics needed to
verify the decision in a plot (fitted exponents, shares, counts).
"""

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Cluster utilities
# ---------------------------------------------------------------------------


def _connected_clusters(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end)] index ranges (inclusive) of True runs."""
    clusters = []
    in_run = False
    start = 0
    for i, m in enumerate(mask):
        if m and not in_run:
            in_run, start = True, i
        elif not m and in_run:
            clusters.append((start, i - 1))
            in_run = False
    if in_run:
        clusters.append((start, len(mask) - 1))
    return clusters


def winding_class(rel_positions: np.ndarray) -> tuple[int, float]:
    """Homotopy class of a relative trajectory around the origin.

    Args:
        rel_positions: (K, 2) samples of r(t), assumed ||r|| > 0.

    Returns:
        (w, delta_theta): integer class index
        w = round((delta_theta - principal(theta_T - theta_0)) / 2 pi)
        and the lifted total angle delta_theta.
    """
    phi = np.unwrap(np.arctan2(rel_positions[:, 1], rel_positions[:, 0]))
    delta = float(phi[-1] - phi[0])
    principal = np.arctan2(
        np.sin(phi[-1] - phi[0]), np.cos(phi[-1] - phi[0])
    )
    w = int(np.round((delta - principal) / (2 * np.pi)))
    return w, delta


# ---------------------------------------------------------------------------
# 1) Active-width scaling classifier (SCP trajectory)
# ---------------------------------------------------------------------------


@dataclass
class WidthCluster:
    t_center: float
    d_min: float
    t_lo: float = 0.0
    t_hi: float = 0.0
    eps: np.ndarray = field(default_factory=lambda: np.array([]))
    widths: np.ndarray = field(default_factory=lambda: np.array([]))
    slope: float = np.nan
    label: str = "ambiguous"  # "point" | "arc" | "ambiguous"


def active_width_classifier(
    times: np.ndarray,
    dist: np.ndarray,
    R: float,
    eps_rel: np.ndarray | None = None,
    slope_point: float = 0.35,
    slope_arc: float = 0.15,
    contact_tol: float = 2e-3,
) -> list[WidthCluster]:
    """Per-cluster point/arc labels from the width-vs-eps scaling.

    Clusters are seeded from the connected active components at the
    smallest eps; the width at each eps is the length of the component
    containing the seed.

    Non-event rule: a cluster whose d_min exceeds R + contact_tol never
    touches the constraint (transition-shell artifact near the 0/1
    boundary: the trajectory dips into the coarse eps shell without
    contact); such clusters are dropped.
    """
    if eps_rel is None:
        # Floor >= ~3x the solver slack (detection tol / osqp eps, 1e-3
        # in configs/phase_diagram.yaml = small.yaml values): below
        # that, widths measure tolerance noise, not geometry.
        eps_rel = np.logspace(-2.5, -1.0, 8)
    eps_list = np.sort(np.asarray(eps_rel)) * R

    seed_clusters = _connected_clusters(dist <= R + eps_list[0])
    if not seed_clusters:
        # Nothing active even at the largest eps? Check that too.
        if not _connected_clusters(dist <= R + eps_list[-1]):
            return []
        # Active only at coarse eps: seed there instead (shallow event).
        seed_clusters = _connected_clusters(dist <= R + eps_list[-1])

    out = []
    for s, e in seed_clusters:
        i_min = s + int(np.argmin(dist[s : e + 1]))
        if dist[i_min] > R + contact_tol:
            continue  # non-event: never touches the constraint
        cl = WidthCluster(t_center=float(times[i_min]), d_min=float(dist[i_min]))
        eps_ok, widths = [], []
        for eps in eps_list:
            comps = _connected_clusters(dist <= R + eps)
            comp = next(((a, b) for a, b in comps if a <= i_min <= b), None)
            if comp is None:
                continue
            a, b = comp
            eps_ok.append(eps)
            widths.append(float(times[b] - times[a]))
            # interval of the near-active span: the component at the
            # LARGEST eps containing the seed (used by fuse_regime to
            # assign dual sub-clusters to this physical event; a
            # center-radius window centered at d_min misses far-end
            # junctions on asymmetric arcs)
            cl.t_lo, cl.t_hi = float(times[a]), float(times[b])
        cl.eps = np.array(eps_ok)
        cl.widths = np.array(widths)
        pos = cl.widths > 0
        if pos.sum() >= 3:
            cl.slope = float(
                np.polyfit(np.log(cl.eps[pos]), np.log(cl.widths[pos]), 1)[0]
            )
            if cl.slope >= slope_point:
                cl.label = "point"
            elif cl.slope <= slope_arc:
                cl.label = "arc"
        out.append(cl)

    # Merge seeds that ended up in the same component at the largest eps
    # (two seeds of one physical cluster): keep the deeper one.
    merged: list[WidthCluster] = []
    comps_max = _connected_clusters(dist <= R + eps_list[-1])
    used = set()
    for a, b in comps_max:
        members = [
            c for c in out if times[a] <= c.t_center <= times[b]
        ]
        if not members:
            continue
        # Members with distinct labels stay separate (e.g., two points in
        # one coarse component); identical-label members merge to deepest.
        by_label: dict[str, list[WidthCluster]] = {}
        for c in members:
            by_label.setdefault(c.label, []).append(c)
        for label, cs in by_label.items():
            if label == "point":
                merged.extend(cs)  # distinct tangencies stay distinct
            else:
                merged.append(min(cs, key=lambda c: c.d_min))
        used.update(id(c) for c in members)
    merged.extend(c for c in out if id(c) not in used)
    merged.sort(key=lambda c: c.t_center)
    return merged


# ---------------------------------------------------------------------------
# 2) Dual-mass classifier (SCP collision duals)
# ---------------------------------------------------------------------------


@dataclass
class DualCluster:
    t_center: float
    n_nodes: int
    mass: float
    max_share: float
    label: str  # "atom" | "arc" | "ambiguous"


def dual_mass_classifier(
    times: np.ndarray,
    duals: np.ndarray,
    mass_floor_frac: float = 1e-4,
    share_atom: float = 0.5,
    share_arc: float = 0.25,
    min_arc_nodes: int = 5,
) -> list[DualCluster]:
    """Per-cluster atom/arc labels from the dual-mass concentration.

    Args:
        duals: (K,) collision duals for one pair (>= 0 up to solver
            noise; negative noise is clipped).
    """
    lam = np.clip(np.asarray(duals, dtype=float), 0.0, None)
    total = lam.sum()
    # Absolute signal gate: if the whole profile is numerical dust
    # (e.g. no truly active constraints, or duals lost), report no
    # information instead of classifying noise.
    if total <= 0 or lam.max() < 1e-8:
        return []
    active = lam > mass_floor_frac * lam.max()
    out = []
    for s, e in _connected_clusters(active):
        seg = lam[s : e + 1]
        mass = float(seg.sum())
        if mass < 1e-3 * total:
            continue  # numerical dust
        share = float(seg.max() / mass)
        n = int(e - s + 1)
        i_max = s + int(np.argmax(seg))
        if share >= share_atom:
            label = "atom"
        elif share <= share_arc and n >= min_arc_nodes:
            label = "arc"
        else:
            label = "ambiguous"
        out.append(
            DualCluster(
                t_center=float(times[i_max]),
                n_nodes=n,
                mass=mass,
                max_share=share,
                label=label,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 3) SCI delta-refinement classifier
# ---------------------------------------------------------------------------


@dataclass
class SciRefinement:
    deltas: list[float]
    counts: list[int]
    contact_times: list[list[float]]
    label: str  # "isolated" | "arc" | "mixed" | "none"
    n_isolated: int


def sci_refinement_classifier(
    run_sci,  # callable: delta -> (contact_times: list[float], converged: bool)
    deltas: tuple[float, ...] = (1e-3, 1e-4, 1e-5),
    cluster_gap: float = 0.05,
) -> SciRefinement:
    """Label from contact-count growth under detection-tol refinement.

    Contacts at the finest delta are grouped into clusters (gap >
    cluster_gap * T implied by caller's time units). A cluster is an
    arc if its member count grew from the coarsest to the finest delta.
    """
    deltas = sorted(deltas, reverse=True)  # coarse -> fine
    counts, all_times = [], []
    for d in deltas:
        tc, _ok = run_sci(d)
        counts.append(len(tc))
        all_times.append(sorted(tc))

    fine = all_times[-1]
    if not fine:
        return SciRefinement(deltas, counts, all_times, "none", 0)

    # Cluster fine-delta contacts by gap.
    clusters: list[list[float]] = [[fine[0]]]
    for t in fine[1:]:
        if t - clusters[-1][-1] <= cluster_gap:
            clusters[-1].append(t)
        else:
            clusters.append([t])

    coarse = all_times[0]
    labels = []
    for cl in clusters:
        lo, hi = cl[0] - cluster_gap, cl[-1] + cluster_gap
        n_coarse = sum(lo <= t <= hi for t in coarse)
        labels.append("arc" if len(cl) > max(n_coarse, 1) else "isolated")

    n_iso = labels.count("isolated")
    if all(lab == "isolated" for lab in labels):
        label = "isolated"
    elif all(lab == "arc" for lab in labels):
        label = "arc"
    else:
        label = "mixed"
    return SciRefinement(deltas, counts, all_times, label, n_iso)


# ---------------------------------------------------------------------------
# Label fusion
# ---------------------------------------------------------------------------


def fuse_regime(width_clusters: list[WidthCluster],
                dual_clusters: list[DualCluster],
                times_span: float = 1.0) -> str:
    """Fuse the two SCP detectors into a regime string.

    Physical events are defined by the WIDTH clusters (contiguous
    near-active spans of d(t)); dual clusters are assigned to the width
    cluster whose span contains them. A single physical arc produces
    the dual signature atom + density + atom (junction atoms pi1, pi2
    bracketing the interior density Lambda dt, cf. eq. arc-measure), so
    several dual sub-clusters inside one width span are ONE event, not
    three. Event label: width label where decisive; dual labels used
    when the width label is ambiguous (any "arc" sub-label, or >= 3
    sub-clusters -> arc; single atom -> point).

    Returns "0", "1", "2", "arc", or a descriptive string for anything
    else (-> counterexample flag).
    """
    if not width_clusters and not dual_clusters:
        return "0"

    events = []
    used_d = set()
    for wc in width_clusters:
        pad = 0.02 * times_span
        members = [
            i for i, dc in enumerate(dual_clusters)
            if i not in used_d
            and wc.t_lo - pad <= dc.t_center <= wc.t_hi + pad
        ]
        used_d.update(members)
        sub = [dual_clusters[i].label for i in members]

        if wc.label == "point":
            lab = "point"
        elif wc.label == "arc":
            lab = "arc"
        else:  # width ambiguous -> decide from the dual signature
            if any(s == "arc" for s in sub) or len(sub) >= 3:
                lab = "arc"
            elif len(sub) == 1 and sub[0] == "atom":
                lab = "point"
            else:
                lab = "ambiguous"
        events.append(lab)

    # dual clusters not covered by any width span (shouldn't happen for
    # a consistent solve; keep them visible rather than dropping them)
    for i, dc in enumerate(dual_clusters):
        if i not in used_d:
            events.append("point" if dc.label == "atom" else dc.label)

    if any(e == "ambiguous" for e in events):
        return "ambiguous:" + "+".join(events)
    n_pts = events.count("point")
    n_arc = events.count("arc")
    if n_arc == 0:
        return {0: "0", 1: "1", 2: "2"}.get(n_pts, f"{n_pts}pts")
    if n_arc == 1 and n_pts == 0:
        return "arc"
    return f"{n_arc}arcs+{n_pts}pts"


COUNTEREXAMPLE_REGIMES = ("3pts", "4pts", "2arcs+0pts", "1arcs+1pts")


def is_counterexample(regime: str, allow_two: bool = False) -> bool:
    """Zero-velocity slice (allow_two=False): the expected menu is
    {0, 1, arc} (a single generic dip of the distance profile) and
    "2" is flagged. Velocity slices (allow_two=True): two-contact
    minimizers are expected, so "2" is a legitimate regime."""
    allowed = ("0", "1", "2", "arc") if allow_two else ("0", "1", "arc")
    if regime in allowed:
        return False
    if regime.startswith("ambiguous"):
        return False  # ambiguous -> verification queue, not counterexample
    return True
