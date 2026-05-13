import numpy as np
import logging

def print_latex_metrics(metrics, ranks=[1, 2, 3, 5, 10], t2m=True, m2t=True, MedR=True):
    vals = [str(x).zfill(2) for x in ranks]
    t2m_keys = [f"t2m/R{i}" for i in vals]
    if MedR:
        t2m_keys += ["t2m/MedR"]
    m2t_keys = [f"m2t/R{i}" for i in vals]
    if MedR:
        m2t_keys += ["m2t/MedR"]

    keys = []
    if t2m:
        keys += t2m_keys
    if m2t:
        keys += m2t_keys
    
    def ff(val_):
        val = str(val_).ljust(5, "0")
        # make decimal fine when only one digit
        if val[1] == ".":
            val = str(val_).ljust(4, "0")
        return val

    str_ = "& " + " & ".join([ff(metrics[key]) for key in keys]) + r" \\"
    dico = {key: ff(metrics[key]) for key in keys}
    print(dico)
    if "t2m/len" in metrics:
        print("Number of samples: {}".format(int(metrics["t2m/len"])))
    else:
        print("Number of samples: {}".format(int(metrics["m2t/len"])))
    print(str_)

    ### Norm part for action recognition

    norm_keys = []
    for key in keys:
        if f"{key}_norm" in metrics:
            norm_keys.append(f"{key}_norm")

    str_ = "& " + " & ".join([ff(metrics[key]) for key in norm_keys]) + r" \\"
    dico = {key: ff(metrics[key]) for key in norm_keys}
    print(dico)
    print(str_)


def all_contrastive_metrics(
    sims, emb=None, threshold=None, rounding=2, return_cols=False, m2t=True, t2m=True
):
    if not t2m and not m2t:
        logging.warning("No metrics asked to be computed")
        return None
    
    text_selfsim = None
    if emb is not None:
        text_selfsim = emb @ emb.T

    if t2m:
        t2m_m, t2m_cols = contrastive_metrics(
            sims, text_selfsim, threshold, return_cols=True, rounding=rounding
        )
    if m2t:
        m2t_m, m2t_cols = contrastive_metrics(
            sims.T, text_selfsim, threshold, return_cols=True, rounding=rounding
        )

    all_m = {}
    if t2m:
        keys = t2m_m.keys()
    else:
        keys = m2t_m.keys()
    for key in keys:
        if t2m:
            all_m[f"t2m/{key}"] = t2m_m[key]
        if m2t:
            all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["t2m/len"] = float(len(sims))
    if m2t:
        all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        if not m2t:
            m2t_cols = None
        return all_m, t2m_cols, m2t_cols
    return all_m


def contrastive_metrics(
    sims,
    text_selfsim=None,
    threshold=None,
    return_cols=False,
    rounding=2,
    break_ties="optimistically",
):  
    n, m = sims.shape
    assert n == m
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    gt_dists = np.diag(dists)[:, None]
    
    if text_selfsim is not None and threshold is not None:
        real_threshold = 2 * threshold - 1
        idx = np.argwhere(text_selfsim >= real_threshold)
        partition = np.unique(idx[:, 0], return_index=True)[1]
        # take as GT the minimum score of similar values
        gt_dists = np.minimum.reduceat(dists[tuple(idx.T)], partition)
        gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    # if there are ties
    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    if return_cols:
        return cols2metrics(cols, num_queries, rounding=rounding), cols
    return cols2metrics(cols, num_queries, rounding=rounding)


def break_ties_average(sorted_dists, gt_dists):
    # fast implementation, based on this code:
    # https://stackoverflow.com/a/49239335
    locs = np.argwhere((sorted_dists - gt_dists) == 0)

    # Find the split indices
    steps = np.diff(locs[:, 0])
    splits = np.nonzero(steps)[0] + 1
    splits = np.insert(splits, 0, 0)

    # Compute the result columns
    summed_cols = np.add.reduceat(locs[:, 1], splits)
    counts = np.diff(np.append(splits, locs.shape[0]))
    avg_cols = summed_cols / counts
    return avg_cols


def break_ties_optimistically(sorted_dists, gt_dists):
    rows, cols = np.where((sorted_dists - gt_dists) == 0)
    _, idx = np.unique(rows, return_index=True)
    cols = cols[idx]
    return cols


def cols2metrics(cols, num_queries=None, rounding=2):
    metrics = {}
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]

    if num_queries is None:
        num_queries = len(cols)
    for val in vals:
        metrics[f"R{val}"] = 100 * float(np.sum(cols < int(val))) / num_queries

    metrics["MedR"] = float(np.median(cols) + 1)

    if rounding is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], rounding)
    return metrics


def contrastive_metrics_m2t_action_retrieval(
    sims,
    motion_cat_idx,
    return_cols=False,
    rounding=2,
    break_ties="averaging", 
    norm_metrics=True
):  
    n, m = sims.shape
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    gt_dists = dists[range(n), motion_cat_idx]
    gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    if norm_metrics:
        motion_cat_idx = np.array(motion_cat_idx)
        cat_metrics = []
        for i in range(np.max(motion_cat_idx) + 1):
            cols_cat = cols[motion_cat_idx==i]
            cat_metrics.append(cols2metrics(cols_cat, rounding=rounding))
        
        print("len(cat_metrics) : ", len(cat_metrics))

        metrics_norm = {}
        keys = cat_metrics[0].keys()
        for k in keys:
            metrics_norm[f"{k}_norm"] = round(np.mean([elt[k] for elt in cat_metrics]), 2)

    metrics = cols2metrics(cols, num_queries, rounding=rounding)

    if norm_metrics:
        metrics.update(metrics_norm)

    if return_cols:
        return metrics, cols
    return metrics


def all_contrastive_metrics_action_retrieval(
    sims, motion_cat_idx, rounding=2, return_cols=False, norm_metrics=True
):
        
    m2t_m, m2t_cols = contrastive_metrics_m2t_action_retrieval(
        sims.T, motion_cat_idx, return_cols=True, rounding=rounding, norm_metrics=norm_metrics
    )

    all_m = {}
    keys = m2t_m.keys()
    for key in keys:
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, m2t_cols
    return all_m


def contrastive_metrics_m2t_action_retrieval_multi_labels(
    sims,
    motion_cat_idx,
    return_cols=False,
    rounding=2,
    break_ties="averaging",
    norm_metrics=True
):
    n, m = sims.shape
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)

    motion_cat_idx = [cat_idx[np.argmin([dists[i, elt] for elt in cat_idx])] for i, cat_idx in enumerate(motion_cat_idx)]
    gt_dists = dists[range(n), motion_cat_idx]
    gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    if norm_metrics:
        motion_cat_idx = np.array(motion_cat_idx)
        cat_metrics = []
        for i in range(np.max(motion_cat_idx) + 1):
            cols_cat = cols[motion_cat_idx==i]
            if len(cols_cat) > 0:
                cat_metrics.append(cols2metrics(cols_cat, rounding=rounding))

        print("len(cat_metrics) : ", len(cat_metrics))

        metrics_norm = {}
        keys = cat_metrics[0].keys()
        for k in keys:
            metrics_norm[f"{k}_norm"] = round(float(np.mean([elt[k] for elt in cat_metrics])), 2)

    metrics = cols2metrics(cols, num_queries, rounding=rounding)

    if norm_metrics:
        metrics.update(metrics_norm)

    if return_cols:
        return metrics, cols
    return metrics


def all_contrastive_metrics_action_retrieval_multi_labels(
    sims, motion_cat_idx, rounding=2, return_cols=False, norm_metrics=True
):

    m2t_m, m2t_cols = contrastive_metrics_m2t_action_retrieval_multi_labels(
        sims.T, motion_cat_idx, return_cols=True, rounding=rounding, norm_metrics=norm_metrics
    )

    all_m = {}
    keys = m2t_m.keys()
    for key in keys:
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, m2t_cols
    return all_m

