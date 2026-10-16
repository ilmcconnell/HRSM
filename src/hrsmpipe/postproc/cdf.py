"""Empirical CDF and CDF matching, copied verbatim from the notebook (cell 114)."""

import numpy as np
from scipy.stats import rankdata


def empirical_cdf(data):
    ranks = rankdata(data, method="average")  # Rank the data
    return ranks / len(data)  # Normalize by the number of points


def cdf_matching(source, target):
    sorted_target = np.sort(target)  # Target distribution sorted
    sorted_target_ranks = rankdata(sorted_target, method="average")
    sorted_target_quantiles = sorted_target_ranks / len(sorted_target)
    ranks = rankdata(source, method="average")  # Source data ranks
    percentiles = ranks / len(source)  # Percentile positions of source data
    matched_values = np.interp(percentiles, sorted_target_quantiles, sorted_target)  # Interpolation
    return matched_values
