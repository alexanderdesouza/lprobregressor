def check_X_y(X, y,
              accept_sparse=False, allow_nd=False,
              copy=False,
              dtype="numeric",
              ensure_2d=True, ensure_min_features=1, ensure_min_samples=1, estimator=None,
              force_all_finite=True,
              order=None,
              multi_output=False,
              warn_on_dtype=False,
              y_numeric=False):
    """
    Input validation method for the estimator. This method checks input X and output y for consistent length, enforces
    that X be 2d and y 1d. Standard input checks are only applied to the output y, such as checking that y does not have
    np.nan or np.inf targets. If the dtype of X is an object, attempt to convert to float, raising on failure.
    """

    X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
                    ensure_2d, allow_nd, ensure_min_samples,
                    ensure_min_features, warn_on_dtype, estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y
