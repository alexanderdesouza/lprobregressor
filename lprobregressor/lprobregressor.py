class LProbRegressor(object):
    """
    Class definition for a linear probabilistic regressor, including constructor, fit and predict methods.
    """
    def __init__(self):
        """
        This is the class initialization method. The input X and output y are assumed to be scaled prior to input.
        Then there is a 'locFitter' and a 'scaFitter', these two models are used to linearly fit the "mean" and
        "standard deviation"...
        """
        # New comment: The input is assumed to be scaled in a context appropriate manner already.
        # self.xscaler = sklearn.preprocessing.RobustScaler(quantile_range=(25, 75))  # these can be adjustable; how is
                                                                                      # this handled in sklearn's model
                                                                                      # classes?
        # self.yscaler = sklearn.preprocessing.RobustScaler(quantile_range=(25, 75))

        # self.locFitter = fitter.LinearRegressor()  # these are custom methods, see the "fitter" module included
                                                     # why custom methods?
        # self.scaFitter = fitter.LinearRegressor()

        self.slr_mean = SimpleLinearRegressor()
        self.slr_std = SimpleLinearRegressor()

        self.pymc3model = pm.Model()  # handles the sampling aspect of this model
    
    def fit(self, X, y, tune=1000, draws=3000, target_accept=0.95):
        # Original comment: "Preprocess the input: check, scale, take finite, remove outliers"
        # New comment(s): We want to perform the following validations and return an error if not:
        #                    * are the input data shaped correctly
        #                    * validate that there are no NaNs present
        # X = x if len(x.shape) > 1 else x.reshape((-1, 1))
        # Y = y
        # Remove NaNs
        # idx = np.where(np.all(np.isfinite(X), axis=1) & np.isfinite(Y))[0]
        # X = X[idx]
        # Y = Y[idx]

        # Will have to do something like this...
        # X, y = check_X_y(X, y,
        #                  accept_sparse=['csr', 'csc', 'coo'],
        #                  y_numeric=True, multi_output=True)

        # Original comment: "Scale data"
        # New comment: This is assumed to have been performed ahead of calling this method.
        # X = self.xscaler.fit_transform(X)
        # Y = self.yscaler.fit_transform(Y.reshape((-1,1))).reshape(-1)

        # Original comment: Remove outliers.
        # New comment: This is done to ensure that the probabilistic function can be well defined.
        # xinliers = np.percentile(X, q=[1,99], axis=0)
        # yinliers = np.percentile(Y, q=[1,99])
        # idx = np.where( np.all( (X > xinliers[0]) & (X < xinliers[1]), axis=1 ) & (Y > yinliers[0]) & (Y < yinliers[1]) )[0]
        # X = X[idx]
        # Y = Y[idx]

        # New comment: Removed the following.
        # self.xtrain = X
        # self.ytrain = Y

        # Fit a first guess of the mean and standard deviation around which to perform probabilistic sampling
        self.slr_mean.fit(X, y, esd_alpha=None)
        self.slr_std.fit(X, np.abs(y-self.slr_std.predict(X)), esd_alpha=None)

        # Fit the mean and standard deviation using a MCMC method
        with self.pymc3model:
            U = np.concatenate([X, np.ones((X.shape[0],1))], axis=1)
            locPriors = pm.Uniform("loc", lower=self.locFitter.coeffs - 5, upper=self.locFitter.coeffs + 5, shape=U.shape[1])
            scaPriors = pm.Uniform("sca", lower=self.scaFitter.coeffs - 5, upper=self.scaFitter.coeffs + 5, shape=U.shape[1])
            start = {'locPriors':self.locFitter.coeffs, 'scaPrior':self.scaFitter.coeffs}
            loc = pm.math.dot(U, locPriors)
            sca = pm.math.dot(U, scaPriors)
            HyperbolicSecant("target", mu=loc, s=sca, observed=Y)
            self.trace = pm.sample(draws=draws, tune=tune, start=start, live_plot=False, step=pm.NUTS(target_accept=target_accept))
            self.locCoeffs = self.trace['loc'].mean(axis=0)
            self.scaCoeffs = self.trace['sca'].mean(axis=0)
    
    def predict(self, x, y, quantiles=None):
        # Preprocess the input: check, scale, take finite, remove outliers
        X = x if len(x.shape) > 1 else x.reshape((-1,1))
        Y = y
        # Scale data
        finite = np.all(np.isfinite(X), axis=1)
        X[finite] = self.xscaler.transform(X[finite])
        Y[finite] = self.yscaler.transform(Y[finite].reshape((-1,1))).reshape(-1)
        # Remove outliers and NaNs
        inlier =  np.all( (X > -5) & (X < 5), axis=1 )  
        # Remove invalid predictions (negative scale)
        U = np.concatenate([X, np.ones((X.shape[0],1))], axis=1)
        valid  = np.dot(U, self.scaCoeffs) > 0
        idx = np.where( finite & inlier & valid )[0]
        X = X[idx]
        Y = Y[idx]
        U = U[idx]
        # Predict stuff
        Z = np.dot(U, self.locCoeffs)
        E = np.dot(U, self.scaCoeffs)
        C = scipy.stats.hypsecant(loc=Z, scale=E).cdf(Y)
        z = self.yscaler.inverse_transform(Z.reshape((-1,1))).reshape(-1)
        e = self.yscaler.scale_ * E
        c = C
        if quantiles is not None:
            if not hasattr(quantiles, "__iter__"):
                quantiles = [quantiles]
            Q = [scipy.stats.hypsecant(loc=Z, scale=E).ppf(u/100.) for u in quantiles]
            q = [self.yscaler.inverse_transform(u.reshape((-1,1))).reshape(-1) for u in Q]
            return z, e, c, idx, q
        else:
            return z, e, c, idx

    def __str__(self):
        """
        Stringify this object
        """
        out = "y = "
        out += "%.2f"%(self.locCoeffs[-1])
        for i,c in enumerate(self.locCoeffs[:-1], 1):
            out += " + %.2f x[%d]"%(c,i)
        out += "\nDy = "
        out += "%.2f"%(self.scaCoeffs[-1])
        for i,c in enumerate(self.scaCoeffs[:-1], 1):
            out += " + %.2f x[%d]"%(c,i)
        return out