import scipy.linalg
import ESD
import numpy as np

class SimpleLinearRegressor(object):
    """

    """

    def fit(self, X, y, esd_alpha=None):
        """
        Fit a linear model to the input X based on the output y using ordinary least squares.

        :param: X: input variables (n samples of m features)
        :param: y: model target (n samples)
        :param: esd_alpha: degree of outlier removal (0 < esd_alpha < 1)
        :return: array of coefficients of x variables
                 intercept
                 index of inliers
        """
        M = np.empty((X.shape[1] + 1, X.shape[1] + 1), dtype=float)
        M[:X.shape[1], :X.shape[1]] = np.dot(X.T, X)
        M[:X.shape[1], X.shape[1]]  = M[X.shape[1], :X.shape[1]] = X.sum(axis=0)
        M[X.shape[1], X.shape[1]]   = X.shape[0]
        #M = (M + M.T)/2.
        b = np.empty(X.shape[1] + 1, dtype=float)
        b[:X.shape[1]] = np.dot(y, X)
        b[X.shape[1]]  = y.sum()

        coeffs = scipy.linalg.solve(M, b, sym_pos=True)
        cvrmtx = scipy.linalg.inv(M)
        chi2   = np.dot(y.T,y) - np.dot(b, np.dot(cvrmtx, b))
        outl   = np.array([], dtype=int)

        # Original comment: Do outlier remove using extreme studentized deviate test.
        # New comment: This forces the linear regression to fit the data assuming it is Guassian distributed, which
        #              may not be the case. Rather than forcing such, this method should allow abnormal fits.
        # if esd_alpha is not None and esd_alpha > 0 and esd_alpha < 1:
        #     z    = np.dot(X, coeffs[:-1]) + coeffs[-1]
        #     outl = ESD.G_ESD_test(z-y, alpha=esd_alpha, max_outliers=len(z)//2, min_cluster_size=len(z)//2, sides="two sided")
        #     inl  = np.setdiff1d(range(len(z)), outl, assume_unique=True)
        #     X = X[inl]
        #     y = y[inl]
        #     # re-fit without outliers
        #     coeffs, cvrmtx, chi2, _ = self.fit_(x=X, y=y, esd_alpha=None)

        return coeffs, cvrmtx, chi2, outl

    def fit(self, x, y, esd_alpha=None):
        """
        Fit linear model using ordinary least squares

        Args:
            x [array]     : variables
            y [array]     : target
            esd_alpha [0<esd_alpha<1] : level of outlier removal
        Returns:
            this instance
        """
        x = np.array(x, dtype=float)
        if len(x.shape) == 1:
            x = x.reshape( (x.shape[0], 1) )
        y = np.array(y, dtype=float)
        coeffs, cvrmtx, chi2, outl = self.fit_(x=x, y=y, esd_alpha=esd_alpha)
        self.coeffs = coeffs
        self.cvrmtx = cvrmtx
        self.chi2   = chi2
        self.outl   = outl
        self.ndtpts = x.shape[0] - len(outl)
        self.nprmts = x.shape[1] + 1
        return self

    def predict(self, x):
        if not hasattr(self, "coeffs"):
            raise RuntimeError("Fit before predict")
        if not hasattr(x, "__iter__"):
            return np.dot([x], self.coeffs[:-1]) + self.coeffs[-1]
        x = np.array(x, dtype=float)
        if len(np.shape(x)) == 1:
            x = x.reshape( (x.shape[0], 1) )
        return np.dot(x, self.coeffs[:-1]) + self.coeffs[-1]

    def __str__(self):
        if not hasattr(self, "coeffs"):
            return "No data fit yet with this instance"
        else:
            out = []
            out.append("%d data points fitted after ignoring %d outliers\n"%(self.ndtpts, len(self.outl)))
            out.append("%.2f chi squared\n"%self.chi2)
            for i, c in enumerate(self.coeffs[:-1]):
                out.append("Variable %d\tCoefficient %.2f\n"%(i,c))
            out.append("Intercept\tValue %.2f"%self.coeffs[-1])
        return "".join(out)
