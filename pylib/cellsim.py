from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from wtlike.loglike import LogLike, PoissonRep


def p_view(self, k,m): #new_edges):
    """ New view with arbitrary partition
    (here group beginning and end, weeks k and m
    """
    from wtlike.cell_data import partition_cells
    from wtlike.lightcurve import fit_cells
    cells = self.cells
    edges = np.append(cells.t-cells.tw/2, cells.iloc[-1].t+cells.iloc[-1].tw/2)
    new_edges = np.append(
                np.append(edges[0], edges[k:m+1]),   edges[-1]);

    # basic copy
    r = self.view()

    # make new set of cells and add poisson fits
    r.cells = partition_cells(self.config, self.cells, new_edges)
    r.fits = fit_cells(self.config, r.cells, )
    return r

class CellSim: 

    def __init__(self, incell, rgen = None):
        """incell: a cell object for calibration
        """
        w = incell.w
        self.S, self.B = incell.S, incell.B
        self.N = len(w)
        self.rgen = rgen

        self.poiss_fit = PoissonRep(LogLike(incell))
        self.flux = self.poiss_fit.flux
                
        # the CDF of the weights
        cdf = stats.ecdf(w).cdf

        delta =0.01  # how far past sample edge to plot
        q = cdf.quantiles
        self.q = np.array( list(q) + [q[-1] + delta])
        self.yq = cdf.evaluate(q)

    def __repr__(self):
        return f"CellSim: S={self.S:.0f}, B={self.B:.0f}, N={self.N}, poiss={self.poiss_fit}" 
    
    def __call__(self, cell):
        """ Return a simulated version of the cell with a new set of weights.
        Flux corresponds to the input model
        """
        simcell = cell.copy() 
        # expected number, using flux from the setup cell. Use with Poisson
        mu = self.flux*cell.S+cell.B 
        simcell.w = self._weight_gen(
            stats.uniform.rvs(
                size=stats.poisson.rvs(mu, random_state=self.rgen),
                random_state=self.rgen))
        simcell.n = len(simcell.w)
        return simcell    
        
    def _weight_gen(self, rgen):
        """Return sampled weight distribution 
        """
        return  self.q[np.searchsorted(self.yq, rgen)]
        
    def alpha_estimate(self, w):
        """ Return alpha and sigma_alpha derived from the weights $w$ and with the cell's
        signal and background estimates
        using the weak-signal approximation to the likelihood
        """  
        W, U  = np.sum(w), np.sum(w*w)
        return (W-self.S)/U, np.sqrt(1/U)
        
    @classmethod
    def check_weights(cls, cell, N=None):
        """ Test weight generation
        """
        sf = cls(cell)
        if N is  None: N = len(cell.w) 
        wunif = sf._weight_gen(np.linspace(0,1,N))
        def stat(w):
            alf = sf.alpha_estimate(w)
            return  f"{len(w):,}|{np.mean(w):.2e}| {np.mean(w**2):.2e}"\
                f"|{alf[0]:.3f} | {alf[1]:.3f}"
        return rf"""           
            | Sample  | N | $<w>$ | $<w^2>$ | $\alpha$ | $\sigma_\alpha$
            | --------|--:|------:|--------:|---------:|-----------
            |data     | {stat(cell.w)}
            |uniform  | {stat(wunif)}
            |random   | {stat(sf._weight_gen(stats.uniform.rvs( size=N)))}

            """
    @classmethod
    def plot_cdf(cls, cell, ax=None, **kwargs):
        """ Plot the CDF of the weights
        """
        sf = cls(cell)
        fig, ax = plt.subplots(figsize=(5, 4)) if ax is None else (ax.figure, ax)
        ax.plot(sf.q[:-1], sf.yq, label='CDF')
        ax.set(xlabel='Weight', ylabel='CDF', title='CDF of Weights',
               xscale='log',xlim=(None,1), ylim=(0,1), **kwargs)
        ax.legend()
        return fig    

        
