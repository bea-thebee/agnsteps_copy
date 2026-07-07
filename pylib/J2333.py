import sys
from utilities.ipynb_docgen import show, capture
from wtlike import WtLike, MJD, UTC, config
from pylib.data_setup import set_theme, show_date, show_link

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

set_theme(sys.argv)

class J2333_plots:
    source= 'PKS J2333-2343'
    neighbor='4FGL J2331.0-2147'

    # def __init__(self, interval=30):
    #     time_bins=(0,0,interval)
    #     wtl = WtLike(self.source,  time_bins=time_bins) 
    #     neighbor_bb = WtLike(self.neighbor,time_bins=time_bins ).bb_view()
    #     self.wtl = wtl.reweighted_view(neighbor_bb)
    #     self.bb = self.wtl.bb_view()
 

    def bb_light_curve(self, interval=365.25, p0=0.05):
        time_bins=(0,0,interval)
        wtl = WtLike(self.source,  time_bins=time_bins); 
        if self.neighbor is not None:
            neighbor_bb = WtLike(self.neighbor,time_bins=time_bins ).bb_view()
            self.wtl = wtl.reweighted_view(neighbor_bb)
        else: self.wtl = wtl
        self.bb = self.wtl.bb_view(p0=p0)
        return self.bb

    def plot_bb(self, interval, p0=0.05, **kwargs):
        with capture('Printout') as out:
            bb = self.bb_light_curve(interval, p0=p0)
        fig, ax = plt.subplots(figsize=(8,3))
        ax.axhline(1,  color='0.4', ls='--')
        kw = dict(yscale='log', ylim=(0.2,20),)
        kw.update(kwargs)
        bb.plot(ax=ax, source_name='', **kw)  
        show(out, summary='Print out')
        show(fig)
        show(bb.fluxes['t tw ts flux errors'.split()],index=False, summary='BB values')

    def flux_figure(self):

        show("""## Compare Fermi flux measurements with H-G""")
        tstart = np.array([MJD('2015-01-01'), MJD('2018-01-05')])                 
        tstop =  np.array([MJD('2016-01-01'), MJD('2019-09-30')])
        pflux = np.array([5.20,  6.0])/5.9
        px = 0.5*(tstart+tstop)
        pxerr = (tstop-tstart)/2

        fig,ax=plt.subplots(figsize=(6 ,3))
        def mjd_range(year):
            return MJD(f'{year}-01-01'), MJD(f'{year+1}-01-01')
        ax.errorbar(x=px, xerr=pxerr, y=pflux, fmt=' ', marker='o', 
                    ms=10, color='magenta', label='H-G et al.' )
        ax.set(xticks=np.arange(56000, 60001, 1000))

        fig = self.bb.plot(ax=ax,ylim=(0,2.4), colors=('none','none', 'maroon'), 
                    legend_loc='upper left', source_name='')

        # ax.legend(legend.get_texts()[2:])

        show(fig)

        df =self.bb.fluxes
        mjd = df.t[0] + df.tw[0]/2
        show(df, summary='BB flux table')

        show(fr"""Notes:
        * Fluxes are relative to $5.90\times 10^{{-12}} \mathrm{{\ erg\ cm^{{-2}}\ s^{{-1}}}}$,
        the *pointlike* average for 14 years.
        * The transition is at MJD {mjd}, or {UTC(mjd)[:-6]}, within a few days.
        * The Hernandez-Garcia *et al.* points are quoted with 0.1% errors, 
        not consistent with our 7% uncertainty for the
        full data set. Also, they quote significance values of 5.1 and 6.2 
        for the two intervals, also inconsistent with the 4FGL-DR3 12-year 
        value of 6.5.
        """)

show("""# Analysis of PKS J2333-2343

Motivating paper: [Hernandez-Garcia et al](https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/stad510/7080132?login=false)


The paper claims that there was a significant transition in 2016 or 2017, speculating that there was a rare jet realignment.

The 4FGL source name is 4FGL J2333.9-2343.
""")
show_date()
def mjd_range(year):
    from wtlike import MJD
    return MJD(f'{year}-01-01'), MJD(f'{year+1}-01-01')
show(f"""It quotes an analysis of _Fermi_ data for the periods:
* 2015: MJD {mjd_range(2015)}
* 2018: MJD {mjd_range(2018)}
""")
assert config.Config().get('keep_pixels', False), 'Need to set keep_pixels'
self = J2333_plots()
show(f"""## Full wtlike light-curve""")
self.plot_bb(30, 0.05)

self.flux_figure()