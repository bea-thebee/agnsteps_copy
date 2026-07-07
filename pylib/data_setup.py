from pathlib import Path
from collections import OrderedDict
import pickle
from astropy.coordinates import SkyCoord

from wtlike import WtLike
from utilities.ipynb_docgen import capture_hide, show, show_fig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def show_date():
    import datetime
    date=str(datetime.datetime.now())[:16]
    show(f"""<h5 style="text-align:right; margin-right:15px"> {date}</h5>""")
def show_link(name):
    show(f'<a href="#{name}">{name}</a>')

def set_theme(argv):
    plt.rcParams['figure.figsize']=[5,3]
    plt.rcParams['figure.dpi'] = 72
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    # sns.set_theme('notebook' if 'talk' not in argv else 'talk', font_scale=1.25) 
    
    sns.set_theme( 'talk', font_scale=1.) 
    if 'paper' in argv: 
        # sns.set_theme('paper')
        sns.set_style('ticks')
    if 'dark' in argv:
        sns.set_style('darkgrid') ##?
        plt.style.use('dark_background')
        plt.rcParams['grid.color']='0.5'
        # plt.rcParams['figure.facecolor']='k'
        dark_mode=True
    else:
        dark_mode=False
        sns.set_style('ticks' if 'paper' in argv else 'whitegrid')
        plt.rcParams['figure.facecolor']='white'
    return dark_mode

# Test statistic for variability 
def TSvar(fitvals):
    """ fitvals: list of log-likelihood functions, one for each cell
    Returns the test statistic for the variability of the light curve.
    """
    return 0 if len(fitvals)<2 else -2 * np.sum([f(1) for f in fitvals])

class VarDB(OrderedDict):
    """Manage a variability data base for Fermi sources

    Original file location on the SLAC s3df system:
    /sdf/home/b/burnett/work/bb_light_curves/files

    Note that the 2022 version as internal DataFrame objects, and cannot be read with pandas 2.
    So the _v1 was rewritten with each light_curve as a dict
    Now: _v3 uses all data, has poisson fits (as truples)
    """
    info_file='files/source_info_v3.pkl'
   
    def __init__(self, info_file=None, reload=False):
        info_file = Path(info_file if info_file is not None else self.info_file)
        if info_file.is_file() and not reload:
            with open(info_file, 'rb') as inp:
                sd = pickle.load(inp)
            print(f'Loaded wtlike-generated variability info for {len(sd)} sources from file `{info_file}`') 
        else:
            raise  FileNotFoundError(f'Did not find {info_file}') 
        # else:
        #     sd = gather(reload=reload)#True)
        #     with open(info_file, 'wb') as out:
        #         pickle.dump(sd, out)
        
        # add the number of light curve intervals and the size of the nearby table
        for key,value in sd.items():
            lc = value['light_curve']  
            near = value['nearby']  
            sd[key]['nbb'] =  len(lc['tw']) if lc is not None else 0
            sd[key]['near'] = len(near) if near is not None else 0
        self.update(sd)
        
    def __repr__(self):
        return f'Collection of variability info for {len(self)} sources'
    
    def load_cats(self, fgl_version='dr4'):

        from utilities.catalogs import Fermi4FGL, UWcat
        print(f"""Load uw1410 and 4FGL-DR4 info for the {len(self)} sources """)

        self.uwcat = uwcat = UWcat('uw1410') #.query('ts>25 & locqual<10')
        uw_coord = SkyCoord(uwcat.ra, uwcat.dec, unit='deg', frame='fk5').galactic
        uwcat.loc[:,'glat'] = uw_coord.b.deg
        uwcat.loc[:,'glon'] = uw_coord.l.deg
        uwcat.loc[:,'nickname'] = uwcat.index
        uwcat.loc[:,'r95'] = 2.45 * np.sqrt(uwcat.a*uwcat.b)*uwcat.systematic
        uwcat.index = uwcat.jname
        index = list(self.keys())

        cols = 'ts r95 locqual specfunc eflux100 e0 glat glon'.split()
        self.df = df = uwcat.loc[index, cols].copy()
        self.df_coord =  SkyCoord(df.glon, df.glat, unit='deg', frame='galactic') 

        nbb = pd.Series(dict(  [ (k,int(v['nbb']))  for k,v in self.items() ] ))
        df.loc[:,'nbb'] = nbb
        # near = pd.Series(dict( [ (k,int(v['near']))  for k,v in self.items() ] ))
        
        try:
            var = pd.Series(dict(  [ (k,int(v['variability']))  for k,v in self.items() ] ))
        except KeyError:
            var = np.nan
        df.loc[:,'bbvar'] = var 
        fglall = Fermi4FGL(fgl_version)
        fgl = fglall[fglall.r95>0].copy() # removes extended
        self.fgl_coord = SkyCoord(fgl.ra, fgl.dec, unit='deg', frame='fk5')
        self.fgl = fgl
        return self
       
    def matchup(self, debug=False, save_to=None):
        if not hasattr(self, 'fgl'): self.load_cats()
        
        fgl, fgl_coord = self.fgl, self.fgl_coord
        uw,  uw_coord =   self.df, self.df_coord
    
        def shower(arg):
            if debug: show(arg)
        
        id_uw, _delta, _ = fgl_coord.match_to_catalog_sky(uw_coord)
        delta = _delta.deg
        shower(f"""* Matched {len(id_uw)} DR4 with the {len(uw)}  UW, of which {len(set(id_uw))} are unique. """)

        match = pd.DataFrame([id_uw, delta], index='id_uw delta'.split()).T
        shower(f'* Created match df w/ {len(match)} entries')
        match.index.name='id_fgl'
        shower(match.head())

        match['dup'] = match.id_uw.duplicated(keep=False)
        mdup = match.query('dup==True')
        # 
        best_dup = pd.DataFrame(
            [g.sort_values('delta').iloc[0] for i,g in mdup.groupby('id_uw')])
        best_dup.index.name = 'id_fgl'
        shower(f"""* picked {len(best_dup)} closest of {len(mdup)} duplicates""")
        # best.hist('delta', bins=50);
        shower(best_dup.head())

        nodup = match[match.dup==False]
        shower(f'combine {len(nodup)} nondup, with {len(best_dup)} best dups')
        dfa = pd.concat([nodup, best_dup])
        shower(dfa.head())
        shower(dfa.describe())
        delta =dfa.delta.clip(0,0.5)
        hkw = dict(bins=np.linspace(0,0.5,26), histtype='step', log=True,lw=2)
        if debug:
            plt.hist(delta[dfa.dup==True],hatch='////', label='had dup', **hkw);
            plt.hist(delta[dfa.dup==False],  **hkw);plt.legend()
            show(plt.gcf())


        shower(f"""## Generate table of uw sources with DR4 counterparts
        Add ...""")
        # dfa['specfunc'] = 
        dft = dfa.copy()
        for col in 'ts r95 glat glon eflux100 specfunc nbb bbvar'.split():
            dft[col] = uw.iloc[dft.id_uw][col].values
        # from 4FGL: only class1 and variability 
        for col in 'class1 variability'.split():
            dft[col] = fgl.iloc[dft.index][col].values

        # derive spectral info from the function
        dft['pindex'] =   dft.specfunc.apply(lambda sf: sf.pars[1]).values.clip(1,3)
        dft['curvature'] =dft.specfunc.apply(lambda sf: sf.curvature())
        dft['e0'] =       dft.specfunc.apply(lambda sf: sf.e0)

        dft['sin_b'] = np.sin(np.radians(dft.glat))
        dft.index = fgl.iloc[dft.index].index
        dft['uw_name'] = uw.iloc[dft.id_uw].index
        
        def categorizer(dfx):
            class1 = dfx.class1.apply( lambda s:s.lower() )
            dfx.loc[:,'association'] = 'other'
            categories = dict([
                ('bll', class1=='bll'),
                ('fsrq',class1=='fsrq'),
                ('bcu', np.isin(class1, ['bcu', 'rdg'])),
                ('unid',class1==''),
                ('psr', np.isin(class1, ['psr', 'msp'])),
            ])

            for k,v in categories.items():
                shower(f'   {k:5} {sum(v):6}')
                dfx.loc[v,'association'] = k
                dfx.loc[v,'category'] = k

        categorizer(dft)

        self.dfx = dft.drop(columns='dup specfunc id_uw'.split())  
        if save_to is not None:
            self.dfx.to_csv(save_to)
        return self

    def add_fft_info(self, ):
        def fmax(v):
            vdf = pd.DataFrame((v['fft_peaks']))
            if len(vdf)==0 or len(vdf)>4:
                return None
            return vdf.iloc[np.argmax(vdf.p1)]
        self.peak_df = pd.DataFrame.from_dict(
            dict(
                [(k, fmax(v)) for k, v  in self.items() if fmax(v) is not None]
            ), orient='index',
        )
        a, b  = set(self.peak_df.index), set(self.dfx.uw_name)
        show(f'* Found {len(self.peak_df)} FFT summaries with 1-4 peaks ({len(a-b)} not in 4FGL.)')
        return self.peak_df


    def light_curve(self, name, ax=None, **kwargs):

        """Draw a light curve plot for the source `name`

        """
        if name not in self: 
            show(f'No light curve for {name}')
            return None
        if self[name]['light_curve'] is None:
            show(f'No light curve for {name}')
            return None
   
        lc = self[name]['light_curve']
        if type(lc)==dict:
            lc = pd.DataFrame.from_dict(lc, orient='index')
        if len(lc)<2:
            y = lc.flux.values[:,0]
        x, xerr = lc.t.values, lc.tw.values/2
        yerr = np.abs(np.array( list(map(lambda x: np.array([x[1],x[0]]), 
                                        lc.errors.values)))).T
        fig, ax = plt.subplots(figsize=(6,2)) if ax is None else (ax.figure, ax)

        ax.axhline(1, color='0.5', ls='--')
        kw = dict( ylim=(0.05,20), yscale='log') #,yticks=[0.1,1,10], yticklabels='0.1 1 10'.split())
        kw.update(kwargs)
        ax.set(**kw)
        ylim= kw.get('ylim', None) 
        if ylim is not None:
            y = y.clip(*ylim)
        ax.errorbar(x,y.clip(*ylim), yerr=yerr,xerr=xerr, 
                    fmt='.', color='maroon', lw=2);
        ax.text(0.95, 0.95, str(name), transform=ax.transAxes, 
                fontsize=10, va='top',ha='right')
        
        x = np.append(x-xerr, [x[-1]+xerr[-1]]);
        y = np.append(y, y[-1])
        ax.step(x, y, color='maroon', where='post', lw=1, )
        return fig
    
    def multi_lc(self, names, ncols=5, row_height=2, fig_width=15, **kwargs):

        nrows = (len(names)+ncols-1)//ncols
        fig, axx = plt.subplots(ncols=ncols,  nrows=nrows, 
                                figsize=(fig_width,row_height*nrows),
                                sharex=True, sharey=True, 
                                gridspec_kw=dict(hspace=0, wspace=0, top=0.9, 
                                                left=0.1,bottom=0.1))
        fig.text(0.5, 0.07, 'Time', ha='center')
        fig.text(0.08, 0.5, 'Flux ratio', va='center', rotation='vertical') 
        for name, ax in zip(names, axx.flatten()):
            self.light_curve( name, ax=ax) 
            ax.set(**kwargs)
        return fig
    
    def select_nearby(self, uw_name:str, radius:float=5)->pd.DataFrame:
        """
        Return uwcat entries
        """
        skycoord = lambda info: SkyCoord(info.glon, info.glat, unit='deg', frame='galactic')
        sk = skycoord(self.uwcat.loc[uw_name])
        
        nearby = self.uwcat.select_cone(sk,radius).sort_values('sep')#['jname ts sep'.split()]
        
        def good_source(uw_name):
            if uw_name not in self: return False
            return self[uw_name]['light_curve'] is not None   
        idx = [i for i in range(len(nearby)) if good_source(nearby.iloc[i].jname) ]
        df = nearby.iloc[idx].copy()
        # add a column 'var' if find 4FGL variabilithy
        dfx = self.dfx.copy()
        dfx.index = dfx.uw_name
        df.loc[:, 'var'] = dfx.variability
        return df
    
chisq = lambda df : -2 * np.sum([f(1) for f in df.fit.values])

def lc_plot(lcf, name='', ax=None, color='maroon', label=None, **kwargs):
    """Draw a light curve plot

    lcf -- a "fluxes" DataFrame 
    """
    from wtlike.lightcurve import LCplotInfo        
    lc_info = LCplotInfo(lcf)
    
    fig, ax = plt.subplots(figsize=(6,2)) if ax is None else (ax.figure, ax)
    kw = dict( ylim=(0.05,20), yscale='log') #,yticks=[0.1,1,10], yticklabels='0.1 1 10'.split())
    kw.update(kwargs)
    ax.set(**kw)
    ax.axhline(1, color='0.5', ls='--')
    ylim= kw.get('ylim', None) 
    
    x,y,yerr,xerr = lc_info.errorbars
    if ylim is not None:
        y = y.clip(*ylim)
        
    ax.errorbar(x,y.clip(*ylim), yerr=yerr,#xerr=xerr, 
                fmt='o', color=color, lw=2, label=label);
    ax.text(0.95, 0.95, str(name), transform=ax.transAxes, 
            fontsize=10, va='top',ha='right')    
    x = np.append(x-xerr, [x[-1]+xerr[-1]]);
    y = np.append(y, y[-1])
    ax.step(x, y, color=color, where='post', lw=2, )
    if label is not None: ax.legend()
    return fig
    
def twodirections(bb, bbr, ax=None):

    fig, ax = plt.subplots(figsize=(8,2.5))  if ax is None else (ax.figues, ax)   
    lc_plot(bb.fits, ax=ax,  ylim=(0.3,3), 
            yticks=[0.25,0.5,1,2,4], yticklabels='1/4 1/2 1 2 4'.split(),
            label=f'forward  {chisq(bb.fits):4.1f}', color='green',alpha=0.5, 
           );
    lc_plot(bbr.fits, ax=ax,  ylim=(0.3,3), 
            yticks=[0.25,0.5,1,2,4], yticklabels='1/4 1/2 1 2 4'.split(),
            label=f'backward {chisq(bbr.fits):4.1f}', color='red',alpha=0.5, 
           );
    font=dict(family='monospace', size=10)
    ax.legend(prop=font).set_title(r'  Direction  $\chi^2$', prop=font)
    return fig

def show_lightcurve(source, p0=0.05, interval=7, comment='', nocaption=False):
    if type(source)==str:
        with capture_hide(f'Setup for {source}') as out:
            wtl = WtLike(source)
        if wtl is None:
            show(out)
            return
        
    else: wtl = source
    
    show(f"""###  {wtl.source_name}
        {comment}""")

    with capture_hide('BB processing output') as output:
        v = wtl.view(interval)
        bb, bbr = v.bb_view(p0), v.bb_view(p0, reverse=True)
    # show(output)
    
    fig = twodirections(bb,bbr)
    caption =None if nocaption else f"""BB-generated light curves for {wtl.source_name} with {interval}-day binning. 
    The green plot has the BB algorithm running forward in time, the red reversed.""" 
    show(fig, caption=caption)