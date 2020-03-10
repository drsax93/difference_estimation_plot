import sys
import numpy as np
import pandas as pd # data manipulation and analysis
from scipy import stats
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns 



# SWARMPLOT

def swarmplot(df, indeces, ax, spread=5, trend=1, operation=np.mean,
              SWARM = 1, swarmPlot_kw=None, trendPlot_kw=None,
              color_palette=sns.set_palette('bright',10)):
    nCols = df.columns; ns = len(df) # total number of groups and samples
    xticks = []; xlab = []
    x_offset = 0
    for index in indeces: # loop over lists of ggroups (multiple controls)
        nC = len(index)
        if swarmPlot_kw is None: # swarmplot style
            swarmPlot_kw = {'label':'Swarm plot', 's':100/np.sqrt(ns)}  
        if trend: # trend line style
            if trendPlot_kw is None:
                trendPlot_kw = {'color':[.5,.5,.5], 'style':'-'}

        ym = []; yy = []; xm = []; xx = []
        for n,i in enumerate(index): # loop over groups
            y_ = df[i]; y_ = y_[~np.isnan(y_)] # take nans out
            if SWARM: # swarmplot - obtain envelope of histogram
                mag_y,_ = np.histogram(y_,bins=10)
                mag_y = np.interp(np.linspace(0, 10, len(y_)),
                                  range(0, len(mag_y)), mag_y)
                mag_y = mag_y/np.max(mag_y)
            else: # scatter
                mag_y = np.max(y_) - np.min(y_)
            off_x = mag_y/spread * (np.random.rand(len(y_))-.5) # random scattering amplitudes
            x_ = (x_offset+n) * np.ones(len(y_)) + off_x # x coords for scattering
            ax.plot(x_, y_, '.', markersize=swarmPlot_kw['s']) # plotting
            xticks.append(x_.mean()); xlab.append(i); ym.append(operation(y_))
            xm.append(x_.mean()); xx.append(x_);  yy.append(y_)
        x_offset+=n+1
        # plot trend line
        if trend==1:
            ax.plot(xm, ym, color=trendPlot_kw['color'],
                   linestyle=trendPlot_kw['style'])
        if trend>1: # paired plot
            for n in range(ns):
                x2p = [xx[i][n] for i in range(nC)]
                y2p = [yy[i][n] for i in range(nC)]
                ax.plot(x2p, y2p, color=trendPlot_kw['color'],
                       linestyle=trendPlot_kw['style'])
    # set axis label and lims
    plt.xticks(xticks, xlab)
    ax.set_ylabel(swarmPlot_kw['label'])
    ax.set_xlim(-1/spread, nC-1 +1/spread)
    miny = np.nanmin(df); maxy = np.nanmax(df)
    eps = (maxy - miny)/10
    ax.set_ylim(miny-eps, maxy+eps)
    sns.despine(ax=ax)
    
    return ax



# BOOTSTRAP ESTIMATION PLOT

def bootstrap(x, nsh = 5000, operation=np.mean):
    mean = []
    x_ = x; x_ = x_[~np.isnan(x_)]
    for n in range(nsh):
        xm = np.random.choice(x_,len(x_))
        mean.append(operation(xm))
    return np.asarray(mean)

def confInt(x,interval):
    # Calculate confident interval given by the extremes in 'interval' of array 'x'
    lenx = len(x[~np.isnan(x)])
    mean = np.nanmean(x)
    SEM = stats.sem(x, nan_policy='omit'); # Standard Error of the Mean
    ts = stats.t.ppf((1+interval)/2, lenx-1); # T-Score
    CI = ts*SEM # Confidence Intervals
    return mean-CI, mean+CI

def bootstrap_plot(df, indeces, ax, operation=np.mean, nsh=1000,
                    nbins=50, ci=.95, spread=5, SMOOTH=[1,1],
                   bootPlot_kw=None, color_palette=sns.set_palette('bright',10)):
    if bootPlot_kw is None:
        bootPlot_kw = {'label':'Difference plot'}
    x_offset = 0; nCtot = 0 # x-axis offset for multiple controls; total number of groups
    min_bc = []; max_bc = [] # mins and max values for y axis lims

    for index in indeces: # loop over lists of ggroups (multiple controls)
        nC = len(index); nCtot+=nC
        # plot control sample
        ref =  df[index[0]]; offset = ref.mean()
        plt.plot(x_offset, 0, 'ko', markersize=10)
        start = x_offset; fin = x_offset + nC-1 + .5/spread
        plt.hlines(0, start, fin, linestyle='--')
        x_offset+=1
        
        for n, i in enumerate(index[1:]): # loop over test groups
            y_ = df[i]; y_ = np.asarray(y_[~np.isnan(y_)]) # exclude possible nans if unpaired analysis
            m_ = bootstrap(y_, nsh=nsh, operation=operation) # perform bootstrap
            m_h = np.histogram(m_, bins=nbins)
            m_pdf = m_h[0] / (np.max(m_h[0]) * 2*spread) # obtain normalised dist to fit the swarmplot spread
            m_binCentres = []
            for mn in range(len(m_h[0])): # obtain the centres of the hist bins
                m_binCentres.append(np.mean([m_h[1][mn+1], m_h[1][mn]]))
            m_binCentres = np.asarray(m_binCentres)
            min_bc.append(np.min(m_binCentres)-offset)
            max_bc.append(np.max(m_binCentres)-offset)
            if SMOOTH[0]: # smooth dist wit gaussian
                m_pdf = gaussian_filter1d(m_pdf, SMOOTH[1])
            # find conf interval - take samples from sorted dist
            ci_ind = np.round((nsh - nsh*ci)/2).astype(int)
            m_sort = np.sort(m_)
            CI_ = [m_sort[ci_ind],m_sort[-ci_ind]]
            # obtain theorethical samples from normal dist
    #         CI = confInt(y_, interval=ci)
    #         print(ci_ind,CI, CI_)
            # Plot distribution
            ax.plot(n+x_offset, m_binCentres.mean()-offset, 'ko', markersize=10) # plot black dot
            ax.fill(m_pdf + n+x_offset, m_binCentres-offset, color=color_palette[n+x_offset]) # plot dist
            ax.vlines(n+x_offset, CI_[0]-offset, CI_[1]-offset, linewidth=2) # plot CI
        x_offset+=n+1
    # labels and axes lims
    ax.set_ylabel(bootPlot_kw['label'])
    miny = np.min([-0.05, np.min(min_bc)])
    maxy = np.max([0.05, np.max(max_bc)])
    eps = (maxy - miny)/10
    ax.set_ylim(miny-eps, maxy+eps)
    ax.set_xlim(-2/spread, nCtot-1 +2/spread)
    sns.despine(ax=ax)
        
    return ax



# MAIN FUNCTION THAT PUTS THE TWO TOGETHER

def estimation_plot(input_, indeces, vertical=1, trend=1, spread=5,
                    operation=np.mean, SWARM=1, nsh=5000, ci=.95, nbins=50,
                    SMOOTH=[1,3], swarmPlot_kw=None, bootPlot_kw=None,
                    color_palette=sns.color_palette('bright',10),
                    FontScale=2):
    ''' INPUTS:
    - input_ = dict() containing the samples, indeces are labels
    - indeces = list of indeces used for multiple control analysis;
            each list element contains the indeces of the samples to compare
            in each analysis -
            e.g. list(ind1,ind2) for 2 controls or list(ind) for just one control
    - vertical = if true used a cumming's estimation layout, Gardner-Altman otherwise
    - trend = if 0 plots no trend line
            if 1 plots the trend line bw mean of samples
            if >1 plots a trend line per sample (MAKE SURE DATA IS PAIRED)
    - spread = control spread of swarmplot and heigght of bootstrapped distribution
    - operation = specify which type of statistic to measure - e.g mean, median, ...
    - SWARM = set to 1 to plot a swarmplot, otherwise scatter uniformly
    - nsh = number of bootstrap samples
    - ci = confidence interval as ratio - e.g. .95
    - nbins = number of bins to estimate bootsrap distribution
    - SMOOTH = list of 2 elements, the first specifys whether to smooth the
            bootstrapped distribution, the second indicates the SD
    - swarmPlot_kw = keywords to modify the style of swarmPlot (to insert more)
    - bootPlot_kw = keywords to modify the style of difference plot (to insert more)
    - color_palette = seaborn color_palette or list of colors to use\
    - FontScale = seaborn font_scale parameter

    OUTPUTS:
    fig, axs = figure and 2 axes handles
    '''

    df_ = []
    for i in input_.keys():
        df_.append(pd.DataFrame({i:input_[i]}))
    df = pd.concat(df_, axis=1)
    cols = df.columns; nC = len(cols)
    
    # Set up the figure
    sns.set(font_scale=FontScale); sns.set_style('ticks')
    if vertical: # Cumming's est plot
        fig, axs = plt.subplots(2, sharex=True, sharey=False,
            gridspec_kw={'hspace': 0},
            figsize=(6*nC,5*nC))
    else: # G-A plot
        fig, axs = plt.subplots(1,2, sharex=True, sharey=False,
            figsize=(8*nC,3*nC))

    # Swarmplot
    swarmplot(df, indeces, axs[0], spread=spread, trend=trend,
              operation=operation, swarmPlot_kw=swarmPlot_kw,
              color_palette=color_palette)
    
    # Distribution plot
    bootstrap_plot(df, indeces, axs[1], spread=spread, ci=ci, nbins=nbins,
                   operation=operation, bootPlot_kw=bootPlot_kw,
                  color_palette=color_palette)
    
    return fig,axs


    '''
    EXAMPLE USE:
    
    - Unpaired example:

    import difference_estimation_plot as dpl
    input_ = {'sample 1': np.random.rand(100), 'sample 2': np.random.rand(90) + 0.4,
         'sample 3': np.random.rand(200) - 0.2}
    KEYS = list(input_.keys())
    fig,axs = dpl.estimation_plot(input_, [KEYS])

    - Paired example:

    import difference_estimation_plot as dpl
    input_ = {'sample 1': np.random.rand(100), 'sample 2': np.random.rand(100) + 0.4,
         'sample 3': np.random.rand(100) - 0.2}
    KEYS = list(input_.keys())
    fig,axs = dpl.estimation_plot(input_, [KEYS], trend=2)

    - Median difference example:

    import difference_estimation_plot as dpl
    input_ = {'sample 1': np.random.rand(100), 'sample 2': np.random.rand(100) + 0.4,
         'sample 3': np.random.rand(100) - 0.2}
    KEYS = list(input_.keys())
    fig,axs = dpl.estimation_plot(input_, [KEYS], trend=1, operation=np.median)
    
    - Multiple controls
    input_ = {'sample 1': np.random.rand(100), 'sample 2': np.random.rand(100) + 0.4,
             'sample 3': np.random.rand(100) - 0.2, 'sample 4': np.random.rand(100) - 0.1}
    KEYS = list(input_.keys())
    fig,axs = dpl.estimation_plot(input_, [KEYS[:2], KEYS[2:]], trend=1)

    '''