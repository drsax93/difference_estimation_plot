import sys
import numpy as np
import pandas as pd # data manipulation and analysis
from scipy import stats
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns 



# SWARMPLOT

def swarmplot(df, indeces, ax, vertical, spread=5, trend=1, operation=np.mean,
              paired=False, SWARM = 1, swarmPlot_kw={}, trendPlot_kw={},
              color_palette=sns.set_palette('bright',10)):
    ### PLOTTING STYLE PARAMETERS
    nCols = 0 # total number of groups and samples
    for l in indeces: nCols+=len(l)
    ns = len(df)
    try: swarmPlot_kw['label'] # swarmplot style
    except: swarmPlot_kw['label'] = 'Swarm plot'
    try: swarmPlot_kw['s']
    except: swarmPlot_kw['s'] = 100/np.sqrt(ns)  
    if trend: # trend line style
        try: trendPlot_kw['color']
        except: trendPlot_kw['color'] = [.5,.5,.5]
        try: trendPlot_kw['style']
        except: trendPlot_kw['style'] = '-'
                
    xticks = []; xlab = []
    x_offset = 0
    for index in indeces: # loop over lists of groups (multiple controls)
        nC = len(index)              
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
            ax.plot(x_, y_, '.', markersize=swarmPlot_kw['s'], color=color_palette[n+x_offset]) # plotting
            xticks.append(x_.mean()); xlab.append(i); ym.append(operation(y_))
            xm.append(x_.mean()); xx.append(x_);  yy.append(y_)
        x_offset+=n+1
        
        if paired and trend: # paired plot
            for n in range(ns):
                x2p = [xx[i][n] for i in range(nC)]
                y2p = [yy[i][n] for i in range(nC)]
                ax.plot(x2p, y2p, color=trendPlot_kw['color'],
                       linestyle=trendPlot_kw['style'])
        elif trend: # plot trend line
            ax.plot(xm, ym, color=trendPlot_kw['color'],
                   linestyle=trendPlot_kw['style'])
    # set axis label and lims
    ax.set_xticks(range(nCols))
    ax.set_xticklabels(xlab)
    ax.set_ylabel(swarmPlot_kw['label'])
    miny = np.nanmin(df); maxy = np.nanmax(df)
    eps = (maxy - miny)/10
    ax.set_ylim(miny-eps, maxy+eps)
    sns.despine(ax=ax)
    if vertical:
        sns.despine(ax=ax, bottom=True)
        ax.set_xticks([])
    
    return ax



# BOOTSTRAP ESTIMATION PLOT

def bootstrap(x, nsh = 10000, operation=np.mean):
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

def bootstrap_plot(df, indeces, ax, operation=np.mean, nsh=10000, vertical=1,
                    paired=False, nbins=50, ci=.95, spread=5, SMOOTH=[1,3],
                   bootPlot_kw={}, color_palette=sns.set_palette('bright',10)):
    ### PLOTTING STYLE PARAMETERS
    nCols = 0 # total number of groups and samples
    for l in indeces: nCols+=len(l)
    ns = len(df)
    try: bootPlot_kw['label']
    except:
        if operation==np.mean:
            bootPlot_kw['label'] = 'Mean difference'
        elif operation==np.median:
            bootPlot_kw['label'] = 'Mean difference'
        else:
            bootPlot_kw['label'] = 'Mean difference'
    try: bootPlot_kw['ci_size']
    except: bootPlot_kw['ci_size'] = nCols*4 # size of black dot
    try: bootPlot_kw['ci_width']
    except: bootPlot_kw['ci_width'] = nCols # width of ci line
    try: bootPlot_kw['ref_width']
    except: bootPlot_kw['ref_width'] = nCols/2 # width of ref line
    try: bootPlot_kw['ref_ls']
    except: bootPlot_kw['ref_style'] = '--' # style of ref line
    
    x_offset = 0;  # x-axis offset for multiple controls
    min_bc = []; max_bc = [] # mins and max values for y axis lims
    m_b = []; ci_b = [] # mean and ci of bootstrapped difference distribution
    
    for index in indeces: # loop over lists of groups (multiple controls)
        nC = len(index)
        # plot control sample
        ref =  df[index[0]]
        if paired: offset = 0
        else: offset = ref.mean()
        if vertical:
            plt.plot(x_offset, 0, 'ko', markersize=bootPlot_kw['ci_size'])
            start = x_offset; fin = x_offset + nC-1 + 1.5/spread
            plt.hlines(0, start, fin, linewidth=bootPlot_kw['ref_width'],
                      linestyle=bootPlot_kw['ref_style'])
        else:
            start = x_offset; fin = x_offset + nC-1 + 1.5/spread
            plt.hlines(0, start, fin, linewidth=bootPlot_kw['ref_width'],
                      linestyle=bootPlot_kw['ref_style'])
        x_offset+=1
        m_b.append([]); ci_b.append([])
        for n, i in enumerate(index[1:]): # loop over test groups
            y_ = df[i]; y_ = np.asarray(y_[~np.isnan(y_)]) # exclude possible nans if unpaired analysis
            if paired:
                m_ = bootstrap(y_-ref, nsh=nsh, operation=operation) # paired diff bootstrap
            else:
                m_ = bootstrap(y_, nsh=nsh, operation=operation) # bootstrap
            m_h = np.histogram(m_, bins=nbins)
            if len(index)>2: # obtain normalised dist to fit the swarmplot spread
                m_pdf = m_h[0] / (np.max(m_h[0]) * (spread/1.5))
            else:
                m_pdf = m_h[0] / (np.max(m_h[0]) * spread)
            m_binCentres = []
            for mn in range(len(m_h[0])): # obtain the centres of the hist bins
                m_binCentres.append(np.mean([m_h[1][mn+1], m_h[1][mn]]))
            m_binCentres = np.asarray(m_binCentres)
            min_bc.append(np.min(m_binCentres)-offset)
            max_bc.append(np.max(m_binCentres)-offset)
            if SMOOTH[0]: # smooth dist wit gaussian
                m_pdf = gaussian_filter1d(m_pdf, SMOOTH[1])
            m_pdf[0] = 0; m_pdf[-1] = 0 # make sure distribution touches CI line
            # find conf interval - take samples from sorted dist
            ci_ind = np.round((nsh - nsh*ci)/2).astype(int)
            m_sort = np.sort(m_)
            CI_ = [m_sort[ci_ind],m_sort[-ci_ind]]
            # obtain theorethical samples from normal dist
    #         CI = confInt(y_, interval=ci)
    #         print(ci_ind,CI, CI_)
            # Plot distribution
            m_b[-1].append(m_binCentres.mean()-offset)
            ci_b[-1].append([CI_[0]-offset, CI_[1]-offset])
            ax.plot(n+x_offset, m_.mean()-offset, 'ko',
                    markersize=bootPlot_kw['ci_size']) # plot black dot
            ax.fill(m_pdf + n+x_offset, m_binCentres-offset, color=color_palette[n+x_offset]) # plot dist
            ax.vlines(n+x_offset, CI_[0]-offset, CI_[1]-offset,
                      linewidth=bootPlot_kw['ci_width']) # plot CI
        x_offset+=n+1
    # labels and axes lims
    ax.set_xticks(range(nC))
    ax.set_xticklabels(index)
    ax.set_ylabel(bootPlot_kw['label'])
    miny = np.min([-0.05, np.min(min_bc)])
    maxy = np.max([0.05, np.max(max_bc)])
    eps = (maxy - miny)/10
    ax.set_ylim(miny-eps, maxy+eps)
    if vertical:
        sns.despine(ax=ax)
    else:
        sns.despine(ax=ax, left=True, right=False)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        
    return ax, m_b, ci_b



# MAIN FUNCTION THAT PUTS THE TWO TOGETHER

def estimation_plot(input_, indeces, vertical=1, trend=1, spread=5, paired=False,
                    operation=np.mean, SWARM=1, nsh=10000, ci=.95, nbins=50,
                    SMOOTH=[1,3], swarmPlot_kw={}, bootPlot_kw={},
                    trendPlot_kw={}, color_palette=sns.color_palette('bright',10),
                    FontScale=2, figsize=None, stat=True):
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
    - paired = set to True if data is paired
    - operation = specify which type of statistic to measure - e.g mean, median, ...
    - SWARM = set to 1 to plot a swarmplot, otherwise scatter uniformly
    - nsh = number of bootstrap samples
    - ci = confidence interval as ratio - e.g. .95
    - nbins = number of bins to estimate bootsrap distribution
    - SMOOTH = list of 2 elements, the first specifys whether to smooth the
            bootstrapped distribution, the second indicates the SD
    - swarmPlot_kw = keywords to modify the style of swarmPlot (to insert more)
            check individual function for more info
    - bootPlot_kw = keywords to modify the style of difference plot (to insert more)
            check individual function for more info
    - trendPlot_kw = keywords to modify the style of trend line plot
            check individual function for more info
    - color_palette = seaborn color_palette or list of colors to use\
    - FontScale = seaborn font_scale parameter
    - figsize = size of the figure to plot as per plt figsize parameter

    OUTPUTS:
    fig, axs = figure and 2 axes handles
    '''

    df_ = []
    for i in input_.keys():
        df_.append(pd.DataFrame({i:input_[i]}))
    df = pd.concat(df_, axis=1)
    nCols = 0 # total number of groups and samples
    for l in indeces: nCols+=len(l)
    ns = len(df)
    
    # Set up the figure
    sns.set(font_scale=FontScale); sns.set_style('ticks')
    if vertical: # Cumming's est plot
        if figsize==None: figsize = (6*nCols,4*nCols)
        fig, axs = plt.subplots(2, sharex=False, sharey=False,
            gridspec_kw={'hspace': 0.1}, figsize=figsize)
    else: # G-A plot
        if figsize==None: figsize = (6*nCols,3*nCols)
        fig, axs = plt.subplots(1,2, sharex=False, sharey=False,
              gridspec_kw={'wspace': 0.1, 'width_ratios': [nCols,nCols-1]},
              figsize=figsize)

    # Swarmplot
    swarmplot(df, indeces, axs[0], vertical, spread=spread, trend=trend, paired=paired,
              operation=operation, swarmPlot_kw=swarmPlot_kw, trendPlot_kw=trendPlot_kw, 
              color_palette=color_palette)
    
    # Distribution plot
    axs[1], m_b, ci_b = bootstrap_plot(df, indeces, axs[1], spread=spread, ci=ci, nbins=nbins,
                                       paired=paired, operation=operation, SMOOTH=SMOOTH,
                                       vertical=vertical,
                                       bootPlot_kw=bootPlot_kw, color_palette=color_palette)
    # set common x axis limits
    xlim = (-1/spread * (nCols/2), nCols-1 + 1/spread * (nCols/2))
    axs[0].set_xlim(xlim); axs[1].set_xlim(xlim)
    if not vertical:
        axs[1].set_xlim(xlim[0]+1, xlim[1])
    
    if stat: return fig, axs, m_b, ci_b
    else: return fig, axs


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