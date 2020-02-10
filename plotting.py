import  matplotlib
from    matplotlib.patches  import  Ellipse
import  matplotlib.pyplot   as      plt
import  numpy as np

# Default optional arguments.
# ax    The Matplotlib axis (if None, an axis will be created).
# show  If True, show the plot when it is ready
# x_label   X-axis label (if None, a default may be used).
# y_label   Y-axis label (if None, a default may be used).
# x_scale   Scaling for the input x-data (usually 1.0 by default).
# y_scale   Scaling for the input y-data (usually 1.0 by default).
# title     If not None, the title string.
# path_out  If not None, the image will be saved to the specified path.

def plot_model(model, ax = None, show = True, params = ['vs']):
    '''
    Plot a LayeredModel.

    Required input

    model   The LayeredModel.
    
    Optional input
    
    See the list of default optional arguments at the top of this file.
    params  A list of which parameters to plot, from 'vs', 'vp', 'rho' and 'vp_vs'.

    Output

    ax      The Matplotlib axis.
    '''

    # Create array of layer corners for plotting.
    h_cum_c         = np.zeros(2*model.n)
    h_cum_c[: -1]   = np.repeat(model.cum_h, 2)[1 :]
    h_cum_c[-1]     = h_cum_c[-2] + 1000.0
    
    x_cs    = []
    labels  = []
    for param in params:
    
        if param == 'vs':
        
            x_c             = np.repeat(model.vs,    2)
            label         = r'$\beta$'
            
        elif param == 'vp':
            
            x_c             = np.repeat(model.vp,    2)
            label         = r'$\alpha$'
            
        elif param == 'rho':
            
            x_c             = np.repeat(model.rho, 2)
            label         = r'$\rho$'
        
        elif param == 'vp_vs':
                        
            x_c     = np.repeat(model.vp/model.vs, 2)    
            label = r'$\alpha$/r$\beta$'
            
        x_cs.append(x_c)
        labels.append(label)
    
    # Create axes if necessary.
    if ax is None:

        fig = plt.figure(figsize = (5.5, 5.5))
        ax  = plt.gca()
    
    for x_c, label in zip(x_cs, labels):
        
        ax.plot(x_c, h_cum_c, '-', label = label, alpha = 0.4)
    
    # Tidy up the plot.
    #ax.set_xlim([3.5, 5.0])
    ax.set_ylim([0.0, h_cum_c[-2]*1.05])
    ax.invert_yaxis()
    ax.grid()
    #ax.set_xlabel(x_label,  fontsize = 14)
    ax.legend()
    ax.set_ylabel('Depth / km',                 fontsize = 14)
    
    # Show the plot.
    if show:
        
        plt.show()
        
    return ax
      
def plot_dispersion(T, c = None, u = None, show = True, out_file = None, x_label = None, y_label = None, ax = None, x_scale = 1.0, y_scale = 1.0):
    '''
    Plot the dispersion calculated by LayeredModel.dispersion().
    
    Required input:

    T           A list of the periods (s).

    Optional input:
    See also the list of default optional arguments at the top of this file.
    c           A list of the phase speed for each mode at each period.
    u           A list of the group speed for each mode at each period.

    Output:

    ax          The Matplotlib axis.
    '''
    
    if ax is None:
        
        fig = plt.figure()
        ax  = plt.gca()

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    n_modes = len(T)
    if c is not None:
        
        for i in range(n_modes):

            label_str = 'Phase speed, mode {:d}'.format(i)
            color = color_cycle[i]
            ax.plot(x_scale/T[i], y_scale*c[i], linestyle = '-', color = color, label = label_str)
           
            #ax.scatter(x_scale/T[i], y_scale*c[i])

    if u is not None:
        
        for i in range(n_modes):
            
            label_str = 'Group speed, mode {:d}'.format(i)
            color = color_cycle[i]
            ax.plot(x_scale/T[i], y_scale*u[i], linestyle = ':', color = color, label = label_str)

    if x_label is not None:
        
        ax.set_xlabel(x_label,             fontsize = 14)
        
    if y_label is not None:
        
        ax.set_ylabel(y_label,   fontsize = 14)
       
    ax.legend()
    plt.tight_layout()
        
    if out_file is not None:
        
        plt.savefig(out_file, dpi = 300, bbox_inches = 'tight')
        
    if show:
        
        plt.show()
        
    return ax

def plot_eigenfuncs(model, u_list, labels, title = None, path_out = None, norm = None, z_lines = None, show = True):
    '''
    Plot the eigenfunctions calculated by the sregn96/slegn96 functions in CPS.

    Required input:
    model   The LayeredModel.
    u_list  A list of eigenfunctions output by model.dispersion().
    labels  The labels for the eigenfunctions in u_list.

    Optional input:
    See the list of default optional arguments at the top of this file.
    norm    If 'max', the eigenfunctions will be normalised by the maximum value amongst all of the eigenfunctions.
    z_lines If not None, these vertical lines will be add to the plot.

    '''


    fig = plt.figure()
    ax  = plt.gca()
    
    if norm == 'max':

        max_list = [np.max(np.abs(u)) for u in u_list]
        u_max = np.max(max_list)

        u_list = [u/u_max for u in u_list]

    for u, label in zip(u_list, labels):

        ax.plot(u, model.cum_h, label = label)

    if norm == 'max':
        ax.set_xlim([-1.05, 1.05])

    #ax.set_ylim([0.0, model.cum_h[-2]*1.05])
    ax.set_ylim([0.0, model.cum_h[-1]])
    ax.invert_yaxis()
    ax.grid()
    ax.set_xlabel('Normalised eigenfunction',  fontsize = 12)
    ax.set_ylabel('Depth / km',                 fontsize = 14)
    ax.legend()
    if title is not None:

        ax.set_title(title)

    if z_lines is not None:

        for z_line in z_lines:

            ax.axhline(z_line, linestyle = ':', color = 'k')

    ax.axvline(linestyle = ':', color = 'k')

    if path_out is not None:

        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

def plot_kernels_sxegn96(model, k_list, labels, title = None, path_out = None, z_lines = None, ax = None, x_label = 'Sensitivity per km', x_lims = None, legend_pos = 'best', show = True):
    '''
    Plot the kernels calculated by the sregn96/slegn96 functions in CPS.

    Required input:
    model   The LayeredModel.
    k_list  A list of kernels output by model.dispersion().
    labels  The labels for the kernels in k_list.

    Optional input:
    See the list of default optional arguments at the top of this file.
    '''

    if ax is None:

        fig = plt.figure()
        ax  = plt.gca()
    
    for k, label in zip(k_list, labels):

        ax.plot(k, model.cum_h, label = label)

    if x_lims is not None:

        ax.set_xlim(x_lims)

    ax.set_ylim([0.0, model.cum_h[-1]])
    ax.invert_yaxis()
    ax.grid()
    ax.set_xlabel(x_label,  fontsize = 12)
    ax.set_ylabel('Depth / km',                 fontsize = 14)
    ax.legend(loc = legend_pos)
    if title is not None:

        ax.text(0.1, 0.85, title, transform = ax.transAxes, fontsize = 28)

    if z_lines is not None:

        for z_line in z_lines:

            ax.axhline(z_line, linestyle = ':', color = 'k')

    ax.axvline(linestyle = ':', color = 'k')

    if path_out is not None:

        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')

    if show:

        plt.show()

def plot_kernels_srfker96(layered_model, periods, sensitivity = None, ax = None, show = True, dv = 'c', iv = 'b', smoothing = False, fill_between = False, colours = None):
    '''
    Plot the sensitivity kernels calculated by the CPS function srfker96.

    Required input:
    layered_model   A LayeredModel.
    periods         Periods at which the sensitivity is calculated.

    Optional input:
    See also the list of default optional arguments at the top of this file.
    sensitivity     The sensitivity of the dependent variable (dv) to the independent variable (iv) as a function of period and depth. If None, the sensitivity will be calculated.
    dv              A string identifying the dependent variable. One of 'c' (phase speed), 'u' (group speed) or 'g' (?).
    iv              A string identifying the independent variable. One of 'a' (P-wave speed), 'b' (S-wave speed), 'qa' (P-wave quality factor) and 'qb' (S-wave quality factor).
    smoothing       If True, the kernels will be smoothed before plotting.
    fill_between    If True, the space between the kernel and depth axis will be filled.
    colours         The colour to plot each period. If None, a default colour map will be used ('Spectral').
    
    Output:

    ax              The Matplotlib axis.
    '''
    
    if sensitivity is None:
        
        sensitivity = layered_model.sensitivity(periods, dv = dv, iv = iv)

    sensitivity = sensitivity*1.0E3
    n_periods = len(periods)
    
    if smoothing:
        
        n_resample  = layered_model.n*10
        z           = np.linspace(
                            0.0,
                            layered_model.cum_h[-1],
                            num = n_resample)
        
        z_mp        = np.zeros(layered_model.n)
        z_mp[:-1]   = layered_model.midpoints
        z_mp[-1]    = z_mp[-2]*1.1   
        x           = np.zeros((n_periods, n_resample))
        
        for i in range(n_periods):
            
            x[i, :] = smooth_box(np.interp(z, z_mp, sensitivity[i, :]), 5)
        
    else:
        
        z         = np.zeros(2*layered_model.n)
        z[: -1]   = np.repeat(layered_model.cum_h, 2)[1 :]
        z[-1]     = z[-2] + 1000.0
        
        x       = np.zeros((n_periods, 2*layered_model.n))
        for i in range(n_periods):
            
            x[i, :] = np.repeat(sensitivity[i, :],    2)
            
    # Create axes if necessary.
    if ax is None:

        fig = plt.figure(figsize = (5.5, 5.5))
        ax  = plt.gca()
        
    # Plot sensitivity kernel.
    n_periods = len(periods)
    c_map = matplotlib.cm.get_cmap('Spectral')
    for i in range(n_periods):
        
        if colours is None:
            
            frac    = i/(n_periods - 1)
            colour  = c_map(frac)
            
        else:
            
            colour  = colours[i]
        
        #sens_c            = np.repeat(sensitivity[i, :],    2)
        
        if fill_between:
            
            ax.fill_betweenx(z,
                            0.0,
                            x2          = x[i, :],
                            color       = colour,
                            alpha       = 0.5)
            ax.plot(x[i, :], z,
                    color = colour,
                    label       = '{:>5.1f}'.format(periods[i]))
            
        else:
        
            ax.plot(x[i, :],
                    z,
                    color = colour,
                    label = '{:>5.1f}'.format(periods[i]))
    
    # Tidy up the plot.
    #ax.set_xlim([3.5, 5.0])
    ax.set_ylim([0.0, z[-2]*1.05])
    #ax.set_ylim([0.0, 350.0])
    ax.invert_yaxis()
    ax.grid(color = 'k', linestyle = ':')
    dv_str_dict = { 'c' : 'C',
                    'u' : 'U',
                    'g' : 'g' }
    iv_str_dict = { 'a'     : 'V$_{p}$',
                    'b'     : 'V$_{s}$',
                    'qa'    : 'Q$_a$',
                    'qb'    : 'Q$_b$' }
    dv_str = dv_str_dict[dv]
    iv_str = iv_str_dict[iv]
    #ax.set_xlabel('Sensitivity, $\partial${}/$\partial${} / 10$^{{-3}}$ km$^{{-1}}$'.format(dv_str, iv_str),    fontsize = 14)
    ax.set_xlabel('$\partial${}/$\partial${} / 10$^{{-3}}$ km$^{{-1}}$'.format(dv_str, iv_str),    fontsize = 12)
    ax.set_ylabel('Depth / km',     fontsize = 12)
    ax.legend(title = 'Period / s', loc = 'lower right')
    
    if show:
        
        plt.show()
        
    return ax

def smooth_box(y, n):
    '''
    Smooth the input y using a simple box filter with a 
    width of n points.
    '''
    
    box         = np.ones(n)/n
    y_smooth    = np.convolve(y, box, mode = 'same')
    
    return y_smooth
