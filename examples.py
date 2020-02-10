import sys

import numpy as np

from layered_model  import LayeredModel
from models         import grand_helmberger_lm
from plotting       import plot_dispersion, plot_eigenfuncs, plot_kernels_sxegn96, plot_kernels_srfker96, plot_model

def example_1():
    '''
    Demonstrates creation of LayeredModel and calculation of dispersion.
    This example requires CPS to be installed.
    '''

    # Create a layered model with increasing Vs with depth and constant Vp/Vs and rho/Vs ratios.
    h           = np.array([20.0, 20.0, 40.0, 10.0, 10.0, 0.0])
    vs          = np.array([3.70, 3.70, 3.70, 4.60, 4.60, 5.00])
    vp_vs       = 1.74
    rho_vs      = 0.77
    model       = LayeredModel( h, vs,
                                vp_vs   = vp_vs,
                                rho_vs  = rho_vs)
                                                

    # Plot the model.
    plot_model(model, show = False, params = ['vp', 'vs', 'rho']) 

    # Calculate fundamental-model Rayleigh wave dispersion for phase speed and group speed at 100 different periods.
    period_min  =  30.0   # seconds
    period_max  = 100.0
    n_periods   = 100
    periods     = np.linspace(period_min, period_max, num = n_periods)
    #
    period_list, c_list, u_list     = model.dispersion(
                                            periods,
                                            silent      = False,
                                            n_modes     = 1,
                                            rayleigh    = True,
                                            spherical   = False,
                                            eigenfuncs  = False,
                                            kernels     = False) 

    # Plot the dispersion curves.
    plot_dispersion(period_list,
                    c       = c_list,
                    u       = u_list,
                    show    = True,
                    x_label = 'Frequency / Hz',
                    y_label = 'Phase and group speed / km s$^{-1}$')

def example_2():
    '''
    Demonstrates the calculation of eigenfunctions and sensitivity kernels using the sregn96 command in CPS (see also example 5, where sensitivity kernels are calculated using the srfker command).
    This example requires CPS to be installed.
    '''

    # Create a simple test model.
    h           = np.array([20.0, 20.0, 40.0, 10.0, 10.0, 0.0])
    vs          = np.array([3.70, 3.70, 3.70, 4.60, 4.60, 4.60])
    vp_vs       = 1.74
    rho_vs      = 0.77
    # 
    model       = LayeredModel( h, vs,
                                vp_vs   = vp_vs,
                                rho_vs  = rho_vs)
    
    # Resample the model with finer layers.
    # Although this slows down the calculation, the eigenfunctions are only output at the layer boundaries, so it is necessary to have many layers to get a clear picture of the eigenfunctions.
    model = model.resample()
    periods = 20.0
    
    # Calculate the phase speed, group speed, eigenfunctions and kernels.
    T, c, u, ur, uz, tr, tz, kch, kca, kcb, kcr\
            = model.dispersion(periods,
                n_modes         = 1,
                rayleigh        = True,
                silent          = False,
                spherical       = False,
                eigenfuncs      = True,
                kernels         = True,
                rescale_kernels = True)

    # Plot the kernels.
    k_list = [kca[0], kcb[0], kcr[0]]
    labels = ['$\partial c/ \partial \\alpha$', '$\partial c/ \partial \\beta$', '$\partial c/ \partial \\rho$']
    plot_kernels_sxegn96(model, k_list, labels, show = False) 

    # Plot the eigenfunctions.
    labels      = ['Ur', 'Uz', 'Tr', 'Tz']
    mode        = 0
    eigenfuncs  = [ur[mode, :], uz[mode, :], tr[mode, :], tz[mode, :]]
    plot_eigenfuncs(model, eigenfuncs, labels, show = True)

def example_3():
    '''
    Demonstrates the calculation of sensitivity kernels using the srfker96 command in CPS (see also example 4, where sensitivity kernels are calculated using sregn96 command).
    This example requires CPS to be installed.
    '''
    
    # Use a pre-defined LayeredModel as the underlying structure.
    base = grand_helmberger_lm()
    
    # Create a model.
    h           = np.array([20.0, 20.0, 40.0, 10.0, 10.0, 0.0])
    vs          = np.array([3.70, 3.70, 3.70, 4.60, 4.60, 4.60])
    vp_vs       = 1.74
    rho_vs      = 0.77
    model       = LayeredModel( h, vs,
                                vp_vs   = vp_vs,
                                rho_vs  = rho_vs,
                                )#base    = base)
    
    # Resample the model with finer layers.
    # Although this slows down the calculation the eigenfunctions are only output at the layer boundaries, so it is necessary to have many layers to get a clear picture of the eigenfunctions.
    model = model.resample()

    # Plot the model.
    plot_model(model, show = False, params = ['vp', 'vs', 'rho']) 


    # Define the periods at which the sensitivity kernels will be calculated.
    periods     = np.array([    20.0,
                                25.0,
                                30.0,
                                40.0,
                                50.0,
                                60.0,
                                70.0,
                                85.0,
                                100.0,
                                120.0,
                                140.0])
    
    # Calculate the sensitivity of the phase speed (c) of the fundamental mode (0) to the shear wave speed (b).
    dv = 'c'
    iv = 'b'
    sensitivity = model.sensitivity(
                        periods,
                        mode        = 0,
                        dv          = 'c',
                        iv          = 'b',
                        spherical   = False)

    # Plot sensitivity kernels.
    plot_kernels_srfker96(
                        model,
                        periods,
                        sensitivity     = sensitivity,
                        dv              = dv,
                        iv              = iv,
                        smoothing       = True,
                        fill_between    = False,
                        colours         = None,
                        show            = True)

def example_4():
    '''
    Demonstrates the calculation of a synthetic receiver function.
    The trftn96 command must be installed from the CPS library.
    '''

    # Create a simple test model.
    # h    The thickness of the layers
    h           = np.array([10.0, 10.0, 10.0, 0.0])
    vs          = np.array([3.00, 3.30, 3.50, 4.60])
    vp_vs       = 1.74
    rho_vs      = 0.77
    model       = LayeredModel( h, vs,
                                vp_vs   = vp_vs,
                                rho_vs  = rho_vs,
                                )
    # Plot the model.
    plot_model(model, show = False, params = ['vp', 'vs', 'rho']) 
    
    # Calculate the receiver function for this model.
    # Try help(LayeredModel.rfn) to see what the parameters are.
    rf_trace = model.rfn(
                    phase   = 'P',
                    rayp    = 0.05,
                    alpha   = 1.0,
                    dt      = 0.1,
                    n       = 512,
                    delay   = 5.0,
                    silent  = False)
    
    # Plot the receiver function.
    rf_trace.plot(show = True)

def example_5():
    '''
    Demonstrates the calculation of the Earth response function.
    This example requires respknt to be installed.
    '''

    # Load a pre-defined layered model from <models.py>.
    # This model will be used as a 'base' to extend the model we define.
    # In this case we choose the shield model of Grand and Helmberger (1984).
    base = grand_helmberger_lm()
    
    # Create a simple test model.
    # h    The thickness of the layers
    h           = np.array([10.0, 10.0, 10.0, 0.0])
    vs          = np.array([3.00, 3.30, 3.50, 4.60])
    vp_vs       = 1.74
    rho_vs      = 0.77
    model       = LayeredModel( h, vs,
                                vp_vs   = vp_vs,
                                rho_vs  = rho_vs,
                                )
    # Plot the model.
    plot_model(model, show = False, params = ['vp', 'vs', 'rho']) 
    
    # Create a synthetic seismogram by calculating the Earth response function and convolving it with the source-time function <./data/st.sac>.
    t_synth, synth, synth_trace = model.earth_response(
                                            phase           = 'p',
                                            component       = 'z',
                                            ray_param       = 0.05,
                                            dt              = 0.01,
                                            Dt              = 45.0,
                                            src_time_file   = 'data/st.sac',
                                            clip            = False,
                                            silent          = False) 

    # Plot the synthetic seismogram.
    synth_trace.plot(show = True)
    
def main():
    
    example_dict = {1 : example_1,
                    2 : example_2,
                    3 : example_3,
                    4 : example_4,
                    5 : example_5}
   
    try:

        assert len(sys.argv) == 2
        example = int(sys.argv[1])

    except:

        print('usage: python3 examples.py i\nwhere i is an integer which identifies one of the examples')
        raise
  
    # Run the specified example.
    example_dict[example]()

if __name__ == '__main__':

    main()
