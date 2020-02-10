import numpy as np

from layered_model import LayeredModel

def grand_helmberger_lm():
    '''
    Create a model approximating the shield model (SNA) of Grand & Helmberger (1984), 
    Geophys. J. R. astr. Soc. (1984) 76, 399-438,
    Upper mantle shear structure of North America.
    '''
    
    gh_h = np.array([    3.75,
                         6.25,
                         6.25,
                         6.25,
                         6.25,
                         6.25,
                        13.00,
                        13.00,
                        13.00,
                        13.00,
                        13.00,
                        12.50,
                        12.50,
                        25.00,
                        25.00,
                        25.00,
                        25.00,
                        25.00,
                        25.00,
                        50.00,
                        50.00,
                        50.00,
                         0.00])
                            
    gh_vs = np.array([  3.2000,
                        3.6754,
                        3.6940,
                        3.7127,
                        3.7313,
                        3.7500,
                        4.6000,
                        4.6000,
                        4.6000,
                        4.6000,
                        4.6000,
                        4.5666,
                        4.5333,
                        4.5000,
                        4.5000,
                        4.4900,
                        4.4800,
                        4.4800,
                        4.4950,
                        4.5933,
                        4.6917,
                        4.7900,
                        5.4500])

    # Define Vp/Vs and density/Vs ratios.
    gh_vp_vs       = 1.74
    gh_rho_vs      = 0.77
           
    gh_layered_model = LayeredModel(gh_h,
                                    gh_vs,
                                    vp_vs   = gh_vp_vs, 
                                    rho_vs  = gh_rho_vs)
                                    
    return gh_layered_model
                                
def AK135f_lm(layers = 'default'):
    '''
    Create layered model from AK135f.

    Input

    layers  Array of layer thicknesses. The bottom layer (infinite half-space) should have a thickness of 0. If layers == 'default', 25 layers of increasing thickness will be used.

    Output

    AK135f_lm   A LayeredModel with the parameters of AK135f.
    '''
    
    # Load the AK135F mode. 
    AK135f_data = AK135f()
    n_params    = AK135f_data.shape[1] - 1
    
    if layers == 'default':
            
        n_layers    = 25
        layers      = np.array([5.0*(1.1**i) for i in range(n_layers)])
        depths      = np.cumsum(layers)
        depths      = np.insert(depths, 0, 0.0)
    
    AK135f_layered = np.zeros((n_params + 1, n_layers))
    AK135f_layered[0, :] = layers
    
    for i in range(n_params): 
    
        for j in range(n_layers):
        
            x0 = np.interp( depths[j],
                            AK135f_data[:, 0],
                            AK135f_data[:, i + 1])
            x1 = np.interp( depths[j + 1],
                            AK135f_data[:, 0],
                            AK135f_data[:, i + 1])
        
            j_in = np.where(
                        np.logical_and(
                            (AK135f_data[:, 0] > depths[j]),
                            (AK135f_data[:, 0] < depths[j + 1])))
                        
            z_in       = (AK135f_data[j_in, 0]).squeeze(axis = 0)
            x_in       = (AK135f_data[j_in, i + 1]).squeeze(axis = 0)
            
            z_span  = np.concatenate(([depths[j]], z_in, [depths[j + 1]]))
            x_span = np.concatenate(([x0], x_in, [x1]))
        
            z_x_int    = np.trapz(x_span, x = z_span)
            AK135f_layered[i + 1, j]  = z_x_int/(depths[j + 1] - depths[j])
    
    AK135f_lm = LayeredModel(   AK135f_layered[0, :],
                                AK135f_layered[3, :],
                                vp  = AK135f_layered[2, :],
                                rho = AK135f_layered[1, :])
                                
    return AK135f_lm

def AK135f():
    '''
    Loads the file 'data/AK135F_AVG.csv'
    See http://rses.anu.edu.au/seismology/ak135/ak135f.html 
    '''
    
    script_dir  = os.path.dirname(__file__)
    data_dir    = os.path.join( script_dir,
                                'data',
                                'AK135F_AVG.csv')
                                
    data        = np.loadtxt(data_dir, delimiter = ',')
    
    return data
