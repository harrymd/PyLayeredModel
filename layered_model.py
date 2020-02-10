import  os
import  subprocess
import  sys

import  numpy               as      np
try:
    import  obspy
except ImportError:
    print('Could not import obspy. It will not be possible to use the following functions: LayeredModel.rfn(), LayeredModel.earth_response(), convolve_traces()')

class LayeredModel():
    '''
    Represents the Earth as 1D stack of isotropic layers.
    For each layer, the thickness, S wavespeed, P wavespeed, density
    and attenuation parameters are specified.
    '''
    
    def __init__(self, h, vs, vp = None, rho = None, vp_vs = None, rho_vs = None, qp = 0.0, qs = 0.0, etap = 0.0, etas = 0.0, frefp = 1.0, frefs = 1.0, name = 'model.txt', base = None, z_pad = None):
        '''
            Input:
        
        (n)     The number of layers (including the bottom layer which
                    is an infinite half space).
        h       (n-by-1) array listing thickness of each
                    The thickness of the bottom layer is ignored.
        vs      (n-by-1) array listing shear speed of each layer.
        vp      (n-by-1) array listing compression speed of each layer.
        rho     (n-by-1) array listing density of each layer.
        vp_vs   (n-by-1 or scalar): Vp/Vs ratio.
        rho_vs  (n-by-1 or scalar): rho/Vs ratio.
        Q factor is given by Q(f) = Qref*((f/fref)^eta) for P and S waves.
        qp, qs  (n-by-1 or scalar): Q-factor of attenuation (at the reference frequency) for P- and S-waves. 
        etap, etas (n-by-1 or scalar): Q-factor frequency exponent (set to 0 for frequency-independent attenuation) for P- and S-waves.
        frefp, frefs (n-b-1 or scalar): Q-factor reference frequency for P- and S-waves.
        name    File name (used only when output is necessary).
        base    Another layered model used to fill deeper layers.
        z_pad   Add an extra layer add the bottom with the same
                properties as the half-space to extend the thickness
                of the non-half-space layers to z_pad. This is useful
                for calculating consistent synthetic seismograms.
            
            Notes:
        
        Only one of vp and vp_vs should be specified.
        Only one of rho and rho_vs must be specified.
                
        '''
        
        vs = np.atleast_1d(vs)
        
        # Fill the Vp, Vs and rho arrays.
        if vp is None:
            
            vp          = vs*vp_vs
            self.vp_vs  = vp_vs
            
        else:
            
            vp = np.atleast_1d(vp)
            self.vp_vs  = vp/vs

        if rho is None:
            
            rho         = vs*rho_vs
            self.rho_vs = rho_vs
            
        else:
            
            rho = np.atleast_1d(rho)
            self.rho_vs = rho/vs
                   
        # Fill the h array (layer thicknesses).
        # The infinitehalf-space at the bottom is assigned a thickness of 0.
        self.h          = np.atleast_1d(h)
        self.h[-1]      = 0.0

        # Fill the class attribute. 
        self.vs         = vs
        self.vp         = vp
        self.rho        = rho
        self.qp         = qp
        self.qs         = qs
        self.etap       = etap
        self.etas       = etas
        self.frefp      = frefp
        self.frefs      = frefs
        self.name       = name
        
        # Extend the model with some deeper layers.
        if base is not None:
            
            self.extend(base)
           
        # Add a layer with the same properties of the bottom layer to make the total thickness of the layers equal to z_pad.
        if z_pad is not None:
            
            h_pad       = z_pad - np.sum(self.h)
            self.h      = np.insert(self.h,   -1, h_pad)
            self.vs     = np.insert(self.vs,  -1, vs[-1])
            self.vp     = np.insert(self.vp,  -1, vp[-1])
            self.rho    = np.insert(self.rho, -1, rho[-1])
            
        # Count the number of layers.
        self.n          = len(self.h)
        
    def query(self, z, mode = 'i'):
        '''
        Returns the value of a model parameter at a specified depth.
        
        Input
        
        z       The depth at which to query the model.
        mode    Choose which parameter to query:
                    'i'     Layer number
                    'vs'    Shear wave speed
        '''

        condition   = (z >= self.cum_h[:,   np.newaxis])
        i           = np.argmin(condition, axis = 0) - 1
        i[i == -1]  = self.n - 1
        
        if mode == 'i':
            
            return i
            
        elif mode == 'vs':
            
            return self.vs[i]
    
    @property
    def tot_h(self):
        '''
        Adds up the total thickness of the layers.
        '''
        
        return np.sum(self.h)
        
    @property
    def cum_h(self):
        '''
        Returns the cumulative thickness (i.e. the depth to the base of) the layers.
        '''
        
        h_cum       = np.zeros(self.n)
        h_cum[1 :]  = np.cumsum(self.h)[: -1]
        
        return h_cum
    
    @property
    def midpoints(self):
        '''
        Mid points of layers.
        '''
        
        mid_points = (self.cum_h[:-1] + self.cum_h[1:])/2.0
        
        return mid_points
    
    def insert_node(self, z, vs, vp = None, rho = None):
        '''
        Not implemented.
        '''

        raise NotImplementedError
    
    def extend(self, base):
        '''
        Extend a layered model so that the lower part matches a second
        layered model.

        Input:

        base    A LayeredModel which is thicker than the LayeredModel being extended.
        '''
        
        # Find the layers of the base model which need to be added.
        j_extra     = np.where(base.cum_h > self.tot_h)[0] - 1
        h_extra     = base.h  [j_extra]
        
        # If necessary, include the partial layer from base mode.
        if (base.cum_h[j_extra[1]] > self.tot_h):
            
            h_extra[0] = base.cum_h[j_extra[1]] - self.tot_h
        
        # Join the starting model to the new lower layers.
        h_new   = np.concatenate((self.h    [:-1],  h_extra))
        vs_new  = np.concatenate((self.vs   [:-1],  base.vs [j_extra]))
        vp_new  = np.concatenate((self.vp   [:-1],  base.vp [j_extra]))
        rho_new = np.concatenate((self.rho  [:-1],  base.rho[j_extra])) 
        
        # Create the new model.
        # Currently, assume that attenuation parameters can be left
        # as default values.
        joined_model = LayeredModel(h_new,
                                    vs_new,
                                    vp      = vp_new,
                                    rho     = rho_new,
                                    name    = self.name)
        
        # Effectively, self = joined_model
        self.__dict__.update(joined_model.__dict__)

    def write_CPS_mod_file(self, spherical = True):
        '''
        Writes a text file describing the model in the format used by
        the CPS library.

        Input

        spherical   Default: True. Determines whether the model will be treated as spherical (therefore CPS will apply an Earth-flattening transformation) or flat.
        '''
        
        outfmt = '{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}{:>12.3e}{:>12.3e}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}\n'
        
        if spherical:
            flat_or_sph_str = 'SPHERICAL EARTH'
        else:
            flat_or_sph_str = 'FLAT EARTH'
        
        header_lines = ['MODEL.01',
                        '',
                        'ISOTROPIC',
                        'KGS',
                        flat_or_sph_str,
                        '1-D',
                        'CONSTANT VELOCITY',
                        'LINE08',
                        'LINE09',
                        'LINE10',
                        'LINE11',
                        '       H(KM)    VP(KM/S)    VS(KM/S)  RHO(GM/CC)          QP          QS        ETAP        ETAS       FREFP       FREFS']
                        
        
        with open(self.name, 'w') as out_file:
            
            for header_line in header_lines:
                
                out_file.write(header_line + '\n')
                
            for i in range(self.n):
                
                out_file.write(outfmt.format(   self.h[i],
                                                self.vp[i],
                                                self.vs[i],
                                                self.rho[i],
                                                self.qp,
                                                self.qs,
                                                self.etap,
                                                self.etas,
                                                self.frefp,
                                                self.frefs))
                                                
    def write_rftn_mod_file(self):
        '''
        Writes a text file describing the model in the format used by
        the Rftn library (including respknt).
        '''
        
        #'   5.5700'
        outfmt = '  1   '
        outfmt = '{:>3d} {:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}\n'
        
        header_line = '{:>3d} name'.format(self.n)
                        
        with open(self.name, 'w') as out_file:
            
            out_file.write(header_line + '\n')
                
            for i in range(self.n):
                
                out_file.write(outfmt.format(   (i + 1),
                                                self.vp[i],
                                                self.vs[i],
                                                self.rho[i],
                                                self.h[i],
                                                self.qp,
                                                self.qs,
                                                0.0,
                                                0.0,
                                                0.25))
    
    def dispersion(self, periods, silent = True, n_modes = 1, rayleigh = True, spherical = False, eigenfuncs = False, kernels = False, rescale_kernels = True):
        '''
        Calculates dispersion of Rayleigh waves or Love waves for
        a 1-D layered Earth model.
        This function is just a Python wrapper for the sdisp96 code
        from the CPS library (which must be installed).   
             
            Input
        
        (n)         The number of periods.
        periods     (n-by-1) Periods at which to calculate dispersion.
        n_modes     Maximum number of modes to calculate.
        rayleigh    Calculate Rayleigh waves if True, otherwise Love waves.
        silent      Suppress CPS output to command line (True or False).
        spherical   Convert layered model into equivalent values for a spherical model.
        eigenfuncs  If True, also calculate, write and read the eigenfunctions.
        kernels     If True, also calculate, write and read the kernels.
                    Note: sregn96/slegn96 calculate the eigenfuncs and kernels simultaneously, so currently it is not possible to set eigenfuncs = True and kernels = False (or vice versa), but the user can simply ignore any outputs they are not interested in.
        rescale_kernels If True, divide the kernels by the layer thickness so that they give the sensitivity per unit thickness.

            Output
        
        T, c, u     Period, phase speed, group speed. Each is a list of arrays.

        If eigenfuncs == True, the output becomes
        T, c, u, ur, uz, tr, tz, kch, kca, kcb, kcr
        where the additional outputs are the eigenfunctions (ur, uz, tr, tz) and kernels (kch, kca, kcb, kcr)

        The eigenfunction is defined in e.g. Keilis-Borok et al. (1989):
        (First equation in section 1.3.)
        u(x, t) = V(z) exp(i(wt - kx))
        where u is the (vector) displacement and V is the (vector) eigenfunction.
        for a given source and observer, V can be expressed in terms of vertical,
        radial (towards observer) and tangential components. Note that the CPS
        codes uses the letter U instead of V.
        ur    Radial component of eigenfunction 
        uz    Vertical component of eigenfunction.
        tr    Radial component of traction.
        tz    Vertical component of traction.
        Each has shape (n_modes, n_layers).

        The kernels are the sensitivity of c (phase speed) with respect to h, a, b, r (layer thickness, P-wave speed [alpha], S-wave speed [beta] and density [rho]). The kernels have shape (n_modes, n_layers). See also the method LayeredModel.sensitivity_kernels() which uses the srfker function instead and can return more sensitivity kernels.
        '''
        
        # Write the model file in CPS format.
        self.write_CPS_mod_file(spherical = spherical)
        
        # Treat the case of a single period.
        if isinstance(periods, float):

            periods = np.atleast_1d(np.array(periods))
        
        # Write the periods file.
        period_file = 'periods_tmp.txt'
        with open(period_file, 'w') as per_file_ID:

            for period in periods:

                #per_file_ID.write('{:>6.1f}\n'.format(period))
                per_file_ID.write('{:>13.8f}\n'.format(period))   

        # Create sprep96 command.
        # This creates the input file sdisp96.dat.
        # The -R flag specifies Rayleigh waves.
        if rayleigh:
            rl_str = 'R'
        else:
            rl_str = 'L'
        sprep96_command = 'sprep96 -M {} -PARR {} -{} -NMOD {:d}'.format(
                            self.name,
                            period_file,
                            rl_str,
                            n_modes)
                                                                
        # Create sdisp96, sregn96/slegn96 and sdpegn96 commands.
        # sdisp96 computes dispersion curves which are saved in
        #   sdisp96.ray or sdisp96.lov.
        # sregn96/slegn96 compute eigenfunctions which are
        #  saved in slegn96.egn or sregn96.egn.
        # sdpegn96 is used to convert the output to a text file
        #   (it can also create plots). The output is in files
        #   SREGN.TXT and SREGNC.PLT.
        sdisp96_command     = 'sdisp96'
        if rayleigh:                        
            sxegn96_command     = 'sregn96'
        else:
            sxegn96_command     = 'slegn96'
        
        if eigenfuncs:
            sxegn96_eigenfunc_command = '{} -DER'.format(sxegn96_command)
            sdpder96_command = 'sdpder96 -{} -TXT'.format(rl_str)

        sdpegn96_command    = 'sdpegn96 -{} -C -TXT'.format(rl_str)
       
        if eigenfuncs:

            if not rayleigh:

                raise NotImplementedError('Eigenfunctions for Love waves not implemented.')

            if not kernels:

                raise NotImplementedError('Eigenfunction calculation without kernel output is not implemented (try setting kernels = True).')

            commands = [sprep96_command,
                        sdisp96_command,
                        sxegn96_command,
                        sxegn96_eigenfunc_command,
                        sdpder96_command,
                        sdpegn96_command]
        else:

            commands = [sprep96_command,
                        sdisp96_command,
                        sxegn96_command,
                        sdpegn96_command]
        
        # Append >/dev/null to commands to make them silent.
        if silent:
            
            commands = ['{} >/dev/null'.format(x) for x in commands]
                    
        # Call each command from the terminal.
        for command in commands:
            
            subprocess.call(command, shell = True)
            
        # Read the dispersion output file.
        # *To do*: Put this into a separate function, for neatness.
        if rayleigh:
            
            out_file    = 'SREGN.TXT'
            plt_file    = 'SREGNC.PLT'
            sdisp_file  = 'sdisp96.ray'
            egn_file    = 'sregn96.egn'
            n_head      = 3
            
        else:
            
            out_file    = 'SLEGN.TXT'
            plt_file    = 'SLEGNC.PLT'
            sdisp_file  = 'sdisp96.lov'
            egn_file    = 'slegn96.egn'
            n_head      = 4
            
        T = []
        c = []
        u = []
        with open(out_file, 'r') as out_file_ID:
            
            out_file_ID.readline()
            
            eof = False
            for i in range(n_modes):
                
                Ti = []
                ci = []
                ui = []
                
                if eof:
                    
                    break
                

                for i in range(n_head):
                    
                    out_file_ID.readline()
                    
                # Read all periods for this mode.
                while True:
                    
                    line = out_file_ID.readline()
                    
                    if line == False:
                        
                        eof = True
                        break
                        
                    elif len(line.split()) == 0:
                        
                        break
                        
                    else:

                        line = line.split()
                        Ti.append(float(line[1]))
                        ci.append(float(line[3]))
                        ui.append(float(line[4]))
                        
                if len(Ti) > 0:
                    
                    T.append(np.array(Ti))
                    c.append(np.array(ci))
                    u.append(np.array(ui))
                        
        # Tidy up.
        files_to_delete = [ self.name,
                            out_file,
                            plt_file,
                            period_file,
                            'sdisp96.dat',
                            sdisp_file,
                            egn_file]

        for file_to_delete in files_to_delete:

           os.remove(file_to_delete)

        # Read the eigenfunctions (if requested).
        if eigenfuncs:
            
            file_eigenfunc  = 'SRDER.TXT'
            file_calplot    = 'SRDER.PLT'
            file_sxegn96    = 'sregn96.der'
            
            ur, uz, tr, tz, kch, kca, kcb, kcr = self.read_eigenfunc_file(file_eigenfunc, n_modes)
            
            if rescale_kernels:
                
                kca = kca/self.h
                kcb = kcb/self.h
                kcr = kcr/self.h

                kca[:, -1] = 0.0
                kcb[:, -1] = 0.0
                kcr[:, -1] = 0.0

            files_to_delete = [file_eigenfunc, file_calplot, file_sxegn96]
            for file_to_delete in files_to_delete:

                os.remove(file_to_delete)

            return T, c, u, ur, uz, tr, tz, kch, kca, kcb, kcr

        else:

            return T, c, u
    
    def read_eigenfunc_file(self, file_eigenfunc, n_modes):
        '''
        A helper method for LayeredModel.dispersion. It reads the eigenfunction file output by sdisp96.

        Input

        file_eigenfunc  The sdisp96 output file (usually called SRDER.TXT).
        n_modes         The number of modes in the output.

        Output:

        ur, uz, tr, tz, kch, kca, kcb, kcr
                        See the documentation for LayeredModel.dispersion().
        '''

        with open(file_eigenfunc, 'r') as in_id:

            # Skipblank line and two  header lines.
            n_head = 3
            for i in range(n_head):

                in_id.readline()

            # Skip model.
            for i in range(self.n):

                in_id.readline()

            ## Skip two blank lines.
            #for i in range(2):
            #
            #    in_id.readline()

            # Prepare the output arrays.
            ur = np.zeros((n_modes, self.n))
            tr = np.zeros((n_modes, self.n))
            uz = np.zeros((n_modes, self.n))
            tz = np.zeros((n_modes, self.n))

            kch = np.zeros((n_modes, self.n))
            kca = np.zeros((n_modes, self.n))
            kcb = np.zeros((n_modes, self.n))
            kcr = np.zeros((n_modes, self.n))

            # Read output for each mode.
            for j in range(n_modes):
                
                # Skip two blank lines.
                for i in range(2):

                    in_id.readline()

                # Skip header line.
                in_id.readline()

                # Skip lines with parameters T, C, U, AR, GAMMA, ZREF
                # (Period, phase speed, group speed, ?, ?, ?).
                for i in range(2):

                    in_id.readline()

                # Skip header lline.
                in_id.readline()

                # Read the eigenfunction for each layer.
                # The eigenfunction is defined in e.g. Keilis-Borok et al. (1989):
                # (First equation in section 1.3.)
                # u(x, t) = V(z) exp(i(wt - kx))
                # where u is the (vector) displacement and V is the (vector) eigenfunction.
                # for a given source and observer, V can be expressed in terms of vertical,
                # radial (towards observer) and tangential components. Note that the CPS
                # codes uses the letter U instead of V.
                # ur    Radial component of eigenfunction 
                # uz    Vertical component of eigenfunction.
                # tr    Radial component of traction.
                # tz    Vertical component of traction.
                # 
                # kch   Partial derivative of phase speed with respect to layer thickness (?)
                # kca   Partial derivative of phase speed with respect to P-wave speed.
                # kcb   Partial derivative of phase speed with respect to S-wave speed.
                # kcr   Partial derivative of phase speed with respect to density.
                #
                for i in range(self.n):

                    line = in_id.readline().split()
                    ur[j, i] = line[1]
                    tr[j, i] = line[2]
                    uz[j, i] = line[3]
                    tz[j, i] = line[4]

                    kch[j, i] = line[5]
                    kca[j, i] = line[6]
                    kcb[j, i] = line[7]
                    kcr[j, i] = line[8]

        return ur, uz, tr, tz, kch, kca, kcb, kcr

    def rfn(self, phase = 'P', rayp = 0.05, alpha = 1.0, dt = 1.0, n = 512, delay = 5.0, silent = True):
        '''
        Calculates receiver function using the trftn96 command.
        
        Input:
        
        phase   Incident phase, 'P' or 'S'.
        rayp    Ray parameter of incident phase, s/km.
        dt      Sampling interval, s.
        n       Number of samples.
        alpha   Width of receiver function filter.
                The filter is given by H(f) = exp((-pi*f/alpha)**2.0)
        delay   The time before the start of the receiver function, s.
        
        Output:
        
        resp_trace  ObsPy trace object of the receiver function.
        '''
        
        # Write the model file in CPS format.
        self.write_CPS_mod_file()
        
        cmd = ( 'trftn96 -{} -RAYP {:>8.5f} '
                '-ALP {:>8.5f} -DT {:>8.5f} '
                '-NSAMP {:>4d} -D {:>8.5f} '
                '-M {}').format(
                phase, rayp, alpha, dt, n, delay, self.name)
                
        # Append >/dev/null to commands to make them silent.
        if silent:
            
            cmd = '{} >/dev/null'.format(cmd)
        
        else:
            
            print(cmd)
            
        # Call each command from the terminal.
        subprocess.call(cmd, shell = True)
        
        # Read the output file.
        out_file    = 'trftn96.sac'
        resp_trace  = obspy.core.stream.read(out_file)[0]
        
        # Tidy up.
        os.remove(self.name)
        os.remove(out_file)
        
        return resp_trace
    
    def earth_response(self, phase = 'p', component = 'z', ray_param = 0.05, dt = 0.01, Dt = 45.0, src_time_file = None, clip = False, silent = False): 
        '''
        A wrapper for respknt which calculates response of a layered model
        to an incident pulse.
    
            Input:
    
        phase       The type of incident wave: 'p' or 's'.
        component   The component of the seismogram, radial ('r'),
                        vertical ('z'), or transverse ('t').
        ray_param   The incident ray parameter (s/km).
        dt          The sampling interval of the synthetic seismogram.
        Dt          The duration of the synthetic seismogram.
        src_tm_file The path to a source-time function. If provided, this will be
                        convolved with the earth response to create a synthetic seismogram.
        clip        If True, the result will be clipped between 10.0 and 25.0 seconds.
                    *To do*: Allow clipping between any specified time range.
        '''
        
        # Choose identifying strings.
        if phase == 'p':
            
            phase_str = '1'
            possible_components = ['r', 'z']
            
        elif phase == 's':
            
            phase_str = '2'
            possible_components = ['r', 'z', 't']
        
        self.write_rftn_mod_file()
        
        # Call respknt to generate the Earth response function.
        respknt_bin = 'respknt'
        respknt_cmd_fmt = './resp_wrapper.bash {} {} {} {} {} {}'
        if silent:
            respknt_cmd_fmt = respknt_cmd_fmt + ' >/dev/null'

        respknt_cmd = respknt_cmd_fmt.format(
                            respknt_bin,
                            self.name,
                            phase_str,
                            str(dt),
                            str(Dt),
                            str(ray_param))
        
        print(respknt_cmd)
        subprocess.call(respknt_cmd, shell = True)
        
        # Read the Earth response as a NumPy array.
        resp_stream_name    = '{}_sp.{}'.format(self.name, component)
        resp_trace          = obspy.core.stream.read(resp_stream_name)[0]
        
        # Trim the data between 10.0 and 25.0 seconds.
        if clip:
             
            i_max       = np.argmax(np.abs(resp_trace.data))
            n_before    = int(round(10.0/resp_trace.stats.delta))
            n_after     = int(round(25.0/resp_trace.stats.delta))
            i_before    = i_max - n_before
            i_after     = i_max + n_after
            new_rate    = resp_trace.stats.sampling_rate
            resp_trace  = obspy.core.trace.Trace(
                            resp_trace.data[i_before : i_after])
            resp_trace.stats.sampling_rate = new_rate
        
        # If a source time function is provide, convolve it with the earth response to get a synthetic seismogram.
        if src_time_file is not None:
            
            # Load the source time function.
            src_time_trace          = obspy.core.stream.read(src_time_file)[0]
            
            resp_trace = convolve_traces(resp_trace, src_time_trace)
            
        t                   = resp_trace.times()
        resp                = resp_trace.data

        # Tidy up.
        files_to_delete = ['{}_sp.{}'.format(self.name, x)
                            for x in possible_components]
        files_to_delete.append(self.name)

        for file_to_delete in files_to_delete:

            os.remove(file_to_delete)
            
        return t, resp, resp_trace

    def resample(self, dh = None):
        '''
        Create a new layered model with a finer sampling interval.

        Currently there are two options:
        dh = None   will automatically choose the sampling interval based on a table below.
        dh = const  will set the sampling interval to const.
        In either case, the example sampling interval is not guaranteed because existing discontinuities in the model will be respected.

        Warning:
        Currently, only parameters vp, vs and rho are carried over to the new LayeredModel.
        
        Output

        A resampled LayeredModel.
        '''
        
        # Initialise new variables as lists.
        h_new   = []
        vs_new  = []
        vp_new  = []
        rho_new = []
        
        dh_table = {(  0.0,  10.0):  0.5,
                    ( 10.0,  20.0):  1.0,
                    ( 20.0,  40.0):  2.0,
                    ( 40.0,  80.0):  4.0,
                    ( 80.0, 160.0):  8.0,
                    (160.0, 320.0): 16.0,
                    (320.0, 640.0): 32.0,
                    (640.0, 999.9): 64.0}
        
        cum_depth = 0.0
        # Loop over each layer.
        for i in range(self.n):
            
            # Find the appropriate layer thickness for this depth.
            if i == (self.n - 1):
                
                mid_depth = 999.0
                
            else:
                
                mid_depth = cum_depth + self.h[i]/2.0
                
            if dh is None:
                
                for rnge in dh_table:
                
                    if (rnge[0] <= mid_depth < rnge[1]):
                    
                        dh = dh_table[rnge]
            
            # Find how many layers this layer should
            # be divided into to approximately match
            # the requested spacing.
            n_i = int(round(self.h[i]/dh))
            
            # There must be at least one layer.
            if n_i == 0:
                
                n_i = 1
            
            # Find the actual thickness of each new
            # layer.
            dh_i = self.h[i]/float(n_i)
            
            # Update the arrays.
            for j in range(n_i):
                
                h_new.  append(dh_i)
                vs_new. append(self.vs [i])
                vp_new. append(self.vp [i])
                rho_new.append(self.rho[i])
                
            cum_depth = cum_depth + self.h[i]
    
        # Cast as NumPy arrays.
        h_new   = np.array(h_new)
        vs_new  = np.array(vs_new)
        vp_new  = np.array(vp_new)
        rho_new = np.array(rho_new)
        
        # Create the new layered model.
        resampled_model = LayeredModel( h_new,
                                        vs_new,
                                        vp      = vp_new,
                                        rho     = rho_new)
                                        
        return resampled_model
            
    def sensitivity(self, periods, mode = 0, dv = 'c', iv = 'b', spherical = False):
        '''
        Wrapper for srfker96 sensitivity kernel calculator.
        Note: could be elastic or anelastic, but currently only
        coded for elastic.
        
        Input
        
        periods A NumPy array of the periods at which the sensitivity should be calculated.
        mode    Overtone number; 0 for fundamental mode, 1 for first overtone,
                etc.
        dv  Dependent variable:
                'c' Phase speed.
                'u' Group speed.
                'g' ?
        iv  Independent variable:
                'a'     P-wave speed.
                'b'     S-wave speed.
                'h'     Layer thickness.
                'qa'    P-wave Q factor.
                'qb'    S-wave Q factor.

        Output

        ker     (n_periods, n_layers) The sensitivity of the dependent variable on the independent variable within each layer of the model. The sensitivity is normalised by the thickness of the layer, so that the bottom half-space (which has infinite thickness) has 0 sensitivity.

        Note:

        See LayeredModel.dispersion(), which can also calculate some sensitivity kernels, including sensitivity to density.
        '''
        
        dv_iv_str = '{}{}'.format(dv, iv)
        dv_iv_index_dict = {    'ca'  :  2,
                                'cb'  :  3,
                                'ua'  :  4,
                                'ub'  :  5,
                                'ch'  :  6,
                                'uh'  :  7,
                                'cqa' :  8,
                                'cqb' :  9,
                                'uqa' : 10,
                                'uqb' : 11,
                                'gqa' : 12,
                                'gqb' : 13    }
        
        index = dv_iv_index_dict[dv_iv_str]
        
        # Prepare files for srfker96. ---------------------------------
        
        kernel_file = 'srfker96.txt'
        
        # Prepare model file.
        self.write_CPS_mod_file(spherical = spherical)

        # Prepare control file.
        disp_file       = 'tdisp.d'
        sobs_file_lines = [ '  4.99999989E-03  4.99999989E-03   0.0000000      4.99999989E-03   0.0000000 ',
                            '1    1    1    1    1    1    1    0    1    0',
                            self.name,
                            disp_file]
        
        sobs_file = 'sobs.d'
        with open(sobs_file, 'w') as sobs_file_ID:
            
            for line in sobs_file_lines:
                
                sobs_file_ID.write(line + '\n')
                
        # Prepare blank dispersion file.
        disp_file_fmt   = 'SURF96 R U X {:>3d}    {:>8.4f}     4.0000     0.1000\n'
        with open(disp_file, 'w') as disp_file_ID:
            
            for period in periods:
                
                disp_file_ID.write(disp_file_fmt.format(mode, period))
                
        # Run srfker96. -----------------------------------------------
        
        srfker_commands = [ 'surf96 39',
                            'surf96 1',
                            'srfker96 > ' + kernel_file,
                            'surf96 39']
                            
        for command in srfker_commands:
            
            print(command)
            subprocess.call(command, shell = True)
            
        # Read the output. --------------------------------------------
        
        n_periods           = len(periods)
        sens                = np.zeros((n_periods, self.n))
        n_head_kern_file    = 3
        
        
        with open(kernel_file, 'r') as kern_file_ID:
            
            # Read elastic sensitivity data.
            for i in range(n_periods):
                
                # Skip header lines.
                for s in range(n_head_kern_file):
                    
                    kern_file_ID.readline()
                    
                # Read data.
                for j in range(self.n):
                    
                    line = kern_file_ID.readline().split()
                    sens[i, j] = float(line[index])
                    
                # Skip anelastic sensitivity data.
                
                # Skip header lines.
                for s in range(n_head_kern_file):
                    
                    kern_file_ID.readline()
                    
                # Skip data lines.
                for j in range(self.n):
                    
                    kern_file_ID.readline()
        
        # Normalise for layer thickness.
        h       = self.h.copy()
        h[-1]   = 6371.0 - np.sum(h)
        sens = sens/h
        
        # Tidy up.
        files_to_delete = [ self.name,
                            kernel_file,
                            sobs_file,
                            'tdisp.d']

        for file_to_delete in files_to_delete:

           os.remove(file_to_delete)
 
        return sens 
    
def convolve_traces(trace_1, trace_2):
    '''
    Convolves two ObsPy trace objects.
    First, if one of the traces has a lower sampling rate,
    it will be up-sampled to match the other trace. After
    the convolution, the result will be down-sampled to
    the lower sampling rate.

    Input
    trace_1, trace_2    ObsPy traces.

    Output
    convolution_trace   The result of the convolution of the two traces.
    '''
    
    if (trace_1.stats.sampling_rate == trace_2.stats.sampling_rate):
        
        sampling_rates_are_equal    = True
        
        trace_A                     = trace_1.copy()
        trace_B                     = trace_2.copy()
        
    else:
        
        sampling_rates_are_equal    = False
        
        if (trace_1.stats.sampling_rate < trace_2.stats.sampling_rate):
        
            trace_with_lower_sampling_rate  = trace_1.copy()
            trace_with_higher_sampling_rate = trace_2.copy()
            final_sampling_rate             = trace_1.stats.sampling_rate
        
        else:
        
            trace_with_lower_sampling_rate  = trace_2.copy()
            trace_with_higher_sampling_rate = trace_1.copy()
            final_sampling_rate             = trace_2.stats.sampling_rate
            
        trace_A = \
            trace_with_lower_sampling_rate.resample(
                trace_with_higher_sampling_rate.stats.sampling_rate)
        trace_B = trace_with_higher_sampling_rate
            
    convolution = np.convolve(
                    trace_A.data,
                    trace_B.data)
                    
    convolution_trace = obspy.core.trace.Trace(
                                    data    = convolution,
                                    header  = { 'sampling_rate' :
                                                trace_A.stats.sampling_rate})
    
    if not sampling_rates_are_equal:
  
        convolution_trace.resample(final_sampling_rate)
        
    return convolution_trace
