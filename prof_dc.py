###     Module: galaxy.py of SIGAME                   ###

# Import other SIGAME modules
import aux as aux
import global_results as glo

# Import other modules
import numpy as np
import pandas as pd
import pdb as pdb
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d,interp2d
import matplotlib.cm as cm
import multiprocessing as mp
import matplotlib.pyplot as plt

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

params                      =   np.load('temp_params.npy').item()
params['z']                 =   'z0'
params['d_data']            =   params['z']+'_data_files/'
params['d_cloudy_models']   =   '/Volumes/Seagate_1TB/cloudy_models/'
for key,val in params.items(): exec(key + '=val')

def run():

    gal_index               =   9
    GR                      =   glo.global_results()

    gal_obj                 =   gal.galaxy(gal_index=gal_index)

    print('\n--- Creating datacubes ---')      
    dat_obj                 =   gal.datacube_of_galaxy(gal_index=gal_index)
    dat_obj.add_dc(ISM_phase='DNG')


#===========================================================================
""" Main galaxy data classes """
#---------------------------------------------------------------------------

class galaxy:
    """
    An object referring to one particular galaxy, containing specific properties of
    that galaxy as attributes.
    Analysis on individual galaxies use the 'galaxy' object while all analysis
    of the entire sample uses the 'galaxy_sample' object, which collects 'galaxy' objects.

    -------------------------
    Attributes: name, type, definition
    -------------------------
    index           int     galaxy index from redshift sample of galaxies
    name            str     galaxy name
    radius          float   galaxy radius
    zred            float   galaxy redshift
    lum_dist        float   luminosity distance at redshift
    N_radial_bins   int     number of bins in radius for plots
    objects         list    string names of objects to add to galaxy

    -------------------------
    Optional keyword argument(s) and default value(s) to __init__(**kwargs)
    -------------------------
    index           0
    N_radial_bins   30
    classification  'spherical'

    -------------------------
    Methods
    -------------------------
    get_radial_axis()
    check_classification()
    add_attr( attr_name , **kwargs )
    check_attr( attr_name , **kwargs )
    """

    def __init__(self,**kwargs):

        # get global results
        import global_results as glo
        GR                  =   glo.global_results()

        # handle default values and kwargs
        args                =   dict(gal_index=0, N_radial_bins=30, classification='spherical', silent=False)
        args                =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')
        if not silent: print("constructing galaxy...")

        # grab some info from global results for this galaxy
        self.radius         =   GR.R_gal[gal_index]
        self.name           =   GR.galnames[gal_index]
        self.zred           =   GR.zreds[gal_index]
        self.SFR            =   GR.SFR[gal_index]
        self.SFRsd          =   GR.SFRsd[gal_index]
        self.UV_str         =   aux.get_UV_str(z1,self.SFRsd)
        self.lum_dist       =   GR.lum_dist[gal_index]
        self.ang_dist_kpc   =   self.lum_dist*1000./(1+self.zred)**2

        # add attributes from args
        for key in args: setattr(self,key,args[key])

        # add objects
        objects             =   ['datacube','particle_data']
        for ob in objects: self.add_attr(ob,**args)

        if not silent: print("galaxy %s constructed.\n" % self.name)

    def get_radial_axis(self):
        """ returns 1D radius array for galaxy """
        radial_axis =   np.linspace(0,self.radius,self.N_radial_bins+1)
        dr          =   radial_axis[1]-radial_axis[0]
        return radial_axis + dr/2.

    def check_classification(self):
        """ checks if galaxy classification is correct (all galaxies are
        initialized with a 'spherical' classification.) """
        self.particle_data.classify_galaxy(self)

    def add_attr(self,attr_name,**kwargs):
        """ creates desired attribute and adds it to galaxy. """
        if hasattr(self, attr_name):
            if not self.silent: print("%s already has attribute %s" % (self.name,attr_name))
        else:
            if not self.silent: print("Adding %s attribute to %s ..." % (attr_name,self.name) )
            exec('ob = %s(self,**kwargs)' % attr_name )
            setattr(self,attr_name,ob)

    def check_for_attr(self,attr_name,**kwargs):
        """ checks if galaxy has a specific attribute, if not then adds
        it. """
        if hasattr( self , attr_name ):
            print("%s has %s." % (self.name,attr_name) )
        else:
            print("%s doesn't have %s. Adding to galaxy ..." % (self.name,attr_name) )
            self.add_attr(attr_name,**kwargs)

class particle_data:

    def __init__(self,gal_ob,**kwargs):

        # handle default values and kwargs
        args    =   dict(units='kpc',silent=False)
        args    =   aux.update_dictionary(args,kwargs)
        for key in args: setattr(self,key,args[key])
        for key in args: exec(key + '=args[key]')
        if not self.silent: print("constructing particle_data...")

        # add labels for spatial dimentions
        dim     =   ['x','y','z']
        for x in dim: setattr(self,x + 'label','%s [%s]' % (x,units))

        # add galaxy
        self.gal_ob =   gal_ob

        # add dictionaries of names for all sim types and ISM phases
        self.add_names()

        if not self.silent: print("particle_data constructed for %s.\n" % gal_ob.name)

    #---------------------------------------------------------------------------

    def __get_sim_name(self,sim_type):
        return d_data + 'particle_data/sim_data/z%.2f_%s_sim.%s' % (self.gal_ob.zred, self.gal_ob.name, sim_type)

    def __get_phase_name(self,ISM_phase):
        return d_data + 'particle_data/ISM_data/z%.2f_%s_%s.h5' % (self.gal_ob.zred, self.gal_ob.name, ISM_phase)

    def add_names(self):

        # make empty containers to collect names
        sim_names   =   {}
        ISM_names   =   {}

        # add names to containers
        for sim_type in sim_types: sim_names[sim_type] = self.__get_sim_name(sim_type)
        for ISM_phase in ISM_phases: ISM_names[ISM_phase] = self.__get_phase_name(ISM_phase)

        # add attributes
        self.sim_names  =   sim_names
        self.ISM_names  =   ISM_names

    def get_raw_data(self,**kwargs):

        # handle default args and kwargs
        args    =   dict(data='sim')
        args    =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        # choose which particle_data to load
        if data == 'sim':
            names   =   self.sim_names
        if data == 'ISM':
            names   =   self.ISM_names

        # make empty container for pandas data
        collection  =   {}

        # collect all data into container
        for key,name in names.items():
        	data 			=	aux.load_temp_file(name,'data')
        	# data1           =   pd.read_pickle(name)
        	# data1['type']   =   key

        	collection[key] 	=	data #.append(data1)

        return collection

    #---------------------------------------------------------------------------

    def __get_rotated_matrix(self,vectors):

        # rotation angle (radians)
        phi =   np.deg2rad( float(inc_dc) )

        # rotation matrix
        rotmatrix   	=   aux.rotmatrix( phi, axis='y' )
        return vectors.dot(rotmatrix)

    def __get_magnitudes(self,vectors):
        return np.sqrt( (vectors**2).sum(axis=1) )

    def get_data(self,**kwargs):

        # get raw sim data
        data  =   self.get_raw_data(**kwargs)[kwargs['data']]

        # get empty arrays for rotated positions and velocities
        size    =   data.shape[0]
        X       =   np.zeros( (size,3) )
        V       =   np.zeros_like(X)

        # populate X and V with unrotated values
        for i,x in enumerate(['x','y','z']):
            X[:,i]  =   data[x].values
            V[:,i]  =   data['v'+x].values

        # rotate X and Y
        X   =   self.__get_rotated_matrix(X)
        V   =   self.__get_rotated_matrix(V)

        # update data with rotated values
        for i,x in enumerate(['x','y','z']):
            data[x]     =   X[:,i]
            data['v'+x] =   V[:,i]

        # add radius and speed columns
        data['radius']  =   self.__get_magnitudes(X)
        data['speed']   =   self.__get_magnitudes(V)

        return data

    #---------------------------------------------------------------------------

    def gas_quiver_plot(self,**kwargs):

        # get default plot params
        plot.set_mpl_params()

        # make figure
        fig =   plt.figure(figsize=(8,6))
        ax  =   fig.add_subplot(111)

        # get gas data
        sim_data    =   self.get_data() # default is already data='sim'
        sim_gas     =   sim_data[ sim_data.type == 'gas' ]

        # make plot object and add it to ax
        results     =   dict(plot_style='quiver', data=sim_gas)
        results     =   aux.update_dictionary(results, kwargs)
        ob          =   plot_it(results)
        ob.add_to_axis(ax,**kwargs)

    def positions_plot(self,**kwargs):

        # handle default values and kwargs
        args    =   dict(dm=False, gas=True, star=True, GMC=False, dif=False)
        args    =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        # pull particle data
        if any([dm, gas, star]): sim_data = self.get_data()
        if any([GMC, dif]): ISM_data = self.get_data(data='ISM')

        # run default plot params
        plot.set_mpl_params()

        # make figure
        fig =   plt.figure()
        ax  =   fig.add_subplot(111,projection='3d')

        # choose different plotting parameters for different sim types
        for sim_type in sim_types:
            if sim_type == 'dm':
                if not dm: continue
                alpha   =   .1
                size    =   .5
                data    =   sim_data[ sim_data.type=='dm' ]
            elif sim_type == 'star':
                if not star: continue
                alpha   =   1
                size    =   20
                data    =   sim_data[ sim_data.type=='star' ]
            else:
                if not gas: continue
                alpha   =   .5
                size    =   1
                data    =   sim_data[ sim_data.type=='gas' ]

            # gather sim results
            sim_results =   dict(plot_style='scatter', dim3=True, data=data, alpha=alpha, size=size, color=sim_colors[sim_type], label=sim_labels[sim_type])
            sim_results =   aux.update_dictionary(sim_results,kwargs)

            # turn sim results into plot_it instance
            sim_ob  =   plot_it(sim_results)

            # add sim_ob onto ax
            sim_ob.add_to_axis(ax,**kwargs)

        # choose different plotting parameters for different ISM phases
        for phase in ISM_phases:
            if phase == 'GMC':
                if not GMC: continue
                alpha   =   .5
                size    =   1
                data    =   ISM_data[ ISM_data.type=='GMC' ]
            elif phase == 'dif':
                if not dif: continue
                alpha   =   .4
                size    =   1
                data    =   ISM_data[ ISM_data.type=='dif' ]

            # gather ISM results
            ISM_results =   dict(plot_style='scatter', dim3=True, data=data, alpha=alpha, size=size, color=sim_colors[sim_type], label=sim_labels[sim_type])
            ISM_results =   aux.update_dictionary(ISM_results, kwargs)

            # turn sim results into plot_it instance
            ISM_ob  =   plot_it(sim_results)

            # add sim_ob onto ax
            ISM_ob.add_to_axis(ax,**kwargs)

        pad =   20
        ax.set_xlabel(self.xlabel, labelpad=pad)
        ax.set_ylabel(self.xlabel, labelpad=pad)
        ax.set_zlabel(self.xlabel, labelpad=pad)
        ax.set_aspect(1)
        ax.legend()

        plt.show()

    def get_map(self,**kwargs):

        # handle default values and kwargs
        args    =   dict(choose_map='GMC', grid_length=100)
        args    =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        # pick data to load
        if choose_map == 'GMC' or choose_map == 'dif':
            data    =   self.get_raw_data(data='ISM')
        else:
            data    =   self.get_raw_data()
        # filter data for selected sim type or ISM phase
        data    =   data[choose_map]

        # get positional grid
        X       =   np.linspace(-x_max_pc, x_max_pc, grid_length) / 1000
        dx      =   X[-1] - X[-2]
        grid    =   np.matmul( X[:,None] , X[None,:] )

        # create empty map
        mass_map    =   np.zeros_like(grid)

        for j,y in enumerate(X[:-1]):
            for k,x in enumerate(X[:-1]):

                # create mask that selects particles in a pixel at position [x,y]
                gas1    =   data[ (data.y >= X[j]) & (data.y < X[j+1]) & (data.x >= X[k]) & (data.x < X[k+1]) ]

                # get total mass of pixel
                mass_map[j,k]   =   np.sum( gas1['m'].values )

        # collect results
        results =   dict(X=X, Y=X, Z=mass_map, title='Total Mass', plot_style='map')
        results =   aux.update_dictionary(results,kwargs)

        # return plot_it instance
        return plot_it(results)

    #---------------------------------------------------------------------------

    def classify_galaxy(self,**kwargs):

        # show positional plot
        self.positions_plot(**kwargs)

        # add classification attribute to galaxy object
        spherical   =   input("Is %s spherical/elliptical? ['y'/'n']" % self.galaxy.name)
        if spherical == 'y':
            setattr(self.galaxy, 'classification', 'spherical')
            return
        disk    =   input("Is %s a disk? ['y'/'n']" % self.galaxy.name)
        if disk == 'y':
            setattr(self.galaxy, 'classification', 'disk')
            return
        else:
            setattr(self.galaxy, 'classification', 'unknown')
            print("set %s classification to 'unknown'." % self.galaxy.name)

class datacube:
    """ holds all the methods for extracting datacubes and generating maps and has attributes for essential spacial data and plotting parameters

    __init__( gal_ob , **kwargs )
    --------------------------------------------
    gal_ob:  galaxy class instanace
    kwargs:
    key             type    default value
    -------------------------------------
    type:           str     'Total'
    label:          str     'Summed Galaxy Profile'
    color:          str     'k'
    xlabel:         str     'km/s'
    ylabel:         str     'Jy'

    attributes:
    -----------
    xlabel:         str     velocity label for plot
    ylabel:         str     flux label for plot
    type:           str     description from kwargs
    label:          str     legend label from kwargs
    color:          str     plotting style from kwargs

    methods:
    --------
    get_cube_phase( phase , **kwargs )
    get_dc_summed( **kwargs )
    get_map( **kwargs )
    preview( **kwargs )
    """

    def __init__(self,gal_ob,**kwargs):

        # handle default values and kwargs
        args                =   dict( type='Total' , label='Summed Galaxy Profile' , color='k', xlabel='x [kpc]', ylabel='y [kpc]', vlabel='km/s', flabel='Jy',silent=False)
        args                =   aux.update_dictionary(args,kwargs)
        for key in args: setattr(self,key,args[key])
        for key in args: setattr(self,key,args[key])

        if not self.silent: print("constructing datacube...")

        # add attributes
        self.gal_obj        =   gal_ob
        self.add_shape()

        if not self.silent: print("datacube constructed for %s.\n" % gal_ob.name)

    def add_shape(self):
        x   =   int(2*x_max_pc/x_res_pc)
        v   =   int(2*v_max/v_res)
        self.shape  =   (v,x,x)

    #---------------------------------------------------------------------------

    def __get_dc_file_location(self,**kwargs):

        # handle default values and kwargs
        args        =   dict(ISM_dc_phase='GMC',target='L_CII', test=False)
        args        =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        # construct file name
        filename 	= 	d_data+'datacubes/%s_%s_%s_i%s_G%s.h5' % (z1, target, ISM_dc_phase, inc_dc, self.gal_obj.gal_index+1)
        if test: filename 	=	d_data+'datacubes/%s_%s_%s_i%s_G%s.h5' % (z1, target, ISM_dc_phase, inc_dc, self.gal_obj.gal_index+1)

        return  filename

    # includes notes and minor changes
    def get_dc(self,**kwargs):

        filename    =   self.__get_dc_file_location(**kwargs)

        dc 			=	0
        try:
            data        =   aux.load_temp_file(filename,'data')
            dc          =   {'data':data['data'][0]}#,'x_axis':data.metadata['x_axis'],'v_axis':data.metadata['v_axis']}
            for key in data.metadata:
            	dc[key]     =   data.metadata[key]
            print('Loaded %s datacube file at %s' % (kwargs['ISM_dc_phase'],filename))
        except:
        	print('Did not find %s datacube file at %s' % (kwargs['ISM_dc_phase'],filename))
            # raise IOError, 'Did not find %s datacube file at %s' % (ISM_dc_phase,filename)

        return dc

    # needs some more thought and perhaps a different implementation
    def get_dc_summed(self,**kwargs):

        # make empty summed array
        summed  =   np.zeros( self.shape )
        # iterate over list of datacubes
        try:
            for i,ISM_dc_phase in enumerate(ISM_dc_phases):
                cube_i          =   self.get_dc(ISM_dc_phase=ISM_dc_phase,**kwargs)
                summed          +=  cube_i['data']
                # consider differenet implementation. can we get x_axis and v_axis from methods? Do regular datacubes still have x_axis and v_axis?
                if i == 0:
                    x_axis  =   cube_i['x_axis']
                    v_axis  =   cube_i['v_axis']
                    v_res 	=   v_axis[1]-v_axis[0]
            result = dict(x_axis=x_axis, v_axis=v_axis, v_res=v_res, datacube=summed)
        except:
            result = dict(datacube=0)

        # is there a way we can return the summed datacube in the same form as the other datacubes so they can be accessed in the exact same way?
        return result

    # includes notes and minor changes
    def get_line_lum(self,**kwargs):

        # handle default values and kwargs
        args            =   dict(ISM_phases=ISM_phases)
        # args            =   dict(target='L_CII',ISM_phases=ISM_phases)
        args            =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        tot_line_lum    =   0
        for i,ISM_phase in enumerate(ISM_phases):
            cube_i          =   self.get_dc(ISM_phase,**kwargs)
            # cube_i          =   self.__get_dc(ISM_phase=ISM_phase,target=target)
            tot_line_lum    +=  cube_i['dc_sum']

        return tot_line_lum

    def get_v_axis(self):
        return np.arange(-v_max,v_max,v_res)

    def get_x_axis_kpc(self):
        return (np.arange(-x_max_pc, x_max_pc, x_res_pc) + x_res_pc/2.)/1000

    def get_kpc_per_arcsec(self):
        return np.arctan(1./60/60./360.*2.*np.pi)*gal_ob.ang_dist_kpc

    def get_x_axis_arcsec(self):
        return self.x_axis_kpc/self.kpc_per_arcsec

    #---------------------------------------------------------------------------

    def get_map(self,**kwargs):

        # handle default values and kwargs
        args    =   dict(target='L_CII', ISM_phase='GMC', method='velocity')
        args    =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        # variable for title selection
        if target == 'm':
            target_str  =   'Mass'
        else:
            target_str  =   'Luminosity'

        # pull datacube
        if ISM_phase in ISM_phases:
            cube    =   self.get_dc(ISM_phase=ISM_phase,target=target)
        else:
            cube    =   self.get_dc_summed(target=target,**kwargs)

        # set up X-Y axies
        X   =   np.linspace(-x_max_pc, x_max_pc, cube['x_axis'].shape[0]) / 1000
        X,Y =   np.meshgrid(X,X)

        # get map values
        if method == 'summed':

            # get the datacube summed over velocity axis
            Z       =   cube['data'].sum(axis=0)
            Z       =   np.log10(Z)

            # get plot title
            title   =   "Summed %s %s" % (ISM_phase,target_str)

            # raise the lower limit higher than what it is set to in make_map
            args['vmin']    =   0

        elif method == 'dispersion':

            # make a copy of the datacube for manipulation
            dc      =   np.array(cube['data'], copy=True)

            # multiply spacial values by their coresponding velocity values
            for i,v in enumerate(cube['v_axis']):
                dc[i,:,:]   *=  v

            # divide result by the original datacube summed over velocity axis
            dc  /=  cube['data'].sum(axis=0)

            # get the standard deviation of the weighted datacube along velocity axis
            Z       =   np.std(dc,axis=0)

            # get plot title
            title   =   "%s %s-Weighted %s" % (ISM_phase,target_str,method.title())

        else: # default

            # get the weighted values
            Z       =   -1. * np.tensordot(cube['v_axis'],cube['data'],1) / cube['data'].sum(axis=0)

            # get plot title
            title   =   "%s %s-Weighted %s" % (ISM_phase,target_str,method.title())

        # collect results
        results =   kwargs
        results.update(dict(X=X, Y=Y, Z=Z.T, title=title))

        # return generic object of results
        return make_map(results)

    def preview(self,**kwargs):

        # handle default values and kwargs
        args    =   dict(phase='GMC', lw_velocity=True, lw_dispersion=True, l_summed=True, mw_velocity=True, mw_dispersion=True, m_summed=True)
        args    =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        # make empty container for map objects
        maps    =   {}

        if lw_velocity:     maps['lw_vel'] = self.get_map(xlabel='', ylabel='', phase=phase, cmap=cm.seismic, **kwargs)

        if lw_dispersion:   maps['lw_dis'] = self.get_map(xlabel='', ylabel='', phase=phase, method='dispersion', cmap=cm.viridis, **kwargs)

        if l_summed:        maps['l_sum']  = self.get_map(label='log(Z)', xlabel='', ylabel='', phase=phase, method='summed', **kwargs)

        if mw_velocity:     maps['mw_vel'] = self.get_map(xlabel='', ylabel='', line='m', phase=phase, cmap=cm.seismic, **kwargs)

        if mw_dispersion:   maps['mw_dis'] = self.get_map(xlabel='', ylabel='', line='m', phase=phase, method='dispersion', cmap=cm.viridis, **kwargs)

        if m_summed:        maps['m_sum']  = self.get_map(label='log(Z)', xlabel='', ylabel='', line='m', phase=phase, method='summed', **kwargs)

        # convert maps to Series
        maps    =   pd.Series(maps)

        # find convenient number of rows and columns
        size    =   maps.size
        n_rows  =   int( divmod( size, np.sqrt(size) )[0] )
        n_cols  =   int( divmod( size, n_rows )[0] )

        # organize panel display
        panels  =   np.empty(n_rows*n_cols, dtype='O')
        for i,m in enumerate(maps):
            panels[i]   =   m

        # set up default plot parameters
        plot.set_mpl_params()

        # make figure
        if n_rows > 3:
            figsize =   (15,15)
        else:
            figsize =   (15,5)
        fig,ax  =   plt.subplots(n_rows,n_cols,figsize=figsize, sharex=True, sharey=True)
        if n_rows == 1:
            panels  =   panels[None,:]
            ax      =   ax[None,:]
        panels  =   panels.reshape([n_rows,n_cols])

        for i in range(n_rows):
            for j in range(n_cols):
                ob  =   panels[i,j]
                if ob == None: continue
                ob.add_to_axis(ax[i,j],**kwargs)

        fig.text(0.5, 0.04, self.xlabel, ha='center')
        fig.text(0.04, 0.5, self.ylabel, va='center', rotation='vertical')
        plt.show(block=False)

#===========================================================================
""" Classes for backend.py """
#---------------------------------------------------------------------------

class datacube_of_galaxy(galaxy):
	"""
	An object that will create a datacube for one galaxy.
	Child class that inherits from parent class 'galaxy'.
	"""

	pass

	def setup_tasks(self):

		# If overwriting, do all subgridding
		if ow:
			self.do_GMC_dc 	=	True
			self.do_DNG_dc 	=	True
			self.do_DIG_dc 	=	True
			print('Overwrite is ON')
		# If not overwriting, check if subgridding has been done
		if not ow:
			self.do_GMC_dc 	=	False
			self.do_DNG_dc 	=	False
			self.do_DIG_dc 	=	False
			# Try to load GMC datacube
			dc 				=	self.datacube.get_dc(ISM_dc_phase='GMC',target=target)
			if type(dc) == int: self.do_GMC_dc = True
			# Try to load DNG and DIG datacube
			dc 				=	self.datacube.get_dc(ISM_dc_phase='DNG',target=target)
			if type(dc) == int: self.do_DNG_dc = True
			dc 				=	self.datacube.get_dc(ISM_dc_phase='DIG',target=target)
			if type(dc) == int: self.do_DIG_dc = True
			print('Overwrite is OFF, will do:')
			if self.do_GMC_dc: print('- Create datacube for GMCs')
			if self.do_DNG_dc: print('- Create datacube for DNG')
			if self.do_DIG_dc: print('- Create datacube for DIG')
			if self.do_GMC_dc + self.do_DNG_dc + self.do_DIG_dc == 0: print('Nothing!')

	def add_dc(self,ISM_phase):

		if ISM_phase == 'GMC': dataframe			=	self.particle_data.get_raw_data(data='ISM')[ISM_phase]
		if ISM_phase in ['DNG','DIG']: dataframe 	=	self.particle_data.get_raw_data(data='ISM')['dif']

		dc,dc_sum,x_axis,v_axis 	=	aux.mk_datacube(self,dataframe,ISM_phase=ISM_phase)

	  	# Comparison with interpolated results
		if target != 'm':
			print('Total luminosity in datacube: %.2e L_sun' % dc_sum)
			print('Total luminosity from interpolation: %.2e L_sun ' % (np.sum(dataframe[target])))
		if target == 'm':
			print('Total mass in datacube: %.2e Msun' % dc_sum)
			print('Total mass from interpolation: %.2e Msun ' % (np.sum(dataframe[target])))

		# Update datacube attribute in ISM dataframe
		metadata 				=	dataframe.metadata
		metadata[target+'_'+ISM_phase+'_done'] = True
		dataframe.metadata 		=	metadata
		if ISM_phase == 'GMC': aux.save_temp_file(dataframe,self.name,self.zred,ISM_phase=ISM_phase)
		if ISM_phase in ['DNG','DIG']: aux.save_temp_file(dataframe,self.name,self.zred,ISM_phase='dif')

		# Save datacube
		dc.metadata 			=	{'dc_sum':dc_sum,'x_axis':x_axis,'v_axis':v_axis}
		aux.save_temp_file(dc,self.name,self.zred,target=target,dc_type=ISM_phase,gal_index=self.gal_index)

