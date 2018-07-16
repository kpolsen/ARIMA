# coding=utf-8
"""
###     Submodule: aux.py of SIGAME              		###
"""

import numpy as np
import pandas as pd
import pdb as pdb
import scipy as scipy
from scipy import optimize
import scipy.stats as stats
import scipy.integrate as integrate
from scipy.interpolate import RegularGridInterpolator,griddata
from scipy.interpolate import interp1d
import multiprocessing as mp
import os
import time
import matplotlib.pyplot as plt
import linecache as lc
import re as re
import sys as sys
import cPickle
import sympy as sy
import astropy as astropy

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

params                      =   np.load('temp_params.npy').item()
for key,val in params.items():
    exec(key + '=val')

#===========================================================================
""" For classes in general """
#---------------------------------------------------------------------------

def update_dictionary(values,new_values):
    """ updates the entries to values with entries from new_values that have
    matching keys. """
    for key in values:
        if key in new_values:
            values[key]     =   new_values[key]
    return values

def save_temp_file(data,galname,zred,**kwargs):
    """
    Stores temporary files according to their sim or ISM type and stage of processing.
    """
    args    =   dict(sim_type='', ISM_phase='', stage='', dc_type='')
    args    =   update_dictionary(args,kwargs)
    for key in args: exec(key + '=args[key]')

    # 1) For me, it is simplier to read and right args as done above (doesn't really matter). 2) update_dictionary returns a dictionary and args needs to be set equal to updated args as done above.
    # args                    =   {'sim_type':'','ISM_phase':'','stage':'','dc_type':''}
    # update_dictionary(args,kwargs)
    # for key in args: exec(key + '=args[key]')

    if sim_type != '':
        data.to_pickle(d_data+'particle_data/sim_data/'+'z'+'{:.2f}'.format(zred)+'_'+galname+'_sim.'+sim_type)

    if ISM_phase != '':
        # Store in h5 data format to retain metadata
        filename = d_data+'particle_data/ISM_data/'+'z'+'{:.2f}'.format(zred)+'_'+galname+'_'+ISM_phase+'.h5'
        h5store(data, 'data', filename, **kwargs)

    if dc_type != '':
        # Store in h5 data format to retain metadata
        filename = d_data+'datacubes/%s_%s_%s_i%s_G%s.h5' % (z1, target, dc_type, inc_dc, kwargs['gal_index']+1)
        h5store(data, 'data', filename, **kwargs)

def load_temp_file(filename,dc_name):
    """
    Way to load metadata with dataframe
    """

    try: 
        with pd.HDFStore(filename) as store:
            data            = store[dc_name]
            meta            = store.get_storer(dc_name).attrs.metadata
        data.metadata   =   meta
        store.close() # will this avoid a 1 kB file from being created??
    except:
        try: 
            data            =   pd.read_pickle(filename)
        except:
            data            =   0

    return data

def h5store(df, dc_name, filename, **kwargs):
    """
    Way to store metadata with dataframe
    """

    try:
        metadata            =   df.metadata
    except:
        metadata            =   {}

    store = pd.HDFStore(filename)
    store.put(dc_name, df)
    store.get_storer(dc_name).attrs.metadata = metadata
    store.close()

def get_UV_str(z1,SFRsd):
    """
    Reads in SFR surface density and compares with SFR surface density of the MW.
    Then finds nearest FUV grid point in cloudy models and returns it as a string.
    """

    if z1 == 'z6':
        UV                  =   [5,35]
        UV_str              =   str(int(UV[np.argmin(np.abs(np.array(UV)-SFRsd/SFRsd_MW))]))
    if z1 == 'z2':
        UV                  =   [0.001,0.02]
        UV_str              =   str(UV[np.argmin(np.abs(np.array(UV)-SFRsd/SFRsd_MW))])
    if z1 == 'z0':
        UV                  =   [0.0002,0.002]
        UV_str              =   str(UV[np.argmin(np.abs(np.array(UV)-SFRsd/SFRsd_MW))])

    return(UV_str)

#===========================================================================
""" For datacube_of_galaxy class """
#---------------------------------------------------------------------------

def mk_datacube(gal_obj,dataframe,ISM_phase='GMC'):

    print('\nNOW CREATING DATACUBES OF %s FOR ISM PHASE %s' % (target,ISM_phase))

    # Create velocity and position axes
    v_axis              =   np.arange(-v_max,v_max+v_max/1e6,v_res)
    x_axis              =   np.arange(-x_max_pc,x_max_pc+x_max_pc/1e6,x_res_pc)
    dv                  =   v_axis[1]-v_axis[0]
    dx                  =   x_axis[1]-x_axis[0]

    # Move axes to center value of each pixel
    v_axis              =   v_axis[0:len(v_axis)-1]+v_res/2.
    x_axis_kpc          =   (x_axis[0:len(x_axis)-1]+x_res_pc/2.)/1000.

    print(' 1) Create (if not there already) radial profiles of all model clouds')
    if ISM_phase == 'GMC': rad_prof_path = d_data+'cloud_profiles/%s_%s_%s_rad_profs.npy' % (target,z1,ISM_phase)
    if ISM_phase != 'GMC': rad_prof_path = d_data+'cloud_profiles/%s_%s_%s_rad_profs_%sUV.npy' % (target,z1,ISM_phase,gal_obj.UV_str)
    if not os.path.exists(rad_prof_path): 
        mk_cloud_rad_profiles(ISM_phase=ISM_phase,target=target,FUV=gal_obj.UV_str)

    print(' 2) Load and position clouds on grids and save')
    t1 = time.clock()
    global clouds
    clouds                                      =   load_clouds(dataframe,target,ISM_phase)
    results                                     =   drizzle(v_axis,x_axis_kpc,rad_prof_path,ISM_phase=ISM_phase,target=target,FUV=gal_obj.UV_str,zred=gal_obj.zred)
    dc,dc_sum                                   =   results['datacube'],results['tot']
    t2 = time.clock()
    dt = t2-t1
    if dt < 60: print('Time it took to do this ISM phase: %.2f s' % (t2-t1))
    if dt > 60: print('Time it took to do this ISM phase: %.2f min' % ((t2-t1)/60.))
    print(target + ' = %.2e [unit] from clouds' % (np.sum(clouds['L_CII'])))
    print(target + ' = %.2e [unit] from datacube' % dc_sum)
    dc                  =   pd.DataFrame({'data':[dc]})
    pdb.set_trace()
    return(dc,dc_sum,x_axis,v_axis)

@profile
def drizzle(v_axis,x_axis_kpc,rad_prof_path,ISM_phase,target,plotting=True,verbose=False,checkplot=False,FUV='5',zred=0):
    '''
    Purpose
    ---------
    Drizzle *all* clouds onto galaxy grid in velocity and space

    '''
    N_clouds                =   len(clouds)
    print('Now drizzling %s %s clouds onto galaxy grid at %s x MW FUV...' % (N_clouds,ISM_phase,FUV))

    # Empty numpy array to hold result
    lenv,lenx               =   len(v_axis),len(x_axis_kpc)
    result                  =   np.zeros([lenv,lenx,lenx])

    if verbose:
        if ISM_phase == 'GMC':
            model_path      =   d_cloudy_models+'GMC/output/'+ext_DENSE+'_'+z1+'/GMC_'
            models          =   pd.read_pickle(d_cloudy_models+'GMC/grids/GMCgrid'+ ext_DENSE + '_' + z1+'_em.models')
        if ISM_phase in ['DNG','DIG']:
            model_path      =   d_cloudy_models+'dif/output/_'+FUV+'UV'+ext_DIFFUSE+'_'+z1+'/dif_'
            models          =   pd.read_pickle(d_cloudy_models+'dif/grids/difgrid_'+FUV+'UV'+ ext_DIFFUSE + '_' + z1+'_em.models')


    # For reference, get total values of "target" from interpolation
    if target == 'm':
        interpolation_result =  clouds[target].values
    else:
        line = target.replace('L_','')
        if ISM_phase in ['GMC']:        interpolation_result    =   clouds[target].values
        if ISM_phase in ['DNG','DIG']:  interpolation_result    =   clouds[target+'_'+ISM_phase].values

    # LOADING
    model_rad_profs         =   np.load(rad_prof_path)
    models_r_pc             =   model_rad_profs[1,:,:]  # pc
    models_SB               =   model_rad_profs[0,:,:]  # Lsun/pc^2
    # Assign these models to clouds in the galaxy:
    model_index             =   clouds['closest_model_i'].values
    clouds_r_pc             =   [models_r_pc[int(i)] for i in model_index]
    clouds_SB               =   [models_SB[int(i)] for i in model_index]
    clouds_R_pc             =   np.max(clouds_r_pc,axis=1)
    vel_disp_gas            =   clouds['vel_disp_gas'].values
    v_proj                  =   clouds['v_proj'].values
    x_cloud                 =   clouds['x'].values
    y_cloud                 =   clouds['y'].values

    # SETUP
    # Some things we will need
    v_max, v_res            =   max(v_axis),v_axis[1]-v_axis[0]
    fine_v_axis             =   np.arange(-v_max,v_max+v_max/1e6,v_res/8.)
    x_res_kpc               =   x_axis_kpc[1]-x_axis_kpc[0]
    x_res_pc                =   x_res_kpc*1000.
    npix_highres_def        =   9# highres pixels per galaxy pixel, giving 3 pixels on either side of central pixel (of resolution x_res_pc)
    x_res_kpc_highres       =   x_res_kpc/npix_highres_def # size of high resolution pixel
    x_res_pc_highres        =   x_res_kpc_highres*1000. # size of high resolution pixel
    pix_area_highres_kpc    =   x_res_kpc_highres**2. # area of high resolution pixel
    pix_area_highres_pc     =   x_res_pc_highres**2. # area of high resolution pixel

    # What galaxy pixel center comes closest to this cloud center?
    min_x                   =   np.min(x_axis_kpc)
    range_x                 =   np.max(x_axis_kpc) - min_x
    x_index                 =   np.round((x_cloud-min_x)/range_x*(lenx-1)).astype(int)
    y_index                 =   np.round((y_cloud-min_x)/range_x*(lenx-1)).astype(int)


    # ----------------------------------------------
    # SMALL CLOUDS
    # Clouds that are "very" small compared to pixel size in galaxy image
    # give all their luminosity to one galaxy pixel
    small_cloud_index       =   [(0 < clouds_R_pc) & (clouds_R_pc <= x_res_kpc*1000./8.)][0]
    print('%s small clouds, unresolved by galaxy pixels' % (len(small_cloud_index[small_cloud_index == True])))
    small_cloud_x_index     =   np.extract(small_cloud_index,x_index)
    small_cloud_y_index     =   np.extract(small_cloud_index,y_index)
    small_cloud_targets     =   np.extract(small_cloud_index,interpolation_result)
    small_cloud_vdisp_gas   =   np.extract(small_cloud_index,vel_disp_gas)
    small_cloud_v_proj      =   np.extract(small_cloud_index,v_proj)
    pixels_outside          =   0
    for target1,vel_disp_gas1,v_proj1,i_x,i_y in zip(small_cloud_targets,small_cloud_vdisp_gas,small_cloud_v_proj,small_cloud_x_index,small_cloud_y_index):
        vel_prof                =       mk_cloud_vel_profile(v_proj1,vel_disp_gas1,fine_v_axis,v_axis)
        try:
            result[:,i_x,i_y]       +=      vel_prof*target1
        except:
            pixels_outside      +=  1

    # ----------------------------------------------
    # LARGER CLOUDS
    # Cloud is NOT very small compared to pixel size in galaxy image
    # => resolve cloud and split into nearby pixels
    large_cloud_index       =   [(0 < clouds_R_pc) & (clouds_R_pc > x_res_pc/8.)][0] # boolean array
    print('%s large clouds, resolved by galaxy pixels' % (len(large_cloud_index[large_cloud_index == True])))
    # Total number of highres pixels to (more than) cover this cloud
    Npix_highres            =   2.2*np.extract(large_cloud_index,clouds_R_pc)/(x_res_pc_highres)
    Npix_highres            =   np.ceil(Npix_highres).astype(int)
    # Count highres cloud pixels in surrounding galaxy pixels
    max_pix_dif             =   np.extract(large_cloud_index,clouds_R_pc)/x_res_pc
    max_pix_dif             =   np.ceil(max_pix_dif).astype(int)
    highres_axis_max        =   (np.array(Npix_highres)*x_res_pc_highres)/2.
    large_cloud_interpol    =   np.extract(large_cloud_index,interpolation_result)
    large_cloud_model_index =   np.extract(large_cloud_index,model_index)
    large_cloud_x_index     =   np.extract(large_cloud_index,x_index)
    large_cloud_y_index     =   np.extract(large_cloud_index,y_index)
    large_cloud_targets     =   np.extract(large_cloud_index,clouds[target].values)
    large_cloud_vdisp_gas   =   np.extract(large_cloud_index,vel_disp_gas)
    large_cloud_v_proj      =   np.extract(large_cloud_index,v_proj)
    large_cloud_R_pc        =   np.extract(large_cloud_index,clouds_R_pc)
    large_models_r_pc       =   [models_r_pc[int(_)] for _ in large_cloud_model_index]
    large_models_SB         =   [models_SB[int(_)] for _ in large_cloud_model_index]
    i                       =   0
    pixels_outside          =   0
    # overwrite_me            =   np.zeros([lenv,lenx,lenx])
    for target1,vel_disp_gas1,v_proj1,i_x,i_y,npix_highres in zip(large_cloud_targets,large_cloud_vdisp_gas,large_cloud_v_proj,large_cloud_x_index,large_cloud_y_index,Npix_highres):
        if np.sum(large_models_SB[i]) > 0:
            # overwrite_me        =   overwrite_me*0.
            vel_prof            =   mk_cloud_vel_profile(v_proj1,vel_disp_gas1,fine_v_axis,v_axis)
            # create grid of coordinates for high-res image of cloud:
            # npix_highres = npix_highres*10. # CHECK
            # x_res_pc_highres = x_res_pc_highres/10. # CHECK
            x_highres_axis      =   np.linspace(-highres_axis_max[i],highres_axis_max[i],npix_highres)
            x_highres_mesh, y_highres_mesh = np.mgrid[slice(-max(x_highres_axis) + x_res_pc_highres/2., max(x_highres_axis) - x_res_pc_highres/2. + x_res_pc_highres/1e6, x_res_pc_highres),\
                            slice(-max(x_highres_axis) + x_res_pc_highres/2., max(x_highres_axis) - x_res_pc_highres/2. + x_res_pc_highres/1e6, x_res_pc_highres)]
            radius              =   np.sqrt(x_highres_mesh**2+y_highres_mesh**2)
            # create high-res image of this cloud, evaluating surface brightness at the center of each high-res pixel
            interp_func_r       =   interp1d(large_models_r_pc[i],large_models_SB[i],fill_value=large_models_SB[i][-1],bounds_error=False)
            im_cloud            =   interp_func_r(radius)
            im_cloud[radius > large_cloud_R_pc[i]]  =   0.

            # Remove "per area" from image units [Lsun/pc^2 -> Lsun or Msun/pc^2 -> Msun]
            im_cloud            =   im_cloud*pix_area_highres_pc

            # CHECK plot
            # R_max = highres_axis_max[i]
            # plt.close('all')
            # plot.simple_plot(figsize=(6, 6),xr=[-R_max,R_max],yr=[-R_max,R_max],aspect='equal',\
            #     x1=x_highres_axis,y1=x_highres_axis,col1=im_cloud,\
            #     contour_type1='mesh',xlab='x [pc]',ylab='y [pc]',\
            #     colorbar1=True,lab_colorbar1='L$_{\odot}$ per cell')
            # plt.show(block=False)


            # Normalize to total luminosity of this cloud:
            im_cloud            =   im_cloud*large_cloud_interpol[i]/np.sum(im_cloud)

            if verbose: print('check\n%.2e Msun from cloud image' % (np.sum(im_cloud)))

            # Identify low rew pixels that we will be filling up
            x_indices           =   [large_cloud_x_index[i] + _ for _ in np.arange(-max_pix_dif[i],max_pix_dif[i]+1)]
            y_indices           =   [large_cloud_y_index[i] + _ for _ in np.arange(-max_pix_dif[i],max_pix_dif[i]+1)]
            x_index_mesh, y_index_mesh = np.mgrid[slice(x_indices[0], x_indices[-1] + 1./1e6, 1),\
                slice(y_indices[0], y_indices[-1] + 1./1e6, 1)]
            x_index_array       =   x_index_mesh.reshape(len(x_indices)**2)
            y_index_array       =   y_index_mesh.reshape(len(y_indices)**2)
            # Center in x and y direction for highres cloud image:
            i_highres_center    =   float(npix_highres)/2.-0.5
            check               =   np.zeros([lenv,lenx,lenx])

            for x_i,y_i in zip(x_index_array,y_index_array):
                xdist_highres_from_cloud_center         =   (int(x_i) - large_cloud_x_index[i]) * npix_highres_def
                ydist_highres_from_cloud_center         =   (int(y_i) - large_cloud_y_index[i]) * npix_highres_def
                x_i_highres         =   int(i_highres_center + xdist_highres_from_cloud_center)# - (npix_highres-1)/2.)
                y_i_highres         =   int(i_highres_center + ydist_highres_from_cloud_center)# - (npix_highres-1)/2.)
                try:
                    # result[:,int(x_i),int(y_i)]             +=  vel_prof*np.sum(im_cloud[x_i_highres:x_i_highres+npix_highres_def+1, y_i_highres:y_i_highres+npix_highres_def+1])
                    check[:,int(x_i),int(y_i)]      +=  vel_prof*np.sum(im_cloud[x_i_highres:x_i_highres+npix_highres_def+1, y_i_highres:y_i_highres+npix_highres_def+1])
                    # overwrite_me                            +=  vel_prof*np.sum(im_cloud[x_i_highres:x_i_highres+npix_highres_def+1, y_i_highres:y_i_highres+npix_highres_def+1])
                except:
                    pixels_outside                  +=1
                    pass

            check              =   check*large_cloud_interpol[i]/np.sum(check)
            result             +=   check

            if verbose:
                if target == 'm':
                    print('%.2e Msun from drizzled image of cloud' % (np.sum(check)))
                    print('%.2e Msun from cloudy model result grid' % (large_cloud_interpol[i]))
                else:
                    print('%.2e Lsun from drizzled image of cloud' % (np.sum(check)))
                    if ISM_phase == 'GMC':
                        print('%.2e Lsun from cloudy grid' % (models[target][large_cloud_model_index[i]]))
                    if ISM_phase in ['DNG','DIG']:
                        if ISM_phase == 'DNG':
                            nearest = models[target][large_cloud_model_index[i]]*models['f_'+line+'_DNG'][large_cloud_model_index[i]]
                            print('%.2e Lsun from cloudy grid nearest model' % nearest)
                            print('%.2e Lsun from interpolation' % (large_cloud_interpol[i]))
                        if ISM_phase == 'DIG':
                            print('%.2e Lsun from cloudy grid nearest model' % (models[target][large_cloud_model_index[i]]*(1-models['f_'+line+'_DNG'][large_cloud_model_index[i]])))

        i                   +=  1

    print('%s highres cloud pixels went outside galaxy image' % (pixels_outside))

    # ----------------------------------------------
    # OPTION 1: CONVERSION TO JY
    if target != 'm':
        # convert to Jy = ergs/s / m^2 / Hz (observed frequency right?)
        tot_Lsun            =   np.sum(result)
        dc_Jy               =   np.nan_to_num(result)
        if target != 'm':
            print(target + ' = %.2e Lsun' % tot_Lsun)
            D_L                     =   get_lum_dist(zred)
            print('Max in Lsun: %.2e ' % np.max(dc_Jy))
            freq_obs                =   params['f_'+target.replace('L_','')]*1e9/(1+zred)
            dc_Jy                   =   dc_Jy*Lsun/(1e-26*v_res*1e3*freq_obs/clight*(4*np.pi*(D_L*1e6*pc2m)**2))
        print('Max in Jy: %.2e ' % np.max(dc_Jy))
        results = {'tot':tot_Lsun,'datacube':dc_Jy}

    # ----------------------------------------------
    # OPTION 2: SAVE MASS AS IT IS
    if target == 'm':
        results = {'tot':np.sum(result),'datacube':np.nan_to_num(result)}

    return(results)

def load_clouds(dataframe,target,ISM_phase):
    '''
    Purpose
    ---------
    Load clouds from saved galaxy files, and convert to a similar format.

    '''

    clouds1             =   dataframe.copy()

    if ISM_phase == 'DIG':
        clouds1[target]     =   dataframe[target+'_DIG'].values

    if ISM_phase == 'DNG':
        clouds1[target]     =   dataframe[target+'_DNG'].values

    if ISM_phase == 'GMC':
        clouds1['R']        =   dataframe['Rgmc'].values

    # TEST
    clouds1             =   clouds1[0:500]


    clouds1['x']        =   dataframe['x'] # pc
    clouds1['y']        =   dataframe['y'] # pc
    clouds1['z']        =   dataframe['z'] # pc

    # TEST
    # clouds1['vx']         =   clouds1['vx']*0.+100.
    # clouds1['vx'][clouds['x'] < 0] =  -100.
    # clouds1['vy']         =   clouds1['vy']*0.
    # clouds1['vz']         =   clouds1['vz']*0.
    # Rotate cloud positions and velocities around y axis
    inc_rad             =   2.*np.pi*float(inc_dc)/360.
    coord               =   np.array([clouds1['x'],clouds1['y'],clouds1['z']])
    coord_rot           =   np.dot(rotmatrix(inc_rad,axis='y'),coord)
    clouds1['x'] = coord_rot[0]
    clouds1['y'] = coord_rot[1]
    clouds1['z'] = coord_rot[2]
    vel                 =   np.array([clouds1['vx'],clouds1['vy'],clouds1['vz']])
    vel_rot             =   np.dot(rotmatrix(inc_rad,axis='y'),vel)
    clouds1['vx'] = vel_rot[0]; clouds1['vy'] = vel_rot[1]; clouds1['vz'] = vel_rot[2]
    clouds1['v_proj']   =   clouds1['vz'] # km/s
    # pdb.set_trace()

    # Cut out only what's inside image:
    radius_pc           =   np.sqrt(clouds1['x']**2 + clouds1['y']**2)
    clouds1             =   clouds1[radius_pc < x_max_pc]
    clouds1             =   clouds1.reset_index(drop=True)

    # TEST!!
    # clouds1           =   clouds1.iloc[0:2]
    # clouds1['x'][0]   =   0
    # clouds1['y'][0]   =   0

    return(clouds1)

def mk_cloud_rad_profiles(ISM_phase='GMC', target='L_CII',verbose=False, FUV='5'):
    '''
    Purpose
    ---------
    Make the radial profile (in the sky plane) for each cloud in the dense and diffuse grid
    '''

    print('Make radial profiles of %s clouds seen face on' % (ISM_phase))
    if ISM_phase in ['DNG','DIG']: print('at %s x MW FUV' % (FUV))

    # Find and load cloudy model results
    if ISM_phase == 'GMC':
        model_path      =   d_cloudy_models+'GMC/output/'+ext_DENSE+'_'+z1+'/GMC_'
        models          =   pd.read_pickle(d_cloudy_models+'GMC/grids/GMCgrid'+ ext_DENSE + '_' + z1+'_em.models')
    if ISM_phase in ['DNG','DIG']:
        model_path      =   d_cloudy_models+'dif/output/_'+FUV+'UV'+ext_DIFFUSE+'_'+z1+'/dif_'
        models          =   pd.read_pickle(d_cloudy_models+'dif/grids/difgrid_'+FUV+'UV'+ ext_DIFFUSE + '_' + z1+'_em.models')

    # Renaming mass parameter from Mgmc to m, if not present:
    if not 'm' in models.keys():
        if ISM_phase == 'GMC': models['m']      =   models['Mgmc']
        if ISM_phase in ['DNG']: models['m']    =   models['m_dif']*models['fm_DNG']
        if ISM_phase in ['DIG']: models['m']    =   models['m_dif']*(1.-models['fm_DNG'])

    # Number of models:
    N_clouds            =   len(models)

    # Number of bins in radial and z direction
    N_rings             =   50
    N_zbins             =   100

    # Empty matrix for results
    clouds              =   np.zeros([3,N_clouds,N_rings])

    part1               =   0.1
    for i in range(0,N_clouds):
        name                =   model_path + str(i)
        try:
            script_in           =   open(name+'.out','r')
        except:
            print('No model output found for %s' % name)
        else:
            # First check whether run completed
            if 'went wrong' in open(name+'.out').read():
                print(name+' crashed')
            if os.stat(name+'.out').st_size == 0:
                print(name+' did not run')
            if 'Cloudy exited OK]' in open(name+'.out').read():

                # Read in cooling rates
                columns             =   ['depth','C_OI','C_OIII','C_NII_122','C_CII','C_NII_205','C_CI_369','C_CI_609','C_CO32','C_CO21','C_CO10']
                cool                =   pd.read_table(name+'.str',names=columns,skiprows=n_comments(name+'.str', '#'),sep='\s',engine='python',index_col=False)
                R_cm                =   np.max(cool['depth'].values)-cool['depth'][::-1].values # cm
                R_pc                =   R_cm/pc2cm # pc
                R_cloud_cm          =   max(R_cm)
                R_cloud_pc          =   max(R_cm)/pc2cm

                # Read in ionization state of the gas
                columns             =   ['depth','Te','Htot','hden','eden','2H_2/H','HI','HII','HeI','HeII','HeIII','CO/C','C1','C2','C3','C4','O1','O2','O3','O4','O5','O6','H2O/O','AV(point)','AV(extend)']
                elem                =   pd.read_table(name+'.ovr',names=columns,skiprows=n_comments(name+'.ovr', '#'),sep='\s',engine='python',index_col=False)
                nH                  =   elem['hden'][::-1].values
                x_HI                =   elem['HI'][::-1].values # to know if cloud cell is DNG or DIG
                x_H2                =   elem['2H_2/H'][::-1].values # to know if cloud cell is DNG or DIG
                x_neu               =   x_HI+x_H2

                # target as a function of radius
                if target == 'm':
                    rad_prof            =   mH*nH # kg/cm^-3
                if target != 'm':
                    cooling_line        =   target.replace('L_','')
                    cooling_rate        =   'C_'+cooling_line
                    rad_prof            =   cool[cooling_rate][::-1].values # ergs/s/cm^3

                # Separate cooling rate (mass) from the different ISM phases
                if ISM_phase == 'GMC':
                    rad_prof_array      =   np.array([rad_prof])
                    L_tot               =   models[target].values
                if ISM_phase == 'DNG':
                    rad_prof_array      =   rad_prof * x_neu
                    if target == 'm': L_tot             =   models[target].values
                    if target != 'm': L_tot             =   models[target].values*models['f_'+cooling_line+'_DNG'].values
                if ISM_phase == 'DIG':
                    rad_prof_array      =   rad_prof * (1.-x_neu)
                    if target == 'm': L_tot             =   models[target].values
                    if target != 'm': L_tot             =   models[target].values*models['f_'+cooling_line+'_DNG'].values

                interp_cooling_rate =   interp1d(R_cm, rad_prof_array)

                # Create array of radii in xy-plane to check out
                dr_xy               =   R_cloud_cm/N_rings
                r_xy_array          =   np.arange(dr_xy/2.,R_cloud_cm,dr_xy)

                # Create empty array to contain luminosity (mass)
                L_r_xy              =   np.zeros(N_rings)
                for i_r_xy in range(N_rings):
                    # for each ring around the z axis, slice the cylinder into rings:
                    try:
                        max_z               =   np.sqrt(R_cloud_cm**2-r_xy_array[i_r_xy]**2) # max height in z to go to...
                    except:
                        pdb.set_trace()
                    dz                  =   max_z/N_zbins
                    for i_z in range(N_zbins):
                        # for each ring of the cylinder, integrate luminosity:
                        V_cylinder          =   dz*dr_xy*2*np.pi*r_xy_array[i_r_xy]
                        R_z                 =   np.sqrt(r_xy_array[i_r_xy]**2 + (i_z*dz)**2) # radius from center at this height z
                        if target == 'm': L_r_xy[i_r_xy]        +=  2.*interp_cooling_rate(R_z)*V_cylinder*1e-7/Lsun # Lsun
                        if target != 'm': L_r_xy[i_r_xy]        +=  2.*interp_cooling_rate(R_z)*V_cylinder/Msun # Msun

                if verbose:
                    print('\n check:')
                    if target == 'm':
                        print('%.2e Msun from summing mass in rings' % np.sum(L_r_xy))
                        L_int                   =   scipy.integrate.simps(4*np.pi*rad_prof_array*1e-7/Lsun*R_cm**2.,R_cm)
                        print('%.2e Msun from integrating density over sphere (CHECK)' % L_int)
                        # L_int_approx          =   models[target+'_int'].values[i]*models['f_'+cooling_line+'_DNG'].values[i]
                        # print('%.2e Msun same, but as stored in GMC/dif module (f_CII CMB-corrected)' % L_int_approx)
                    else:
                        print('%.2e Lsun from summing luminosity in rings' % np.sum(L_r_xy))
                        L_int                   =   scipy.integrate.simps(4*np.pi*rad_prof_array*1e-7/Lsun*R_cm**2.,R_cm)
                        print('%.2e Lsun from integrating cooling rate over sphere (CHECK)' % L_int)
                        L_int_approx            =   models[target+'_int'].values[i]*models['f_'+cooling_line+'_DNG'].values[i]
                        print('%.2e Lsun same, but as stored in GMC/dif module (f_CII CMB-corrected)' % L_int_approx)

                # Correct luminosity to match the total emergent (CMB-corrected) luminosity:
                L_r_xy                  =   L_r_xy*L_tot[i]/np.sum(L_r_xy)
                if verbose:
                    if target == 'm':
                        print('%.2e Msun from summing density in rings, scaled' % np.sum(L_r_xy))
                    else:
                        print('%.2e Lsun from emergent intensity, CMB-corrected' % L_tot[i])
                        print('%.2e Lsun from summing luminosity in rings, scaled' % np.sum(L_r_xy))

                # Create empty array to contain surface brightness (density)
                SB_r_xy             =   np.zeros(N_rings)
                for i_r_xy in range(N_rings):
                    # Convert luminosity (mass) to surface brightness (density)
                    dr                      =   r_xy_array[1]-r_xy_array[0] # cm
                    area_ring               =   np.pi*((r_xy_array[i_r_xy] + dr/2.)**2 - (r_xy_array[i_r_xy] - dr/2.)**2) # cm^2
                    SB_r_xy[i_r_xy]         =   L_r_xy[i_r_xy]/(area_ring/pc2cm**2) # Lsun/pc^2
                
                if verbose:
                    L_SB                =   0
                    rs                  =   np.arange(0,200,1) # pc
                    dr                  =   rs[1]-rs[0]
                    pix_area            =   dr**2
                    x_mesh, y_mesh      =   np.mgrid[slice(-max(rs) + dr/2., max(rs) - dr/2. + dr/1e6, dr),\
                            slice(-max(rs) + dr/2., max(rs) - dr/2. + dr/1e6, dr)]
                    radius              =   np.sqrt(x_mesh**2 + y_mesh**2)
                    interp_func_r       =   interp1d(r_xy_array/pc2cm,SB_r_xy,fill_value=SB_r_xy[-1],bounds_error=False)
                    im_cloud            =   interp_func_r(radius)*pix_area
                    print('%.2e Lsun from summing surface brightness in rings, scaled' % np.sum(im_cloud))

                clouds[0,i,:]               =   SB_r_xy # Lsun/pc^2 or Msun/pc^2
                clouds[1,i,:]               =   r_xy_array/pc2cm # radial axis in pc
                clouds[2,i,:]               =   L_r_xy # Lsun or Msun

        if 1.*i/N_clouds > part1:
            print('\r %s %% done!' % int(part1*100))
            part1                   =   part1+0.1
    if ISM_phase == 'GMC': np.save(d_data+'cloud_profiles/%s_%s_%s_rad_profs.npy' % (target, z1, ISM_phase), clouds)
    if ISM_phase in ['DNG','DIG']: np.save(d_data+'cloud_profiles/%s_%s_%s_rad_profs_%sUV.npy' % (target, z1, ISM_phase, FUV), clouds)

def mk_cloud_vel_profile(v_proj,vel_disp_gas,fine_v_axis,v_axis,plotting=False):
    '''
    Purpose
    ---------
    Make the velocity profile for *one* cloud

    What this function does
    ---------
    Calculates the fraction of total flux [Jy] going into the different velocity bins

    Arguments
    ---------
    v_proj: projected line-of-sight velocity of the cloud

    vel_disp_gas: velocity dispersion of the cloud

    v_axis: larger velocity axis to project clouds onto
    '''

    if vel_disp_gas > 0:

        # Evaluate Gaussian on fine velocity axis
        Gaussian                =   1./np.sqrt(2*np.pi*vel_disp_gas**2) * np.exp( -(fine_v_axis-v_proj)**2 / (2*vel_disp_gas**2) )

        # Numerical integration over velocity axis bins
        v_res                   =   (v_axis[1]-v_axis[0])/2.

        vel_prof                =   abs(np.array([integrate.trapz(fine_v_axis[(fine_v_axis >= v-v_res) & (fine_v_axis < v+v_res)], Gaussian[(fine_v_axis >= v-v_res) & (fine_v_axis < v+v_res)]) for v in v_axis]))

        if plotting:
            plot.simple_plot(fig=2,
                fontsize=12,xlab='v [km/s]',ylab='F [proportional to Jy]',\
                x1=fine_v_axis,y1=Gaussian,col1='r',ls1='--',\
                x2=v_axis,y2=vel_prof,col2='b',ls2='-.')

            plt.show(block=False)

        # Normalize that profile
        vel_prof                =   vel_prof*1./np.sum(vel_prof)

        # If vel_disp_gas is very small, vel_prof sometimes comes out with only nans, use a kind of Dirac delta function:
        if np.isnan(np.max(vel_prof)):
            vel_prof                =   v_axis*0.
            vel_prof[find_nearest(v_axis,v_proj,find='index')] = 1.

    else:
        # If vel_disp_gas = 0 km/s, use a kind of Dirac delta function:
        vel_prof                =   v_axis*0.
        vel_prof[find_nearest(v_axis,v_proj,find='index')] = 1.

    return(vel_prof)

#===============================================================================
""" Cosmology """
#-------------------------------------------------------------------------------

def get_lum_dist(zred):
    '''
    Purpose
    ---------
    Calculate luminosity distance for a certain redshift

    returns D_L in Mpc

    '''

    from astropy.cosmology import FlatLambdaCDM
    cosmo               =   FlatLambdaCDM(H0=hubble*100., Om0=omega_m, Ob0=1-omega_m-omega_lambda)

    try:
        if len(zred) > 1:
            D_L                 =   cosmo.luminosity_distance(zred).value
            zred_0              =   zred[zred == 0]
            if len(zred_0) > 0:
                D_L[zred == 0]      =   3+27.*np.random.rand(len(zred_0)) # Mpc (see Herrera-Camus+16)

    except:
        D_L                 =   cosmo.luminosity_distance(zred).value
        if zred == 0:
            # np.random.seed(len(zred_0))
            D_L                 =   100.*np.random.rand(1) # Mpc
            D_L                 =   D_L[0]

    # ( Andromeda is rougly 0.78 Mpc from us )

    return(D_L)

#===============================================================================
""" For cloudy model output analysis """
#-------------------------------------------------------------------------------

def n_comments(fn, comment='#'):
	'''
	Purpose
	---------
	Counts the number of lines to ignore in a cloudy file to get to the last iteration

	Arguments
	---------
	fn: name of cloudy output file - str

	comment: the string used to comment lines out in cloudy - str
	default = '#'


	'''

	with open(fn, 'r') as f:
	    n_lines         =   0
	    n_comments      =   0
	    n_tot           =   0
	    pattern = re.compile("^\s*{0}".format(comment))
	    for l in f:
	        if pattern.search(l) is None:
	            n_lines += 1         # how many lines of data?
	        else:
	            n_comments += 1      # how many lines of comments?
	            n_lines = 0          # reset line count
	        n_tot       +=      1
	n_skip      =   n_tot - n_lines     # (total number of lines in file minus last iteration)
	# n_skip      =   (n_comments-1)*n_lines+n_comments
	#print('Skip this many lines: ',n_skip)
	return n_skip

#===========================================================================
""" Some arithmetics """
#---------------------------------------------------------------------------

def rad(foo,labels):
    # Calculate distance from [0,0,0] in 3D for DataFrames!
    if len(labels)==3: return np.sqrt(foo[labels[0]]**2+foo[labels[1]]**2+foo[labels[2]]**2)
    if len(labels)==2: return np.sqrt(foo[labels[0]]**2+foo[labels[1]]**2)

def find_nearest(array,value,find='value'):
    idx = (np.abs(array-value)).argmin()
    if find == 'value': return array[idx]
    if find == 'index': return idx

# duplicate function in section 'Cosmology' - in use - can we switch to get_lum_dist()?

def luminosity_distance(zred):
	"""luminosity distance

	args
	----
	zred: redshift

	returns
	-------
	luminosity distance - scalar - Mpc
	"""

	def I(a): # function to integrate
		return 1/(a * np.sqrt( (omega_r/a**2) + (omega_m/a) + (omega_lambda * a**2) )) # num
	LL = 1/(1+zred) # num
	UL = 1 # num
	integral = integrate.quad(lambda a: I(a), LL, UL)
	d_p = (clight/1000./(100.*hubble)) * integral[0]
	return d_p * (1+zred)

def lin_reg_boot(x,y,n_boot=5000,plotting=False):
    print('Fit power law and get errors with bootstrapping!')

    n_boot                  =   5000
    nGal                    =   len(x)

    def calc_slope(i, x1, y1, nGal):

        random_i                        =   (np.random.rand(nGal)*nGal).astype(int)

        slope1,inter1,foo1,foo2,foo3    =   stats.linregress(x1[random_i],y1[random_i])
        return slope1,inter1

    slope,inter,x1,x2,x3    =   stats.linregress(x,y)
    boot_results            =   [[calc_slope(i,x1=x,y1=y,nGal=nGal)] for i in range(0,n_boot)]
    slope_boot              =   [boot_results[i][0][0] for i in range(0,n_boot)]
    inter_boot              =   [boot_results[i][0][1] for i in range(0,n_boot)]

    # if plotting:
    #     # Make plot of distribution!
    #     yr = [0,240]
    #     plot.simple_plot(fig=j+1,xlab = ' slope from bootstrapping %s times' % n_boot,ylab='Number',\
    #         xr=plot.axis_range(slope_boot,log=False),yr=yr,legloc=[0.05,0.8],\
    #         histo1='y',histo_real1=True,x1=slope_boot,bins1=n_boot/50.,\
    #         x2=[slope,slope],y2=yr,lw2=1,ls2='--',col2='blue',lab2='Slope from fit to models',\
    #         x3=[np.mean(slope_boot),np.mean(slope_boot)],y3=yr,lw3=1,ls3='--',col3='green',lab3='Mean slope from bootstrapping')
    #         # figname='plots/line_SFR_relations/bootstrap_results/'+line+'slope_boot.png',figtype='png')
    #     plt.show(block=False)

    print('Slope from fit to models: %s' % np.mean(slope))
    print('Bootstrap mean: %s' % np.mean(slope_boot))
    print('Bootstrap std dev: %s' % np.std(slope_boot))

    print('Intercept from fit to models: %s' % np.mean(inter))
    print('Bootstrap mean: %s' % np.mean(inter_boot))
    print('Bootstrap std dev: %s' % np.std(inter_boot))

    return(slope,inter,np.std(slope_boot),np.std(inter_boot))

def rotmatrix(angle,axis='x'):

	cos 		=	np.cos(angle)
	sin 		=	np.sin(angle)

	if axis == 'x':
		rotmatrix 		= 	np.array([[1,0,0],[0,cos,-sin],[0,sin,cos]])

	if axis == 'y':
		rotmatrix 		= 	np.array([[cos,0,sin],[0,1,0],[-sin,0,cos]])

	if axis == 'z':
		rotmatrix 		= 	np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])

	return rotmatrix

def annulus_area(radius,x0,y0,dr,dx):

    # get upper integration limit
    xf  =   lambda r: min([np.sqrt(r**2 - y0**2), x0+dx])
    rf  =   radius+dr

    # make lambda function for the area under the curve where the annulus intercepts a smal CC region
    f   =   lambda x, r: min([0.5*x*np.sqrt(r**2 - x**2) + np.arctan(x/np.sqrt(r**2-x**2))*r**2, y0+dx])

    # get the area where r = radius+dr
    A1  =   f(xf(rf),rf) - f(x0,rf)

    # get the area where r = radius
    A2  =   f(xf(radius),radius) - f(x0,radius)

    return abs(A1-A2)

#===============================================================================
""" Conversions """
#-------------------------------------------------------------------------------

def LsuntoJydv(Lsun,zred=7,d_L=69727,nu_rest=1900.5369):
	""" Converts luminosity (in Lsun) to velocity-integrated flux (in Jy*km/s)

	args
	----
	Lsun: numpy array
	solar luminosity (Lsun)

	zred: scalar
	redshift z (num)

	d_L: scalar
	luminosity distance (Mpc)

	nu_rest: scalar
	rest frequency of fine emission line (GHz)

	returns
	Jy*km/s array
	------
	"""

	return Lsun * (1+zred)/(1.04e-3 * nu_rest * d_L**2)

def solLum2mJy(Lsunkms, zred, d_L, nu_rest):
	""" Converts solar luminosity/(km/s) to milli-jansky/(km/s)

	args
	----
	Lsunkms: numpy array
	solar luminosity / vel bin ( Lsun/(km/s) )

	zred: scalar
	redshift z (num)

	d_L: scalar
	luminosity distance (Mpc)

	nu_rest: scalar
	rest frequency of fine emission line (GHz)

	returns
	Jy/(km/s) array
	------
	"""

	return Lsunkms * (1+zred)/(1.04e-3 * nu_rest * d_L**2) * 1000

def Jykms2solLum(Jykms, nu_rest, zred):
    """returns integrated Jy to solar luminosity

    args
    ----
    Jykms: scalar
    total in Jy*km/s (W/Hz/m^2*km/s)

    nu_rest: scalar
    rest frequency (GHz)

    zred: scalar
    redshift (num)

    d_L: scalar
    luminosity distance (Mpc)

    returns
    -------
    scalar
    solar luminosity
    """

    if type(zred) == 'int' or type(zred) == 'float':
    	d_L                 =   get_lum_dist(zred)
    else:
    	d_L                 =   np.array([get_lum_dist(z) for z in zred])

    if type(zred) == list:
    	zred 				= 	np.array(zred)

    # print("'Jykms2solLum' still needs verification")
    return 1.04e-3 * Jykms * (nu_rest/(1+zred)) * d_L**2

def disp2FWHM(sig):
    return 2*np.sqrt(2*np.log(2)) * sig

def W_m2_to_Jykm_s(line,zred,I):
    '''
    Purpose
    ----------
    Convertes flux in [W/m^2] to velocity integrated flux [Jy km/s]

    Arguments
    ----------
    - line: string
        Options:
        - CII
        - NII_122
        - NII_205
        - OIII
    - zred: redshift - float
    - I: intensity [W/m^2] - float

    Returns
    ----------
    - velocity integrated flux [Jy km/s] - float
    '''

    Mpc2m           =   pc2m * 1e6
    f_line          =   params['f_' + line]
    return (4 * np.pi * I * (Mpc2m**2) * (1 + zred)) / (1.04e-3 * Lsun * f_line)

def Jykm_s_to_W_m2(line,zred,I):
    '''
    Purpose
    ----------
    Converts velocity integrated flux [Jy km/s] to flux in [W/m^2]
    '''

    Mpc2m           =   pc2m * 1e6
    f_line          =   params['f_' + line]
    return I * (1.04e-3 * Lsun * f_line) / (4 * np.pi * (Mpc2m**2) * (1 + zred))

#===============================================================================
""" Other functions """
#-------------------------------------------------------------------------------

def diff_percent(x1,x2):
	'''
	Purpose
	-------
	Return difference in percent relative to x1: (x1-x2)/x1


	'''

	diff 			=	(x1-x2)/x1*100.

	return(diff)

def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold='nan')
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)

def line_name(line,latex=False):
	'''
	Purpose

	Get name for a line in pretty writing
	'''

	line_dictionary = {\
		'CII':'[CII]',\
		'OI':'[OI]',\
		'OIII':'[OIII]',\
		'NII_122':'[NII]122',\
		'NII_205':'[NII]205',\
		'CI_609':'CI(1-0)609',\
		'CI_369':'CI(2-1)369',\
		'CO32':'CO(3-2)',\
		'CO21':'CO(2-1)',\
		'CO10':'CO(1-0)'}

	return(line_dictionary[line])

def directory_checker(dirname):
    """ if a directory doesn't exist, then creates it """
    dirname =   str(dirname)
    if not os.path.exists(dirname):
        print("creating directory: %s" % dirname)
        try:
            os.mkdir(dirname)
        except:
            os.stat(dirname)

def directory_path_checker(pathway):
    """ checks that all the directories in a pathway exist; if they don't exist,
    then they are created."""

    # create and initialize list of indexes
    indexes =   []
    indexes.append( pathway.find('/') )
    index   =   indexes[0]

    # append index which marks the beginning of a new subdirectory
    while index >= 0:
        index   =   pathway.find('/',indexes[-1]+1)
        if index > 0: indexes.append(index)

    if indexes[0] == 0: indexes = indexes[1:]
    # run directory_checker for each directory in pathway
    for index in indexes:   directory_checker( pathway[:index] )

def check_version(module,version_required):

	version 		=	module.__version__

	for i,subversion in enumerate(version.split('.')):
		if int(subversion) < version_required[i]:
			print('\nActive version of module %s might cause problems...' % module.__name__)
			print('version detected: %s' % version)
			print('version required: %s.%s.%s' % (version_required[0],version_required[1],version_required[2]))
			break
		if i == len(version.split('.'))-1:
			print('\nNo version problems for %s module expected!' % module.__name__)
