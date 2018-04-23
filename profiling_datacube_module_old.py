from __future__ import print_function

# coding=utf-8
###     For profiling datacube.py of SIGAME             	###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb
import os as os
import scipy as scipy
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import aux as aux
# import plot as plot
import time
import multiprocessing as mp
from pathlib2 import Path

params                      =   np.load('params.npy').item()
for key,val in params.items():
    exec(key + '=val')

d_temp = '/Users/Karen/code/SIGAME_dev/sigame/temp/'

def mk_datacube(galname=[''], SFRsd=SFRsd_MW, igalnum=0, zred=[], los=[0,0,1], overwrite=False, targets=target):
	'''
	Purpose
	-------
	Create datacube of galaxy in velocity and space

	What this function does
	---------
	1) Create radial profiles of all model clouds
	2) Position clouds on grid and save resulting matrix

	Arguments
	---------
	igalnum: galaxy index - int
	default = 0; first galaxy name in galnames list from parameter file

	v_res: velocity resolution [km/s] - float/int
	default = 10

	x_res_pc: spatial resolution [pc] - float/int
	default = 1,000

	v_max: max velocity [km/s] - float/int
	default = 500

	x_max_pc: max spatial extend [pc] - float/int
	default = 10,000

	los: Line Of Sight directions - list
	default = [1,0,0] = z direction

	target: what are we putting into the datacube? - str
	options:
		- line emission: 'L_CII'
		- gas mass: 'mass'

	'''

	plt.close('all')

	print('\n** Creating datacubes with the following settings: **\n')

	global clouds
    
	# Calculate FUV field strength
	FUV_list         	=   [5,35]
	FUV                 =   str(int(FUV_list[np.argmin(np.abs(np.array(FUV_list)-SFRsd/SFRsd_MW))]))

	print('Galaxy: %s' % galname)
	print('FUV field: %s x MW FUV' % FUV)
	print('Target: %s' % targets)
	print('Size of galaxy image: +/-%s kpc' % (x_max_pc/1000.))
	print('Size of velocity axis: +/-%s km/s' % (v_max))
	print('Spatial resolution: %s pc' % x_res_pc)
	print('Velocity resolution: %s km/s' % v_res)

	dc_sum_Lsun 		=	{}
	for target in targets:
		dc_sum_Lsun[target+'_GMC_dc'] 	=	[0]
		dc_sum_Lsun[target+'_DNG_dc'] 	=	[0]
		dc_sum_Lsun[target+'_DIG_dc'] 	=	[0]

	for target in targets:

		print('\nNow creating datacubes of %s' % target)

		# Create velocity and position axes
		v_axis 				=	np.arange(-v_max,v_max+v_max/1e6,v_res)
		x_axis 				=	np.arange(-x_max_pc,x_max_pc+x_max_pc/1e6,x_res_pc)
		dv 					=	v_axis[1]-v_axis[0]
		dx 					=	x_axis[1]-x_axis[0]

		# Move axes to center value of each pixel
		v_axis 				=	v_axis[0:len(v_axis)-1]+v_res/2.
		x_axis 				=	x_axis[0:len(x_axis)-1]+x_res_pc/2.

		print('\n1) Create (if not there already) radial profiles of all model clouds')
		if not os.path.exists(d_temp+'datacubes/cloud_profiles/GMC/%s_radial_profiles_%s.npy' % (target,z1)): mk_cloud_rad_profiles(ISM_phase='GMC', target=target, FUV=FUV)
		for ISM_phase in ['DNG','DIG']:
			if not os.path.exists(d_temp+'datacubes/cloud_profiles/%s/%s_radial_profiles_%s_%sUV.npy' % (ISM_phase,target,z1,FUV)): mk_cloud_rad_profiles(ISM_phase=ISM_phase, target=target, FUV=FUV)

		print('\n2) Load and position clouds on grids for GMCs, DNG and DIG and save in d_temp + datacubes/%s/' % target)

		if los == [1,0,0]: inc = 'x'
		if los == [0,1,0]: inc = 'y'
		if los == [0,0,1]: inc = 'z'

		for ISM_phase in ['DNG','DIG']:
			filename = d_temp+'datacubes/%s/%s/%s_x%s_v%s_i%s_%s.npy' % (target, ISM_phase, z1, int(x_res_pc), int(v_res), inc, galname)
			path = Path(filename)
			if path.exists():
				print(ISM_phase+' datacube already made!')
			if (not path.exists()) or (overwrite):
				t1 = time.clock()
				clouds 								=	load_clouds(galname=galname,zred=zred,igalnum=igalnum,ISM_phase=ISM_phase,target=target,los=los)
				dc_Jy,dc_sum_Lsun[target+'_'+ISM_phase+'_dc'] 	=	drizzle(v_axis,x_axis,ISM_phase=ISM_phase,target=target,FUV=FUV,zred=zred)
				dc_dictionary 					=	{'datacube':dc_Jy,'x_axis':x_axis,'v_axis':v_axis}
				# np.save(filename, dc_dictionary)
				t2 = time.clock()
				dt = t2-t1
				if dt < 60: print('Time it took to do this ISM phase: %.2f s' % (t2-t1))
				if dt > 60: print('Time it took to do this ISM phase: %.2f min' % ((t2-t1)/60.))
	print('\ndone with this galaxy!')

	# pdb.set_trace()
	# return(dc_sum_Lsun)

@profile
def drizzle(v_axis,x_axis,ISM_phase,target,plotting=True,verbose=False,checkplot=False,FUV='5',zred=0):
	'''
	Purpose
	---------
	Drizzle *all* clouds onto galaxy grid in velocity and space

	'''

	N_clouds 				=	len(clouds)
	print('\nNow drizzling %s %s clouds onto galaxy grid at %s x MW FUV...' % (N_clouds,ISM_phase,FUV))	

	# Empty numpy array to hold result
	lenv,lenx 				=	len(v_axis),len(x_axis)
	result					=	np.zeros([lenv,lenx,lenx])

	# For reference, get total values of "target" from interpolation
	if ISM_phase in ['DNG','DIG']: interpolation_result 	=	clouds[target+'_'+ISM_phase].values
	if ISM_phase in ['GMC']: interpolation_result 		=	clouds[target].values

	# LOADING
	# Get radial surface brightness profiles for all model clouds
	if ISM_phase in ['DNG','DIG']: model_rad_profs 	=   np.load(d_temp+'datacubes/cloud_profiles/%s/%s_radial_profiles_%s_%sUV.npy' % (ISM_phase,target,z1,FUV))
	if ISM_phase in ['GMC']: model_rad_profs 		=   np.load(d_temp+'datacubes/cloud_profiles/%s/%s_radial_profiles_%s.npy' % (ISM_phase,target,z1))
	models_r_pc				=	model_rad_profs[1,:,:] 	# pc
	models_SB 				=	model_rad_profs[0,:,:] 	# Lsun/pc^2
	# Assign these models to clouds in the galaxy:
	model_index 			=	clouds['closest_model_i'].values
	clouds_r_pc 			=	[models_r_pc[int(i)] for i in model_index]
	clouds_SB 				=	[models_SB[int(i)] for i in model_index]
	clouds_R_pc 			=	np.max(clouds_r_pc,axis=1) 
	vel_disp_gas 			=	clouds['vel_disp_gas'].values
	v_proj 					=	clouds['v_proj'].values
	x_cloud 				=	clouds['x'].values
	y_cloud 				=	clouds['y'].values

	# SETUP
	# Some things we will need
	v_max, v_res 			=	max(v_axis),v_axis[1]-v_axis[0]
	fine_v_axis 			=	np.arange(-v_max,v_max+v_max/1e6,v_res/8.)
	x_res_pc 				=	x_axis[1]-x_axis[0]
	npix_highres 			=	9 # highres pixels per galaxy pixel, giving 3 pixels on either side of central pixel (of resolution x_res_pc)
	x_res_pc_highres 		=	x_res_pc/npix_highres # size of high resolution pixel
	pix_area_highres_pc 	=	x_res_pc_highres**2. # area of high resolution pixel

	# What galaxy pixel center comes closest to this cloud center?
	min_x 					= 	np.min(x_axis)
	range_x 				= 	np.max(x_axis) - min_x
	x_index 				=	np.round((x_cloud-min_x)/range_x*(lenx-1))
	y_index 				=	np.round((y_cloud-min_x)/range_x*(lenx-1))

	# pdb.set_trace()

	# ----------------------------------------------
	# SMALL CLOUDS
	# Clouds that are "very" small compared to pixel size in galaxy image
	# give all their luminosity to one galaxy pixel
	small_cloud_index 		=	[(0 < clouds_R_pc) & (clouds_R_pc <= x_res_pc/8.)][0]
	print('%s small clouds, unresolved by galaxy pixels' % (len(small_cloud_index[small_cloud_index == True])))
	small_cloud_x_index 	=	np.extract(small_cloud_index,x_index)
	small_cloud_y_index 	=	np.extract(small_cloud_index,y_index)
	small_cloud_targets 	=	np.extract(small_cloud_index,clouds[target].values)
	small_cloud_vdisp_gas 	=	np.extract(small_cloud_index,vel_disp_gas)
	small_cloud_v_proj 		=	np.extract(small_cloud_index,v_proj)
	for target1,vel_disp_gas1,v_proj1,i_x,i_y in zip(small_cloud_targets,small_cloud_vdisp_gas,small_cloud_v_proj,small_cloud_x_index,small_cloud_y_index):
		vel_prof 				=		mk_cloud_vel_profile(v_proj1,vel_disp_gas1,fine_v_axis,v_axis)
		result[:,i_x,i_y] 		+=		vel_prof*target1

	# ----------------------------------------------
	# LARGER CLOUDS
	# Cloud is "NOT very" small compared to pixel size in galaxy image
	# => resolve cloud and split into nearby pixels
	large_cloud_index 		=	[(0 < clouds_R_pc) & (clouds_R_pc > x_res_pc/8.)][0] # boolean array
	print('%s large clouds, resolved by galaxy pixels' % (len(large_cloud_index[large_cloud_index == True])))
	Npix_highres 			=	2.2*np.extract(large_cloud_index,clouds_R_pc)/x_res_pc_highres 
	Npix_highres 			=	np.ceil(Npix_highres).astype(int)
	# Count highres cloud pixels in surrounding galaxy pixels
	max_pix_dif 			=	np.extract(large_cloud_index,clouds_R_pc)/x_res_pc
	max_pix_dif 			=	np.ceil(max_pix_dif).astype(int)
	highres_axis_max 		=	(np.array(Npix_highres)*x_res_pc_highres)/2.
	large_cloud_interpol 	= 	np.extract(large_cloud_index,interpolation_result)
	large_cloud_model_index =	np.extract(large_cloud_index,model_index)
	large_cloud_x_index 	=	np.extract(large_cloud_index,x_index)
	large_cloud_y_index 	=	np.extract(large_cloud_index,y_index)
	large_cloud_targets 	=	np.extract(large_cloud_index,clouds[target].values)
	large_cloud_vdisp_gas 	=	np.extract(large_cloud_index,vel_disp_gas)
	large_cloud_v_proj 		=	np.extract(large_cloud_index,v_proj)
	large_cloud_R_pc 		=	np.extract(large_cloud_index,clouds_R_pc)
	large_models_r_pc 		=	[models_r_pc[int(_)] for _ in large_cloud_model_index]
	large_models_SB 		=	[models_SB[int(_)] for _ in large_cloud_model_index]
	i 						=	0
	for target1,vel_disp_gas1,v_proj1,i_x,i_y in zip(large_cloud_targets,large_cloud_vdisp_gas,large_cloud_v_proj,large_cloud_x_index,large_cloud_y_index):
		vel_prof 				=		mk_cloud_vel_profile(v_proj1,vel_disp_gas1,fine_v_axis,v_axis)
		# create grid of coordinates for high-res image of cloud:
		x_highres_axis 		=	np.linspace(-highres_axis_max[i],highres_axis_max[i],Npix_highres[i])
		x_highres_mesh, y_highres_mesh = np.mgrid[slice(-max(x_highres_axis) + x_res_pc_highres/2., max(x_highres_axis) - x_res_pc_highres/2. + x_res_pc_highres/1e6, x_res_pc_highres),\
						slice(-max(x_highres_axis) + x_res_pc_highres/2., max(x_highres_axis) - x_res_pc_highres/2. + x_res_pc_highres/1e6, x_res_pc_highres)]
		radius 				=	np.sqrt(x_highres_mesh**2+y_highres_mesh**2)
		# create high-res image of this cloud, evaluating surface brightness at the center of each high-res pixel
		interp_func_r      	=   interp1d(large_models_r_pc[i],large_models_SB[i],fill_value=large_models_SB[i][0],bounds_error=False)
		im_cloud 			=	interp_func_r(radius)
		im_cloud[radius > large_cloud_R_pc[i]] 	= 	0.
		# Convert image to luminosity units [Lsun/pc^2 -> Lsun]
		im_cloud 			=	im_cloud*pix_area_highres_pc

		# Normalize to total luminosity of this cloud:
		im_cloud 			=	im_cloud*interpolation_result[i]/np.sum(im_cloud)

		# Identify low rew pixels that we will be filling up
		x_indices 			=	[large_cloud_x_index[i] + _ for _ in np.arange(-max_pix_dif[i],max_pix_dif[i]+1)]
		y_indices 			=	[large_cloud_y_index[i] + _ for _ in np.arange(-max_pix_dif[i],max_pix_dif[i]+1)]
		x_index_mesh, y_index_mesh = np.mgrid[slice(x_indices[0], x_indices[-1] + 1./1e6, 1),\
			slice(y_indices[0], y_indices[-1] + 1./1e6, 1)]
		x_index_array 		=	x_index_mesh.reshape(len(x_indices)**2)
		y_index_array 		=	y_index_mesh.reshape(len(y_indices)**2)
		i_highres_center 	=	float(Npix_highres[i])/2.-0.5

		for x_i,y_i in zip(x_index_array,y_index_array):
			x_i_highres 		=	int(i_highres_center + (int(x_i) - large_cloud_x_index[i] ) * npix_highres - (npix_highres-1)/2.)
			y_i_highres 		=	int(i_highres_center + (int(y_i) - large_cloud_y_index[i] ) * npix_highres - (npix_highres-1)/2.) 
			try:
				result[:,int(x_i),int(y_i)] 	+=	vel_prof*np.sum(im_cloud[x_i_highres:x_i_highres+npix_highres+1, y_i_highres:y_i_highres+npix_highres+1])
			except:
				print('highres cloud image went outside galaxy image')
				pass

		if verbose:
			print('check')
			print('%.2e Lsun from drizzled image' % (np.sum(one_cloud_drizzled)))
			if ISM_phase == 'GMC':
				models 			=   pd.read_pickle('cloudy_models/GMC/grids/GMCgrid'+ ext_DENSE + '_' + z1+'_em.models')
			if ISM_phase in ['DNG','DIG']:
				models 			=   pd.read_pickle('cloudy_models/dif/grids/difgrid_'+FUV+'UV'+ ext_DIFFUSE + '_' + z1+'_em.models')
				if ISM_phase == 'DNG': 
					print('%.2e Lsun from cloudy grid' % (models['L_CII'][large_cloud_model_index[i]]*models['f_CII_DNG'][large_cloud_model_index[i]]))
				if ISM_phase == 'DIG': 
					print('%.2e Lsun from cloudy grid' % (models['L_CII'][large_cloud_model_index[i]]*(1-models['f_CII_DNG'][large_cloud_model_index[i]])))
			print(large_cloud_interpol[i])

		# Normalize to total luminosity of this cloud:
		# print('\n')
		# print(interpolation_result[i])
		# print(np.sum(one_cloud_drizzled))
		# print(np.sum(im_cloud))
		# pdb.set_trace()
		# one_cloud_drizzled 	=	one_cloud_drizzled*interpolation_result[i]/np.sum(one_cloud_drizzled)

		# result 				+=	one_cloud_drizzled

		i 					+=	1
		# result[:,i_x,i_y] 		+=		vel_prof*target1
		if i == 100: break

	# ----------------------------------------------
	# CONVERSION TO JY
	# convert to Jy = ergs/s / m^2 / Hz (observed frequency right?)
	tot_Lsun 			=	np.sum(result)
	dc_Jy         		=   np.nan_to_num(result)
	if target != 'mass':
		print('%.2f Lsun' % tot_Lsun)
		D_L 					=	aux.luminosity_distance(zred)
		print('Max in Lsun: %.2f ' % np.max(dc_Jy))
		freq_obs 				=	params['f_'+target.replace('L_','')]*1e9/(1+zred)
		dc_Jy 					=	dc_Jy*Lsun/(1e-26*v_res*1e3*freq_obs/clight*(4*np.pi*(D_L*1e6*pc2m)**2))
	print('Max in Jy: %.2f ' % np.max(dc_Jy))


	return(dc_Jy,tot_Lsun)



def load_clouds(galname,zred,igalnum,ISM_phase,target,los):
	'''
	Purpose
	---------
	Load clouds from saved galaxy files, and convert to a similar format.

	'''

	print('\nNow loading %s clouds in galaxy %s (number %s...)' % (ISM_phase,galname,igalnum+1))

	if ISM_phase == 'GMC': file = d_temp+'GMC/emission/'+ext_DENSE+'/z'+'{:.2f}'.format(zred)+'_'+galname+'_GMC'+ext_DENSE+'_em.gas'
	if ISM_phase == 'DIG': file = d_temp+'dif/emission/'+ext_DIFFUSE+'/z'+'{:.2f}'.format(zred)+'_'+galname+'_dif'+ext_DIFFUSE+'_em.gas'
	if ISM_phase == 'DNG': file = d_temp+'dif/emission/'+ext_DIFFUSE+'/z'+'{:.2f}'.format(zred)+'_'+galname+'_dif'+ext_DIFFUSE+'_em.gas'

	clouds 				=	pd.read_pickle(file)
	clouds1 			=	clouds.copy()


	if ISM_phase == 'DIG':
		clouds1[target]		=	clouds[target+'_DIG'].values

	if ISM_phase == 'DNG':
		clouds1[target]		=	clouds[target+'_DNG'].values

	if ISM_phase == 'GMC':
		clouds1['R']		=	clouds['Rgmc'].values

	clouds1['x'] 		=	clouds['x']*1000. # pc
	clouds1['y'] 		=	clouds['y']*1000. # pc

	if los == [0,0,1]: clouds1['v_proj'] 	=	clouds['vz'] # km/s

	# Cut out only what's inside image:
	radius_pc 			=	np.sqrt(clouds1['x']**2 + clouds1['y']**2)
	clouds1 			=	clouds1[radius_pc < x_max_pc]
	clouds1 			=	clouds1.reset_index(drop=True)

	# TEST!!
	# clouds1 			=	clouds1.iloc[0:2]
	# clouds1['x'][0] 	=	0
	# clouds1['y'][0] 	=	0

	return(clouds1)

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
		Gaussian 				=	1./np.sqrt(2*np.pi*vel_disp_gas**2) * np.exp( -(fine_v_axis-v_proj)**2 / (2*vel_disp_gas**2) )

		# Numerical integration over velocity axis bins
		v_res 					=	(v_axis[1]-v_axis[0])/2.

		vel_prof 				=	abs(np.array([integrate.trapz(fine_v_axis[(fine_v_axis >= v-v_res) & (fine_v_axis < v+v_res)], Gaussian[(fine_v_axis >= v-v_res) & (fine_v_axis < v+v_res)]) for v in v_axis]))

		if plotting:
			plot.simple_plot(fig=2,
				fontsize=12,xlab='v [km/s]',ylab='F [proportional to Jy]',\
				x1=fine_v_axis,y1=Gaussian,col1='r',ls1='--',\
				x2=v_axis,y2=vel_prof,col2='b',ls2='-.')

			plt.show(block=False)

		# Normalize that profile
		vel_prof 				=	vel_prof*1./np.sum(vel_prof)

	else:

		vel_prof 				=	v_axis*0.

		vel_prof[aux.find_nearest_index(v_axis,v_proj)] = 1.

	return(vel_prof)

mk_datacube(galname='h10_s45_G1',igalnum=0,zred=6.125,targets=['L_CII'])
