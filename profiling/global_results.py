###     Module: global_results.py of SIGAME                   ###

# Import modules
import numpy as np
import pandas as pd
import pdb as pdb
import aux as aux
# import galaxy as gal
import cPickle as cPickle

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

params                      =   np.load('temp_params.npy').item()
for key,val in params.items():
    exec(key + '=val')


class global_results:
    '''
    An object referring to the global results of a selection of
    galaxies, containing global properties of as attributes.
    '''

    def __init__(self,**kwargs):

        args                    =   dict(z1=z1)
        args                    =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        try:
	        file_location       =   self.__get_GR_file_location(z1)
	        GR_dict             =   pd.read_pickle(file_location)
        except IOError:
            print('Searched for %s ' % file_location)
            print('No file with model results found, will create one')
            GR_dict             =   self.__create_GR_file(z1)

        # Set values in stored dictionary as attributes to global result object
        for key in GR_dict: setattr(self,key,GR_dict[key])

        # Search for and create attributes
        missing_attr            =   self.__find_missing_attr()

        # for key in missing_attr: setattr(self,key,missing_attr[key])
        self.__save_GR_file(self.__dict__,file_location)

    def print_results(self):

        print('\n ONLY SIMULATION STUFF')
        print('+%80s+' % ((5+20+15+15+10+10+10+10+8)*'-'))
        print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s|%10s' % ('Name'.center(5), 'sim name'.center(20), 'Stellar mass'.center(15), 'Gas mass'.center(15), 'SFR'.center(10), 'Sigma_SFR'.center(10), 'z'.center(10), 'Z_SFR'.center(10)))
        print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s|%10s' % (''.center(5), ''.center(20), '[10^9 Msun]'.center(15), '[10^9 Msun]'.center(15), '[Msun/yr]'.center(10), '[MW units]'.center(10), ''.center(10), '[solar]'.center(10)))
        print('+%80s+' % ((5+20+15+15+10+10+10+10+8)*'-'))
        for gal_index in range(0,len(self.galnames)):
            # if self.zreds[gal_index] == 0:
            print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s|%10s' % ('G'+str(gal_index+1),self.galnames[gal_index].center(20),\
                '{:.2e}'.format(self.M_star[gal_index]),\
                '{:.2e}'.format(self.M_gas[gal_index]),\
                '{:.4f}'.format(self.SFR[gal_index]),\
                '{:.4f}'.format(self.SFRsd[gal_index]/SFRsd_MW),\
                '{:.4f}'.format(self.zreds[gal_index]),\
                '{:.4f}'.format(self.Zsfr[gal_index])))
        print('+%80s+' % ((5+20+15+15+10+10+10+10+8)*'-'))

        # if z1 == 'z0':
        print('\n ISM PROPERTIES')
        print('+%80s+' % ((5+20+15+15+10+10+10+10+8)*'-'))
        print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s|%10s' % ('Name'.center(5), 'sim name'.center(20), 'D_L'.center(15), 'M_mol'.center(15), 'M_dif'.center(10), 'L_CII'.center(10), 'L_NII_205'.center(10), 'R_gal'.center(10)))
        print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s|%10s' % (''.center(5), ''.center(20), '[Mpc]'.center(15), '[Msun]'.center(15), '[Msun]'.center(10), '[L_sun]'.center(10), '[L_sun]'.center(10), '[kpc]'.center(10)))
        print('+%80s+' % ((5+20+15+15+10+10+10+10+8)*'-'))
        for gal_index in range(0,len(self.galnames)):
            # if self.zreds[gal_index] == 0:
            print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s|%10s' % ('G'+str(gal_index+1),self.galnames[gal_index].center(20),\
                '{:.4f}'.format(self.lum_dist[gal_index]),\
                '{:.2e}'.format(self.M_mol[gal_index]),\
                '{:.2e}'.format(self.M_dif[gal_index]),\
                '{:.2e}'.format(self.L_CII_tot[gal_index]/SFRsd_MW),\
                '{:.2e}'.format(self.L_NII_205_tot[gal_index]),\
                '{:.4f}'.format(self.R_gal[gal_index])))
        print('+%80s+' % ((5+20+15+15+10+10+10+10+8)*'-'))


        pdb.set_trace()

    def print_galaxy_properties(self,**kwargs):

        args        =   dict(gal_index=0)
        args        =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')


        print('\nProperties of Galaxy number %s, %s, at redshift %s' % (gal_index+1,self.galnames[gal_index],self.zreds[gal_index]))

        # Print these properties
        print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))
        print('|%20s|%20s|%15s|' % ('Property'.center(20), 'Value'.center(20), 'Name in code'.center(15)))
        print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))
        print('|%20s|%20s|%15s|' % ('Redshift'.center(20), '{:.3f}'.format(self.zreds[gal_index]), 'zred'.center(15)))
        print('|%20s|%20s|%15s|' % ('Radius'.center(20), '{:.3f}'.format(np.max(self.R_gal[gal_index])), 'R_gal'.center(15)))
        print('|%20s|%20s|%15s|' % ('Stellar mass'.center(20), '{:.3e}'.format(self.M_star[gal_index]), 'M_star'.center(15)))
        print('|%20s|%20s|%15s|' % ('ISM mass'.center(20), '{:.3e}'.format(self.M_gas[gal_index]), 'M_gas'.center(15)))
        print('|%20s|%20s|%15s|' % ('DM mass'.center(20), '{:.3e}'.format(self.M_dm[gal_index]), 'M_dm'.center(15)))
        print('|%20s|%20s|%15s|' % ('Dense gas mass'.center(20), '{:.3e}'.format(self.M_dense[gal_index]), 'M_dense'.center(15)))
        print('|%20s|%20s|%15s|' % ('SFR'.center(20), '{:.3f}'.format(self.SFR[gal_index]), 'SFR'.center(15)))
        print('|%20s|%20s|%15s|' % ('SFR surface density'.center(20), '{:.4f}'.format(self.SFRsd[gal_index]), 'SFRsd'.center(15)))
        print('|%20s|%20s|%15s|' % ('SFR-weighted Z'.center(20), '{:.4f}'.format(self.Zsfr[gal_index]), 'Zsfr'.center(15)))
        print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))

    def __get_GR_file_location(self,z1,**kwargs):

        file_location   =   d_temp+'global_results/'+z1+'_'+str(nGal)+'gals'+ext_DENSE+ext_DIFFUSE

        return file_location

    def __save_GR_file(self,GR_dict,file_location):

        # GR_df           =   pd.DataFrame(GR_dict)

        cPickle.dump(GR_dict,open(file_location,'wb'))

    def __create_GR_file(self,z1,**kwargs):

        return GR_file

    def __find_missing_attr(self):
		# Check for list of attributes, and call __set_attr() if missing

		attributes      =   ['lum_dist','N_gal','M_mol','M_dif','L_CII_tot','L_NII_205_tot']

		for attr in attributes:
			if not hasattr(self, attr): attr_values     =   self.__set_attr(attr)
			# attr_values         =   self.__set_attr(attr)
			# pdb.set_trace()
			# print(attr,attributes)
		# print('hey')

    def __set_attr(self,attr):
        # Get missing attributes

        if attr == 'lum_dist':
            self.lum_dist           =  	aux.get_lum_dist(self.zreds.values)

        if attr == 'N_gal':
            self.N_gal              =   len(self.galnames)

        if attr == 'M_mol':
            M_mol                   =   np.zeros(len(self.galnames))
            for gal_index in range(0,len(self.galnames)):
                gal_obj             =   gal.galaxy(gal_index=gal_index,silent=True)
                GMCgas              =   gal_obj.particle_data.get_raw_data(data='ISM')['GMC']
                if type(GMCgas) != int: M_mol[gal_index] = np.sum(GMCgas['m'])
            self.M_mol              =    M_mol

        if attr == 'M_dif':
            M_dif                   =   np.zeros(len(self.galnames))
            for gal_index in range(0,len(self.galnames)):
                gal_obj             =   gal.galaxy(gal_index=gal_index,silent=True)
                difgas              =   gal_obj.particle_data.get_raw_data(data='ISM')['dif']
                if type(difgas) != int: M_dif[gal_index] = np.sum(difgas['m'])
            self.M_dif              =    M_dif

        if attr == 'L_CII_tot':
            L_CII_tot               =   np.zeros(len(self.galnames))
            for gal_index in range(0,len(self.galnames)):
                gal_obj             =   gal.galaxy(gal_index=gal_index,silent=True)
                dc_summed           =   gal_obj.datacube.get_dc_summed(target='L_CII')['datacube']
                if type(dc_summed) != int:
                    mom0            =   dc_summed.sum(axis=0)*v_res # Jy*km/s
                    L_CII_tot[gal_index]    =   np.sum(mom0)
            self.L_CII_tot          =   L_CII_tot

        if attr == 'L_NII_205_tot':
            L_NII_205_tot           =   np.zeros(len(self.galnames))
            for gal_index in range(0,len(self.galnames)):
                gal_obj             =   gal.galaxy(gal_index=gal_index,silent=True)
                dc_summed           =   gal_obj.datacube.get_dc_summed(target='L_NII_205')['datacube']
                if type(dc_summed) != int:
                    mom0                        =   dc_summed.sum(axis=0)*v_res # Jy*km/s
                    L_NII_205_tot[gal_index]    =   np.sum(mom0)
            self.L_NII_205_tot                  =   L_NII_205_tot
