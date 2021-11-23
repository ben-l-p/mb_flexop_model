#! /usr/bin/env python3
import h5py as h5
import numpy as np
import pandas as pd
   


class FLEXOPFuselage:
    def __init__(self, m, structure, case_name, case_route, **kwargs):
        """
        
        Key-Word Arguments:
        """
        self.m = m
        self.structure = structure

        self.route = case_route
        self.case_name = case_name

    def generate(self): 

        structure = self.structure

        self.n_elem = structure.self.n_elem
        self.n_node_elem = structure.self.n_node_elem
        self.n_node = structure.self.n_node
        self.n_elem_main = structure.self.n_elem_main
        self.n_node_main = structure.self.n_node_main
        self.n_elem_fuselage = structure.self.n_elem_fuselage
        self.n_node_fuselage = structure.self.n_node_fuselage
        self.n_elem_tail = structure.self.n_elem_tail
        self.n_node_tail = structure.self.n_node_tail


        nonlifting_body_node = np.zeros((self.n_node,), dtype=bool)
        nonlifting_body_distribution = np.zeros((self.n_elem,), dtype=int) - 1
        nonlifting_body_m = np.zeros((self.n_nonlifting_bodies, ), dtype=int)
        radius = np.zeros((self.n_node,))
        a_ellipse = np.zeros((self.n_node,))
        b_ellipse = np.zeros((self.n_node,))
        z_0_ellipse = np.zeros((self.n_node,))
        radius = np.zeros((self.n_node,))
        we = 0
        wn = 0

        # right wing
        nonlifting_body_node[wn:wn + self.n_node_main] = False
        we += self.n_elem_main
        wn += self.n_node_main

        # left wing
        nonlifting_body_node[wn:wn + self.n_node_main] = False
        we += self.n_elem_main
        wn += self.n_node_main -1

        #fuselage (beam?, body ID = 0)
        i_body = 0
        
        nonlifting_body_node[0] = True
        nonlifting_body_node[wn:wn + self.n_node_fuselage-1] = True
        nonlifting_body_distribution[we:we + self.n_elem_fuselage] = i_body
        nonlifting_body_m[i_body] = self.m
        #radius[wn:wn + self.n_node_fuselage] = get_ellipsoidal_geometry(x[wn:wn + self.n_node_fuselage], thickness_ratio_ellipse,0) #np.genfromtxt('radius_wanted.csv',delimiter=',')
        # radius_fuselage = create_fuselage_geometry()
        x_coord_fuselage = np.sort(self.structure.x[nonlifting_body_node])
        idx_junction = self.find_index_of_closest_entry(x_coord_fuselage, self.structure.x[0])
        x_coord_fuselage += abs(min(x_coord_fuselage))
        a_ellipse_tmp, b_ellipse_tmp, z_0_ellipse_tmp = self.generate_fuselage_geometry(x_coord_fuselage)
        a_ellipse[0] = a_ellipse_tmp[idx_junction]
        b_ellipse[0] = b_ellipse_tmp[idx_junction]
        z_0_ellipse[0] = z_0_ellipse_tmp[idx_junction]


        a_ellipse_tmp= np.delete(a_ellipse_tmp,idx_junction)
        b_ellipse_tmp= np.delete(b_ellipse_tmp,idx_junction)
        z_0_ellipse_tmp= np.delete(z_0_ellipse_tmp,idx_junction)
        a_ellipse[wn:wn + self.n_node_fuselage-1] =  a_ellipse_tmp
        b_ellipse[wn:wn + self.n_node_fuselage-1] =  b_ellipse_tmp
        z_0_ellipse[wn:wn + self.n_node_fuselage-1] =  z_0_ellipse_tmp
        
        with h5.File(self.route + '/' + self.case_name + '.nonlifting_body.h5', 'a') as h5file:
            h5file.create_dataset('shape', data='specific')
            h5file.create_dataset('a_ellipse', data=a_ellipse)
            h5file.create_dataset('b_ellipse', data=b_ellipse)
            h5file.create_dataset('z_0_ellipse', data=z_0_ellipse)
            h5file.create_dataset('surface_m', data=nonlifting_body_m)
            h5file.create_dataset('nonlifting_body_node', data=nonlifting_body_node)

            h5file.create_dataset('surface_distribution', data=nonlifting_body_distribution)
            
            # radius
            radius_input = h5file.create_dataset('radius', data=radius)
            radius_input.attrs['units'] = 'm'

    def find_index_of_closest_entry(self, array_values, target_value):
        return (np.abs(array_values - target_value)).argmin()

    def generate_fuselage_geometry(self, x_coord_fuselage):
        df_fuselage = pd.read_csv('../01_case_files/flexOp_data/fuselage_geometry.csv', sep=";")
        y_coord_fuselage = self.interpolate_fuselage_geometry(x_coord_fuselage, df_fuselage, 'y', True)
        z_coord_fuselage_upper = self.interpolate_fuselage_geometry(x_coord_fuselage, df_fuselage, 'z', True)
        z_coord_fuselage_lower = self.interpolate_fuselage_geometry(x_coord_fuselage, df_fuselage, 'z', False)
        b_ellipse_tmp = (np.array(z_coord_fuselage_upper) - np.array(z_coord_fuselage_lower))/2.
        z_0_ellipse_tmp = b_ellipse_tmp - abs(np.array(z_coord_fuselage_lower))
        return y_coord_fuselage, b_ellipse_tmp, z_0_ellipse_tmp

    def interpolate_fuselage_geometry(self, x_coord_beam, df_fuselage, coord, upper_surface=True):
        if coord == 'y':
            
            df_fuselage = df_fuselage.iloc[:,:2].dropna()
            
        else: 
            df_fuselage = df_fuselage.iloc[:,2:].dropna()
            first_and_last_row_df = df_fuselage.iloc[[0, -1]]
            if upper_surface:
                df_fuselage = df_fuselage[df_fuselage.iloc[:,1]>0.0]
            else:    
                df_fuselage= df_fuselage[df_fuselage.iloc[:,1]<0.0]
            df_fuselage = pd.concat([first_and_last_row_df, df_fuselage]).drop_duplicates()
            df_fuselage = df_fuselage.sort_values(df_fuselage.columns[0])
        y = []
        for x in  x_coord_beam:
            if x in df_fuselage.iloc[:,0].tolist():
                y.append(df_fuselage[df_fuselage.iloc[:,0] == x].iloc[0,1])
            else:
                values_adjacent_right = df_fuselage[df_fuselage.iloc[:,0] >= x].iloc[0, :]
                values_adjacent_left = df_fuselage[df_fuselage.iloc[:,0] <= x].iloc[-1, :]
                x_known = [values_adjacent_right.iloc[0], values_adjacent_left.iloc[0]]
                y_known = [values_adjacent_right.iloc[1], values_adjacent_left.iloc[1]]

                y.append(y_known[0]+ (x-x_known[0])/(x_known[1]- x_known[0])*(y_known[1]-y_known[0]))
        return y