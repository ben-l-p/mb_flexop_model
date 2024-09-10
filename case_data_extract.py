import numpy as np
import pandas
from create_multibody_flexop import FlexopAeroelastic
from time import time


def case_data_extract(model: FlexopAeroelastic, case_data):
    dict_out = {'x': model.x_local,
                'y': model.y_local,
                'z': model.z_local,
                'tip_angle': model.rigid_sweep_ang,
                'u_inf': model.input_settings['u_inf'],
                'rho': model.input_settings['rho'],
                'gust_length': model.input_settings['gust_length'],
                'n_tstep': model.input_settings['n_tstep'],
                'dt': model.input_settings['dt'],
                'num_node': model.num_node,
                'num_node_tot': model.num_node['total'],
                'num_elem': model.num_elem,
                'num_elem_tot': model.num_elem['total'],
                'num_node_elem': model.num_node_elem,
                'flow': model.input_settings['flow'],
                'case_name': model.case_name,
                'case_route': model.case_route,
                'total_time': time() - model.start_time,
                'case_run': 1,
                }

    n_tstep = model.input_settings['n_tstep']

    beam_pos = np.zeros([n_tstep, dict_out['num_node_tot'], 3])
    psi = np.zeros([n_tstep] + list(case_data.structure.timestep_info[0].psi.shape))
    for i_ts in range(n_tstep):
        beam_pos[i_ts, :, :] = case_data.structure.timestep_info[i_ts].pos
        psi[i_ts, :, :, :] = case_data.structure.timestep_info[i_ts].psi

    beam_pos_init = case_data.structure.ini_info.pos
    psi_init = case_data.structure.ini_info.psi

    dict_out.update({'beam_pos': beam_pos, 'beam_pos_init': beam_pos_init, 'psi_init': psi_init, 'psi': psi})

    zeta = [np.zeros((n_tstep, *surf.shape)) for surf in case_data.aero.timestep_info[0].zeta]
    gamma = [np.zeros((n_tstep, *surf.shape)) for surf in case_data.aero.timestep_info[0].gamma]
    zeta_star = [np.zeros((n_tstep, *surf.shape)) for surf in case_data.aero.timestep_info[0].zeta_star]
    gamma_star = [np.zeros((n_tstep, *surf.shape)) for surf in case_data.aero.timestep_info[0].gamma_star]
    n_surf = len(case_data.aero.timestep_info[0].zeta)

    for i_t in range(n_tstep):
        for i_s in range(n_surf):
            zeta[i_s][i_t, :, :] = case_data.aero.timestep_info[i_t].zeta[i_s]
            gamma[i_s][i_t, :, :] = case_data.aero.timestep_info[i_t].gamma[i_s]
            zeta_star[i_s][i_t, :, :] = case_data.aero.timestep_info[i_t].zeta_star[i_s]
            gamma_star[i_s][i_t, :, :] = case_data.aero.timestep_info[i_t].gamma_star[i_s]

    dict_out.update({'zeta': zeta, 'gamma': gamma, 'zeta_star': zeta_star, 'gamma_star': gamma_star, 'psi': psi})

    if 'Modal' in model.input_settings['flow']:
        mode_freqs = case_data.structure.timestep_info[0].modal['freq_natural']
        eigenvalues = case_data.structure.timestep_info[0].modal['eigenvalues']
        eigenvectors = case_data.structure.timestep_info[0].modal['eigenvectors']
        m = case_data.structure.timestep_info[0].modal['M']
        k = case_data.structure.timestep_info[0].modal['K']

        modal_dict = {'mode_freqs': mode_freqs, 'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors, 'M': m,
                      'K': k}
        dict_out.update({"Modal": modal_dict})

    if 'AeroForcesCalculator' in model.input_settings['flow']:
        force_data = pandas.read_csv('./output/%s/forces/forces_aeroforces.txt' % model.case_name, delimiter=', ',
                                     index_col=False).to_dict()
        moment_data = pandas.read_csv('./output/%s/forces/moments_aeroforces.txt' % model.case_name, delimiter=', ',
                                      index_col=False).to_dict()

        forces_a_s = np.zeros([n_tstep, 3])
        forces_g_s = np.zeros([n_tstep, 3])
        moments_a_s = np.zeros([n_tstep, 3])
        moments_g_s = np.zeros([n_tstep, 3])
        forces_a_u = np.zeros([n_tstep, 2])
        forces_g_u = np.zeros([n_tstep, 3])
        moments_a_u = np.zeros([n_tstep, 3])
        moments_g_u = np.zeros([n_tstep, 3])

        for i_ts in range(n_tstep):
            forces_a_s[i_ts, :] = [force_data['fx_steady_a'][i_ts], force_data['fy_steady_a'][i_ts],
                                   force_data['fz_steady_a'][i_ts]]
            forces_g_s[i_ts, :] = [force_data['fx_steady_G'][i_ts], force_data['fy_steady_G'][i_ts],
                                   force_data['fz_steady_G'][i_ts]]
            moments_a_s[i_ts, :] = [moment_data['mx_steady_a'][i_ts], moment_data['my_steady_a'][i_ts],
                                    moment_data['mz_steady_a'][i_ts]]
            moments_g_s[i_ts, :] = [moment_data['mx_steady_G'][i_ts], moment_data['my_steady_G'][i_ts],
                                    moment_data['mz_steady_G'][i_ts]]
            forces_a_u[i_ts, :] = [force_data['fx_unsteady_a'][i_ts], force_data['fy_unsteady_a'][i_ts]]
            forces_g_u[i_ts, :] = [force_data['fx_unsteady_G'][i_ts], force_data['fy_unsteady_G'][i_ts],
                                   force_data['fz_unsteady_G'][i_ts]]
            moments_a_u[i_ts, :] = [moment_data['mx_unsteady_a'][i_ts], moment_data['my_unsteady_a'][i_ts],
                                    moment_data['mz_unsteady_a'][i_ts]]
            moments_g_u[i_ts, :] = [moment_data['mx_unsteady_G'][i_ts], moment_data['my_unsteady_G'][i_ts],
                                    moment_data['mz_unsteady_G'][i_ts]]

        force_dict = {"moments_a_s": moments_a_s,
                      "moments_g_s": moments_g_s,
                      "forces_a_s": forces_a_s,
                      "forces_g_s": forces_g_s,
                      "moments_a_u": moments_a_u,
                      "moments_g_u": moments_g_u,
                      "forces_a_u": forces_a_u,
                      "forces_g_u": forces_g_u}
        dict_out.update({'AeroForcesCalculator': force_dict})

    if 'BeamLoads' in model.input_settings['flow']:
        steady_applied_forces = np.zeros((n_tstep, *case_data.structure.timestep_info[0].steady_applied_forces.shape))
        unsteady_applied_forces = np.zeros((n_tstep, *case_data.structure.timestep_info[0].steady_applied_forces.shape))

        for i_ts in range(n_tstep):
            steady_applied_forces[i_ts, ...] = case_data.structure.timestep_info[i_ts].steady_applied_forces
            unsteady_applied_forces[i_ts, ...] = case_data.structure.timestep_info[i_ts].unsteady_applied_forces

        dict_out.update({'BeamLoads': {'steady_applied_forces': steady_applied_forces,
                                       'unsteady_applied_forces': unsteady_applied_forces}})

    return dict_out
