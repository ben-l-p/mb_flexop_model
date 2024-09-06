import os
import shutil
import numpy as np

import flexop

# Simulation inputs
case_name = 'flexop_orig'
case_route = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/cases/'
case_out_folder = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/output/'

u_inf = 40.
u_inf_dir = np.array((1., 0., 0.))
m = 8
m_star_fact = 2
physical_time = 2.
c_ref = 0.35
rho = 1.225
dt = c_ref / (m * u_inf)
n_tstep = int(physical_time / dt)

# Remove old case files and outputs
try:
    shutil.rmtree(case_out_folder)
    shutil.rmtree(case_route)
except:
    pass

try:
    os.makedirs(case_out_folder)
except FileExistsError:
    pass

try:
    os.makedirs(case_out_folder)
except FileExistsError:
    pass

settings = dict()
settings['SHARPy'] = {
    'flow': ['BeamLoader',
             'AerogridLoader',
             'Modal',
             # 'DynamicCoupled',
             ],
    'case': case_name,
    'route': case_route,
    'write_screen': 'on',
    'write_log': 'on'
}

settings['BeamLoader'] = {
    'unsteady': 'off',
    'orientation': np.array((1., 0., 0., 0.))}

settings['AerogridLoader'] = {
    'unsteady': False,
    'aligned_grid': True,
    'mstar': m_star_fact * m,
    'freestream_dir': u_inf_dir,
    'wake_shape_generator': 'StraightWake',
    'wake_shape_generator_input': {'u_inf': u_inf,
                             'u_inf_direction': u_inf_dir,
                             'dt': dt}
}

settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'off',
                                              'max_iterations': 950,
                                              'delta_curved': 1e-1,
                                              'min_delta': 1e3,
                                              'newmark_damp': 5e-3,
                                              'gravity_on': True,
                                              'gravity': 9.81,
                                              'num_steps': n_tstep,
                                              'dt': dt}

settings['StepUvlm'] = {'print_info': 'on',
                        'num_cores': 8,
                        'convection_scheme': 3,
                        'velocity_field_generator': 'SteadyVelocityField',
                        'velocity_field_input': {'u_inf': u_inf,
                                                 'u_inf_direction': u_inf_dir},
                        'rho': rho,
                        'n_time_steps': n_tstep,
                        'dt': dt,
                        'gamma_dot_filtering': 3}

settings['DynamicCoupled'] = {'print_info': 'on',
                              'structural_substeps': 0,
                              'dynamic_relaxation': 'on',
                              'cleanup_previous_solution': 'on',
                              'structural_solver': 'NonLinearDynamicPrescribedStep',
                              'structural_solver_settings': settings['NonLinearDynamicPrescribedStep'],
                              'aero_solver': 'StepUvlm',
                              'aero_solver_settings': settings['StepUvlm'],
                              'fsi_substeps': 200,
                              'minimum_steps': 1,
                              'relaxation_steps': 150,
                              'final_relaxation_factor': 0.0,
                              'n_time_steps': n_tstep,
                              'dt': dt,
                              'include_unsteady_force_contribution': 'off',
                              'postprocessors': ['BeamPlot', 'AerogridPlot'],
                              'postprocessors_settings': {'BeamPlot': {'include_rbm': 'on',
                                                                       'include_applied_forces': 'on'},
                                                          'AerogridPlot': {
                                                              'u_inf': u_inf,
                                                              'include_rbm': 'on',
                                                              'include_applied_forces': 'on',
                                                              'minus_m_star': 0}}}

settings['AerogridPlot'] = {'include_rbm': 'off',
                            'include_applied_forces': 'on',
                            'minus_m_star': 0}

settings['BeamPlot'] = {'include_rbm': 'off',
                        'include_applied_forces': 'on'}

settings['Modal'] = {'NumLambda': 20,
                     'rigid_body_modes': 'off',
                     'print_matrices': 'off',
                     'save_data': 'off',
                     'continuous_eigenvalues': 'off',
                     'dt': 0,
                     'plot_eigenvalues': False,
                     'max_rotation_deg': 15.,
                     'max_displacement': 0.15,
                     'write_modes_vtk': True,
                     'use_undamped_modes': True}

model = flexop.FLEXOP(case_name, case_route, case_out_folder)
model.init_aeroelastic(m=m, wing_only=True)
model.clean()
model.generate()
model.create_settings(settings)
case_data = model.run()

pass
