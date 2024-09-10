import os
import sys
import scipy.io as spio
import numpy as np
from create_multibody_flexop import FlexopAeroelastic
import sharpy.sharpy_main
from case_data_extract import case_data_extract

# 108 cases
# ordering is by: index = i_gust * len(ang) + i_ang
import sys
index = int(sys.argv[1])

gust_lengths = np.array((1., 2., 4., 6., 8., 10., 13., 16., 20., 25., 30., 40.))
tip_angles = np.deg2rad(np.array((-40., -30., -20., -10., 0., 10., 20., 30., 40.)))

i_gust = index // len(tip_angles)
i_ang = index % len(tip_angles)

# varies between cases
gust_length = gust_lengths[i_gust]
rigid_sweep_ang = tip_angles[i_ang]

# Simulation inputs
case_name = f'rigid_flexop_ang{np.rad2deg(rigid_sweep_ang):.1f}_glength{gust_length:.1f}'
case_route = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/cases/'
case_out_folder = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/output/'
mat_out_file = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/flexop_out.mat'

try:
    os.makedirs(case_route)
except FileExistsError:
    pass

try:
    os.makedirs(case_out_folder)
except FileExistsError:
    pass

u_inf = 45.
u_inf_dir = np.array((1., 0., 0.))
gust_intensity = 0.5
m = 10
m_star_fact = 4.
physical_time = 0.2
c_ref = 0.35
rho = 1.225
dt = c_ref / (m * u_inf)
n_tstep = int(physical_time / dt)
sigma = 0.3
roll = np.deg2rad(0.)
alpha = np.deg2rad(0.)
yaw = np.deg2rad(0.)
use_multibody = False
use_aero = True
include_tail = False
use_airfoil = True
use_rigid_sweep = True
num_elem_warp_main = 2
num_elem_warp_tip = 2
flow = ['BeamLoader',
        'AerogridLoader',
        'Modal',
        'StaticCoupled',
        'DynamicCoupled',
        'AeroForcesCalculator',
        'BeamLoads']

# rotation about z - we will add the offset during model generation
omega_u = 2 * np.pi / physical_time
amp_u = np.deg2rad(30.)

u_rhs = np.zeros((n_tstep, 3))
u_dot_rhs = np.zeros((n_tstep, 3))
u_lhs = np.zeros((n_tstep, 3))
u_dot_lhs = np.zeros((n_tstep, 3))

t = np.linspace(dt, physical_time, n_tstep)
ang = amp_u * np.sin(omega_u * t)
ang_dot = amp_u * omega_u * np.cos(omega_u * t)

u_rhs[:, 2] = ang
u_dot_rhs[:, 2] = ang_dot
u_lhs[:, 2] = -ang
u_dot_lhs[:, 2] = -ang_dot

input_angle_rhs_dir = case_route + 'input_angle_rhs.npy'
input_velocity_rhs_dir = case_route + 'input_velocity_rhs.npy'
input_angle_lhs_dir = case_route + 'input_angle_lhs.npy'
input_velocity_lhs_dir = case_route + 'input_velocity_lhs.npy'

settings = {'use_multibody': use_multibody,
            'include_tail': include_tail,
            'use_airfoil': use_airfoil,
            'use_aero': use_aero,
            'use_rigid_sweep': use_rigid_sweep,
            'rigid_sweep_ang': rigid_sweep_ang,
            'num_elem_warp_main': num_elem_warp_main,
            'num_elem_warp_tip': num_elem_warp_tip,
            'alpha': alpha,
            'yaw': yaw,
            'roll': roll,
            'm_wing': m,
            'sigma': sigma,
            'dt': dt,
            'rho': rho,
            'n_tstep': n_tstep,
            'm_star_fact': m_star_fact,
            'u_inf': u_inf,
            'u_inf_dir': u_inf_dir,
            'gust_intensity': gust_intensity,
            'gust_length': gust_length,
            'flow': flow,
            }

use_control = False
# constraint_settings = {'flare_angle': np.deg2rad(0.)}
# constraint_settings = {'use_control': use_control,
#                        'input_angle_rhs_dir': input_angle_rhs_dir,
#                        'input_velocity_rhs_dir': input_velocity_rhs_dir,
#                        'input_angle_lhs_dir': input_angle_lhs_dir,
#                        'input_velocity_lhs_dir': input_velocity_lhs_dir,
#                        'u_rhs': u_rhs,
#                        'u_dot_rhs': u_dot_rhs,
#                        'u_lhs': u_lhs,
#                        'u_dot_lhs': u_dot_lhs}

model = FlexopAeroelastic(case_name, case_route, **settings)
# model.add_constraint('clamped')
# model.add_constraint('free_hinge', **constraint_settings)
# model.add_constraint('prescribed_hinge', **constraint_settings)
model.generate_h5()
model.generate_settings()

case_data = sharpy.sharpy_main.main(['', case_route + '/' + case_name + '.sharpy'])

try:
    data_out = spio.loadmat(mat_out_file)['flexop_data']
except FileNotFoundError:
    data_out = np.full((len(gust_lengths), len(tip_angles)), fill_value={'case_run': 0}, dtype=dict)

data_out[i_gust, i_ang] = case_data_extract(model, case_data)
spio.savemat(mat_out_file, {'flexop_data': data_out})
