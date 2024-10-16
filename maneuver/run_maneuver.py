import os
import sys
import numpy as np
from create_multibody_flexop import FlexopAeroelastic
import sharpy.sharpy_main

import warnings
warnings.filterwarnings("ignore")

alpha = np.deg2rad(-1.888390743495187429e-01)
delta = np.deg2rad(-2.696958969597490263e+00)
thrust = 3.997980788977186339e+00

# alpha = 0.
# delta = 0.
# thrust = 0.
# thrust = 0.

# Simulation inputs
case_name = 'flexop_maneuver'
case_route = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/cases/'
case_out_folder = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/output/'
pickle_out_file = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/flexop_trim.pkl'

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
gust_intensity = 0.
gust_length = 1.
m = 10
m_star_fact = 1.5
physical_time = 5.
c_ref = 0.35
rho = 1.225
# rho = 0.
dt = c_ref / (m * u_inf)
n_tstep = int(physical_time / dt)
sigma = 1.
roll = np.deg2rad(0.)
yaw = np.deg2rad(0.)
use_multibody = True
use_aero = True
# use_aero = False
use_jax = True
include_tail = True
use_airfoil = True
use_rigid_sweep = False
free = True
cfl1 = False
num_elem_warp_main = 2
num_elem_warp_tip = 2
flow = ['BeamLoader',
        'AerogridLoader',
        'DynamicCoupled',
        ]

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
            'use_jax': use_jax,
            'use_rigid_sweep': use_rigid_sweep,
            'num_elem_warp_main': num_elem_warp_main,
            'num_elem_warp_tip': num_elem_warp_tip,
            'alpha': alpha,
            'yaw': yaw,
            'roll': roll,
            'm_wing': m,
            'sigma': sigma,
            'dt': dt,
            'rho': rho,
            'free': free,
            'cfl1': cfl1,
            'n_tstep': n_tstep,
            'm_star_fact': m_star_fact,
            'u_inf': u_inf,
            'u_inf_dir': u_inf_dir,
            'gust_intensity': gust_intensity,
            'gust_length': gust_length,
            'flow': flow,
            'elevator_angle': delta,
            'thrust': thrust
            }

# use_control = True
use_control = False
# constraint_settings = {'use_control': use_control,
#                        'input_angle_rhs_dir': input_angle_rhs_dir,
#                        'input_velocity_rhs_dir': input_velocity_rhs_dir,
#                        'input_angle_lhs_dir': input_angle_lhs_dir,
#                        'input_velocity_lhs_dir': input_velocity_lhs_dir,
#                        'u_rhs': u_rhs,
#                        'u_dot_rhs': u_dot_rhs,
#                        'u_lhs': u_lhs,
#                        'u_dot_lhs': u_dot_lhs}

constraint_settings = {'flare_angle': np.deg2rad(0.)}

model = FlexopAeroelastic(case_name, case_route, **settings)
# model.add_constraint('prescribed_hinge', **constraint_settings)
model.add_constraint('free_hinge', **constraint_settings)
model.generate_h5()
model.generate_settings()

case_data = sharpy.sharpy_main.main(['', case_route + '/' + case_name + '.sharpy'])

