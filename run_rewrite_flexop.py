import os
import shutil
import numpy as np
from structure_mb_rewrite import FlexopAeroelastic
import sharpy.sharpy_main

# Simulation inputs
case_name = 'flexop_multibody'
case_route = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/cases/'
case_out_folder = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/output/'

# Remove old case files and outputs
try:
    shutil.rmtree(case_out_folder)
    shutil.rmtree(case_route)
except:
    pass

try:
    os.makedirs(case_route)
except FileExistsError:
    pass

try:
    os.makedirs(case_out_folder)
except FileExistsError:
    pass

u_inf = 20.
u_inf_dir = np.array((1., 0., 0.))
m = 6
m_star_fact = 1
physical_time = 1.
c_ref = 0.35
rho = 1.225
dt = c_ref / (m * u_inf)
n_tstep = int(physical_time / dt)
sigma = 0.3
roll = np.deg2rad(0.)
alpha = np.deg2rad(5.)
yaw = np.deg2rad(0.)
use_multibody = True
use_aero = True
include_tail = False
use_airfoil = True

flow = ['BeamLoader',
        'AerogridLoader',
        'DynamicCoupled']

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
            'alpha': alpha,
            'yaw': yaw,
            'roll': roll,
            'm_wing': m,
            'sigma': sigma,
            'dt': dt,
            'n_tstep': n_tstep,
            'm_star_fact': m_star_fact,
            'u_inf': u_inf,
            'u_inf_dir': u_inf_dir,
            'flow': flow,
            }

use_control = True
# constraint_settings = {'flare_angle': np.deg2rad(20.)}
constraint_settings = {'use_control': use_control,
                       'input_angle_rhs_dir': input_angle_rhs_dir,
                       'input_velocity_rhs_dir': input_velocity_rhs_dir,
                       'input_angle_lhs_dir': input_angle_lhs_dir,
                       'input_velocity_lhs_dir': input_velocity_lhs_dir,
                       'u_rhs': u_rhs,
                       'u_dot_rhs': u_dot_rhs,
                       'u_lhs': u_lhs,
                       'u_dot_lhs': u_dot_lhs,
                       'n_elem_warp_main': 4,
                       'n_elem_warp_tip': 2}

model = FlexopAeroelastic(case_name, case_route, **settings)
model.add_constraint('clamped')
# model.add_constraint('free_hinge', **constraint_settings)
model.add_constraint('prescribed_hinge', **constraint_settings)
model.generate_h5()
model.generate_settings()

sharpy.sharpy_main.main(['', case_route + '/' + case_name + '.sharpy'])

pass
