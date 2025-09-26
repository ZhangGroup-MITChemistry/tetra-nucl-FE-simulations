import numpy as np
import openmm.unit as unit
import argparse

# Generates a point grid within bounding lines for maximum and minimum d13-d24 distance as expected for an e2e
# Boundaries of x points defined by the following two lines:
# 1. slope (80/-25+5) and intercept: -(80-0)/(-25+5)*(-5)
# 2. slope (80/25-5) and intercept: -(80-0)/(25-5)*(5)

# Given a current window script will print the grid points associated with the window and axis (0 or 1).
# Will also calculate a k such that if we assume the distribution of collective variables from the umbrella window is distributed according to a 2D normal (* 1/KbT),
# the resulting distribution will cover at least 1/2 the inter-gridpoint spacing with 1 standard deviation. This is necessary since on the x-axis we have
# an increasing inter-point width with increasing y. We could alternatively just run more simulation windows to fill in gaps, but the necessity of this will depend on the
# resulting 2D distribution generated from the combined simulations.


GAS_CONSTANT = (1.0 * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).in_units_of(unit.kilojoule_per_mole / unit.kelvin)
RT = (300 * unit.kelvin) * GAS_CONSTANT

parser = argparse.ArgumentParser()
parser.add_argument("--dbug", action='store_true')
parser.add_argument('--curr_window', type=int,required=True)
parser.add_argument('--axis', type=int, required=True)
parser.add_argument("--k", action='store_true')
args = parser.parse_args()

optimal_spacing = 7.5
ub = 81.75
lb = 2
y_count = int(np.ceil((ub - lb)/optimal_spacing) + 1)
grid_dim1 = y_count
scalar = 66

grid_range = np.linspace(lb,ub,y_count)

y_spacing = grid_range[1] - grid_range[0]

y_points = np.empty(0)
x_points = np.empty(0)
x_spacing = np.empty(0)
k_x = np.empty(0)
k_y = np.empty(0)
window_id = np.zeros(1, dtype=int)
for i in range(grid_dim1):
    b_lb = -(80-0)/(-25+5)*(-5)
    x_lb = (-25+5)/(80-0) * (grid_range[i] - b_lb)
    b_ub = -(80-0)/(25-5)*(5)
    x_ub = (25-5)/(80-0) * (grid_range[i] - b_ub)
    x_count = int(np.ceil((x_ub - x_lb)/optimal_spacing) + 1)
    this_x_points = np.linspace(x_lb, x_ub, x_count)
    x_points = np.append(x_points, this_x_points)
    x_spacing = np.append(x_spacing, np.tile(this_x_points[1] - this_x_points[0],x_count))
    y_points = np.append(y_points, np.tile(grid_range[i],x_count))
    k_x = np.append(k_x, np.tile((2/(RT * (this_x_points[1] - this_x_points[0])**2).value_in_unit(unit.kilojoule_per_mole) * scalar),x_count))
    k_y = np.append(k_y, np.tile((2/(RT * (y_spacing)**2).value_in_unit(unit.kilojoule_per_mole) * scalar),x_count))
    window_id = np.append(window_id, np.arange(window_id[-1]+1, window_id[-1] + x_count+1))

window_id = window_id[:-1]
grid_points = np.stack((x_points, y_points, k_x, k_y, window_id), axis=-1)
if args.dbug:
    print(grid_points)
    exit()

if args.k:
    if args.axis == 0:
        print(f"{grid_points[args.curr_window][2]:.4f}")
    if args.axis == 1:
        print(f"{grid_points[args.curr_window][3]:.4f}")
else:
    print(f"{grid_points[args.curr_window][args.axis]:.4f}")

