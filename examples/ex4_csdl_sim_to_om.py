import openmdao.api as om
import csdl_alpha as csdl
import numpy as np
from csdl_om_connect import CSDLExplicitComponent
import modopt as mo

# minimize      x^4 + y^4 
#      wrt      x, y 
# subject to    x >= 0,
#               x + y  = 1,
#               x - y >= 1.

# CSDL JaxSimulator
#################################################

recorder = csdl.Recorder()
recorder.start()

x = csdl.Variable(name = 'x', value=1.0)
y = csdl.Variable(name = 'y', value=1.0)

objective = x**2 + y**2
constraint_1 = x + y
constraint_2 = x - y
objective.add_name('objective')
constraint_1.add_name('constraint_1')
constraint_2.add_name('constraint_2')

recorder.stop()

quartic_sim = csdl.experimental.JaxSimulator(recorder=recorder,
                                             additional_inputs=[x, y],
                                             additional_outputs=[objective, constraint_1, constraint_2])

# OpenMDAO Model
#################################################

# 1. Define input and output names
#################################################

# Both are defined as a list of strings of CSDL variable names.
# The names must be valid names of the CSDL variables 
# exposed as `additional_inputs` and `additional_outputs` in the JaxSimulator.
in_names = ['x', 'y']
out_names = ['objective', 'constraint_1', 'constraint_2']

# 2. Generate a CSDLExplicitComponent object from a CSDL Jax Simulator
########################################################################

quartic_comp = CSDLExplicitComponent(quartic_sim, 
                                     in_names=in_names,
                                     out_names=out_names)

# OpenMDAO Problem
#################################################
quartic_prob = om.Problem()
quartic_prob.model.add_subsystem('quartic', quartic_comp, promotes=['*'])

# Set the design variables, objective, and constraints
# Since we promoted all the inputs and outputs, we must use promoted names
quartic_prob.model.add_design_var('x', lower=0.0)
quartic_prob.model.add_design_var('y')

quartic_prob.model.add_objective('objective')

quartic_prob.model.add_constraint('constraint_1', equals=1.0)
quartic_prob.model.add_constraint('constraint_2', lower =1.0)

# Set up the OpenMDAO problem
quartic_prob.setup()

# Optimization
#################################################

prob = mo.OpenMDAOProblem(problem_name='quartic', om_problem=quartic_prob)
optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter':20})
optimizer.solve()
optimizer.print_results(all=True)

# Validate results
#################################################

optimized_solution = {'x' : np.array([1.00000000e+00, 1.11022302e-16]), 'objective': 1.0, 'nfev': 2, 'njev': 2, 'nit':2}

assert np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
assert np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
assert optimizer.results['nfev'] == optimized_solution['nfev']
assert optimizer.results['njev'] == optimized_solution['njev']
assert optimizer.results['nit'] == optimized_solution['nit']