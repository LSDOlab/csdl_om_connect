import openmdao.api as om
import csdl_alpha as csdl
import numpy as np
from csdl_om_connect import OpenMDAOExplicitOperation
import modopt as mo

# minimize      x^4 + y^4 
#      wrt      x, y 
# subject to    x >= 0,
#               x + y  = 1,
#               x - y >= 1.

# OpenMDAO Explicit Component
#################################################

class QuarticComp(om.ExplicitComponent):
    def setup(self): 
        self.add_input('x', 1.)
        self.add_input('y', 1.)
        
        self.add_output('objective')
        self.add_output('constraint_1')
        self.add_output('constraint_2')

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['objective'] = x**4 + y**4
        outputs['constraint_1'] = x + y
        outputs['constraint_2'] = x - y

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']

        partials['objective', 'x'] = 4 * x**3
        partials['objective', 'y'] = 4 * y**3

        partials['constraint_1', 'x'] = 1.0
        partials['constraint_1', 'y'] = 1.0
        partials['constraint_2', 'x'] = 1.0
        partials['constraint_2', 'y'] = -1.0

# OpenMDAO Problem
#################################################

quartic_prob = om.Problem()
quartic_prob.model.add_subsystem('quartic', QuarticComp(), promotes=['*'])
# Note that the problem needs to be set up before connecting it to CSDL
quartic_prob.setup()

# CSDL model
#################################################

recorder = csdl.Recorder()
recorder.start()

# 1. Define the inputs that go into the OpenMDAO problem from CSDL
####################################################################

# Inputs are defined within a dictionary with keys as names of inputs
# in the OpenMDAO problem and values as csdl.Variable objects.
# Only those inputs that are independent variables in OpenMDAO can be passed.
# Both promoted and unpromoted names in OpenMDAO will work as keys, 
# e.g., 'quartic.x', 'quartic.y', 'x', or 'y' all will work.

inputs = {}
inputs['quartic.x'] = csdl.Variable(value=1.0, name='x')
inputs['y'] = csdl.Variable(value=1.0, name='y')

# 2. Define the outputs that need to be passed from OpenMDAO to CSDL
####################################################################

# It should be a list with the names of the outputs in the OpenMDAO component.
# Only those outputs that are not independent variables in OpenMDAO can be passed.
# Once again, both promoted and unpromoted names in OpenMDAO will work as keys,
# e.g., 'quartic.objective', 'objective', 'quartic.constraint_1', 'constraint_1', etc. will work.

out_names = ['quartic.objective', 'constraint_1', 'quartic.constraint_2']

# 3. Generate an OpenMDAOExplicitOperation object from the OpenMDAO problem
#######################################################################################

quartic = OpenMDAOExplicitOperation(quartic_prob)

# 4. Evaluate the operation to generate the output variables corresponding to out_names
#######################################################################################

# The output variables are returned in a dictionary with keys as names in out_names
# and values as csdl.Variable objects.

outputs = quartic.evaluate(inputs, out_names)

# Set the design variables, objective and constraints
inputs['quartic.x'].set_as_design_variable(lower=0.0)
inputs['y'].set_as_design_variable()

outputs['quartic.objective'].set_as_objective()

outputs['constraint_1'].set_as_constraint(equals=1.0)
outputs['quartic.constraint_2'].set_as_constraint(lower =1.0)

recorder.stop()

sim = csdl.experimental.JaxSimulator(recorder)

# Optimization
#################################################

prob = mo.CSDLAlphaProblem(problem_name='quartic', simulator=sim)
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