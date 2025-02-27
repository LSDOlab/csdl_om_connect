import openmdao.api as om
import csdl_alpha as csdl
import numpy as np
from csdl_om_connect import OpenMDAOExplicitOperation
import pytest
import modopt as mo

def test_exec_comp():
    exec_comp = om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3')

    recorder = csdl.Recorder()
    recorder.start()

    inputs = {}

    inputs['x'] = csdl.Variable(value=1.0, name='x')
    inputs['y'] = csdl.Variable(value=1.0, name='y')

    inputs['x'].set_as_design_variable(lower=0.0)
    inputs['y'].set_as_design_variable()

    # OpenMDAOExplicitOperation from an ExecComp object
    paraboloid = OpenMDAOExplicitOperation(exec_comp)

    outputs = paraboloid.evaluate(inputs, ['f'])
    outputs['f'].set_as_objective()
    optimized_solution = {'x' : np.array([ 6.66665446, -7.33332915]), 'objective': -27.33333333321794, 'nfev': 7, 'njev': 7, 'nit':7}

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder)

    prob = mo.CSDLAlphaProblem(problem_name='paraboloid', simulator=sim)
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results(all=True)

    assert np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
    assert np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
    assert optimizer.results['nfev'] == optimized_solution['nfev']
    assert optimizer.results['njev'] == optimized_solution['njev']
    assert optimizer.results['nit'] == optimized_solution['nit']


def test_unpromoted_exec_comp_model():
    exec_comp = om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3')
    prob = om.Problem()
    prob.model.add_subsystem('paraboloid', exec_comp)
    exec_comp_model = prob.model

    recorder = csdl.Recorder()
    recorder.start()

    inputs = {}

    # Note that promoted names will not work, say 'x' and 'y' instead of 'paraboloid.x' and 'paraboloid.y'
    inputs['paraboloid.x'] = csdl.Variable(value=1.0, name='x')
    inputs['paraboloid.y'] = csdl.Variable(value=1.0, name='y')

    inputs['paraboloid.x'].set_as_design_variable(lower=0.0)
    inputs['paraboloid.y'].set_as_design_variable()

    # OpenMDAOExplicitOperation from an unpromoted ExecComp model
    paraboloid = OpenMDAOExplicitOperation(exec_comp_model)

    outputs = paraboloid.evaluate(inputs, ['paraboloid.f'])
    outputs['paraboloid.f'].set_as_objective()
    optimized_solution = {'x' : np.array([ 6.66665446, -7.33332915]), 'objective': -27.33333333321794, 'nfev': 7, 'njev': 7, 'nit':7}

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder)

    prob = mo.CSDLAlphaProblem(problem_name='paraboloid', simulator=sim)
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results(all=True)

    assert np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
    assert np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
    assert optimizer.results['nfev'] == optimized_solution['nfev']
    assert optimizer.results['njev'] == optimized_solution['njev']
    assert optimizer.results['nit'] == optimized_solution['nit']

def test_promoted_exec_comp_model():
    exec_comp = om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3')
    prob = om.Problem()
    prob.model.add_subsystem('paraboloid', exec_comp, promotes=['*'])
    exec_comp_model = prob.model

    recorder = csdl.Recorder()
    recorder.start()

    inputs = {}

    # Note that unpromoted names will not work, say 'paraboloid.x' and 'paraboloid.y' instead of 'x' and 'y'
    inputs['x'] = csdl.Variable(value=1.0, name='x')
    inputs['y'] = csdl.Variable(value=1.0, name='y')

    inputs['x'].set_as_design_variable(lower=0.0)
    inputs['y'].set_as_design_variable()

    # OpenMDAOExplicitOperation from a promoted ExecComp model
    paraboloid = OpenMDAOExplicitOperation(exec_comp_model)

    outputs = paraboloid.evaluate(inputs, ['f'])
    outputs['f'].set_as_objective()
    optimized_solution = {'x' : np.array([ 6.66665446, -7.33332915]), 'objective': -27.33333333321794, 'nfev': 7, 'njev': 7, 'nit':7}

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder)

    prob = mo.CSDLAlphaProblem(problem_name='paraboloid', simulator=sim)
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results(all=True)

    assert np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
    assert np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
    assert optimizer.results['nfev'] == optimized_solution['nfev']
    assert optimizer.results['njev'] == optimized_solution['njev']
    assert optimizer.results['nit'] == optimized_solution['nit']


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

def test_explicit_component():

    recorder = csdl.Recorder()
    recorder.start()

    inputs = {}
    inputs['x'] = csdl.Variable(value=1.0, name='x')
    inputs['y'] = csdl.Variable(value=1.0, name='y')

    out_names = ['objective', 'constraint_1', 'constraint_2']

    quartic = OpenMDAOExplicitOperation(QuarticComp())

    outputs = quartic.evaluate(inputs, out_names)

    # Set the design variables, objective and constraints
    inputs['x'].set_as_design_variable(lower=0.0)
    inputs['y'].set_as_design_variable()
    
    outputs['objective'].set_as_objective()

    outputs['constraint_1'].set_as_constraint(equals=1.0)
    outputs['constraint_2'].set_as_constraint(lower =1.0)
    optimized_solution = {'x' : np.array([1.00000000e+00, 1.11022302e-16]), 'objective': 1.0, 'nfev': 2, 'njev': 2, 'nit':2}

    recorder.stop()
    
    sim = csdl.experimental.JaxSimulator(recorder)

    prob = mo.CSDLAlphaProblem(problem_name='quartic', simulator=sim)
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9})
    optimizer.solve()
    optimizer.print_results(all=True)

    assert np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
    assert np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
    assert optimizer.results['nfev'] == optimized_solution['nfev']
    assert optimizer.results['njev'] == optimized_solution['njev']
    assert optimizer.results['nit'] == optimized_solution['nit']


def test_unpromoted_explicit_component_model():

    prob = om.Problem()
    prob.model.add_subsystem('quartic', QuarticComp())
    quartic_model = prob.model

    recorder = csdl.Recorder()
    recorder.start()

    inputs = {}

    # Note that promoted and unpromoted names are same here.
    # so 'x' and 'y' instead of 'quartic.x' and 'quartic.y' will not work
    inputs['quartic.x'] = csdl.Variable(value=1.0, name='x')
    inputs['quartic.y'] = csdl.Variable(value=1.0, name='y')

    inputs['quartic.x'].set_as_design_variable(lower=0.0)
    inputs['quartic.y'].set_as_design_variable()

    # OpenMDAOExplicitOperation from an OpenMDAO model defined with an ExplicitComponent object
    quartic = OpenMDAOExplicitOperation(quartic_model)

    outputs = quartic.evaluate(inputs, ['quartic.objective', 'quartic.constraint_1', 'quartic.constraint_2'])
    outputs['quartic.objective'].set_as_objective()
    outputs['quartic.constraint_1'].set_as_constraint(equals=1.0)
    outputs['quartic.constraint_2'].set_as_constraint(lower =1.0)
    optimized_solution = {'x' : np.array([1.00000000e+00, 1.11022302e-16]), 'objective': 1.0, 'nfev': 2, 'njev': 2, 'nit':2}

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder)

    prob = mo.CSDLAlphaProblem(problem_name='quartic', simulator=sim)
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results(all=True)

    assert np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
    assert np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
    assert optimizer.results['nfev'] == optimized_solution['nfev']
    assert optimizer.results['njev'] == optimized_solution['njev']
    assert optimizer.results['nit'] == optimized_solution['nit']

def test_promoted_explicit_component_model():

    prob = om.Problem()
    prob.model.add_subsystem('quartic', QuarticComp(), promotes=['*'])
    quartic_model = prob.model

    recorder = csdl.Recorder()
    recorder.start()

    inputs = {}

    # Note that both promoted and unpromoted names will work, e.g., 'quartic.x', 'quartic.y', 'x', or 'y'
    inputs['quartic.x'] = csdl.Variable(value=1.0, name='x')
    inputs['y'] = csdl.Variable(value=1.0, name='y')

    inputs['quartic.x'].set_as_design_variable(lower=0.0)
    inputs['y'].set_as_design_variable()

    # OpenMDAOExplicitOperation from an OpenMDAO model defined with an ExplicitComponent object
    quartic = OpenMDAOExplicitOperation(quartic_model)

    outputs = quartic.evaluate(inputs, ['quartic.objective', 'constraint_1', 'quartic.constraint_2'])
    outputs['quartic.objective'].set_as_objective()
    outputs['constraint_1'].set_as_constraint(equals=1.0)
    outputs['quartic.constraint_2'].set_as_constraint(lower =1.0)
    optimized_solution = {'x' : np.array([1.00000000e+00, 1.11022302e-16]), 'objective': 1.0, 'nfev': 2, 'njev': 2, 'nit':2}

    recorder.stop()

    sim = csdl.experimental.PySimulator(recorder)
    # sim = csdl.experimental.JaxSimulator(recorder)

    prob = mo.CSDLAlphaProblem(problem_name='quartic', simulator=sim)
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results(all=True)

    assert np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
    assert np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
    assert optimizer.results['nfev'] == optimized_solution['nfev']
    assert optimizer.results['njev'] == optimized_solution['njev']
    assert optimizer.results['nit'] == optimized_solution['nit']

def test_problem():

    prob0 = om.Problem()
    prob0.model.add_subsystem('quartic', QuarticComp(), promotes=['*'])

    prob = om.Problem()
    prob.model.add_subsystem('quartic', QuarticComp(), promotes=['*'])
    prob.setup()

    recorder = csdl.Recorder()
    recorder.start()

    inputs = {}

    # Note that unpromoted names will not work, say 'quartic.x' and 'quartic.y' instead of 'x' and 'y'
    inputs['x'] = csdl.Variable(value=1.0, name='x')
    inputs['y'] = csdl.Variable(value=1.0, name='y')

    inputs['x'].set_as_design_variable(lower=0.0)
    inputs['y'].set_as_design_variable()

    # OpenMDAOExplicitOperation from an OpenMDAO problem
    quartic = OpenMDAOExplicitOperation(prob)

    with pytest.raises(ValueError) as excinfo:
        _ = OpenMDAOExplicitOperation(prob0)

    outputs = quartic.evaluate(inputs, ['objective', 'constraint_1', 'constraint_2'])
    outputs['objective'].set_as_objective()
    outputs['constraint_1'].set_as_constraint(equals=1.0)
    outputs['constraint_2'].set_as_constraint(lower =1.0)
    optimized_solution = {'x' : np.array([1.00000000e+00, 1.11022302e-16]), 'objective': 1.0, 'nfev': 2, 'njev': 2, 'nit':2}

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder)

    prob = mo.CSDLAlphaProblem(problem_name='quartic', simulator=sim)
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results(all=True)

    assert np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
    assert np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
    assert optimizer.results['nfev'] == optimized_solution['nfev']
    assert optimizer.results['njev'] == optimized_solution['njev']
    assert optimizer.results['nit'] == optimized_solution['nit']


if __name__ == '__main__':
    test_exec_comp()
    test_unpromoted_exec_comp_model()
    test_promoted_exec_comp_model()

    test_explicit_component()
    test_unpromoted_explicit_component_model()
    test_promoted_explicit_component_model()

    test_problem()

    print('All tests passed!')

    # The following section is for testing the numerical accuracy of the results

    prob1 = om.Problem()
    prob1.model.add_subsystem('paraboloid', om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))
    prob1.model.add_design_var('paraboloid.x')
    prob1.model.add_design_var('paraboloid.y')
    prob1.model.add_objective('paraboloid.f')
    prob1.setup()

    problem = mo.OpenMDAOProblem(problem_name='paraboloid', om_problem=prob1)
    optimizer = mo.SLSQP(problem, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(problem.x0)
    optimizer.solve()
    optimizer.print_results(all=True)

    prob2 = om.Problem()
    prob2.model.add_subsystem('quartic', QuarticComp(), promotes=['*'])
    prob2.model.add_design_var('x', lower=0.)
    prob2.model.add_design_var('y')
    prob2.model.add_objective('objective')
    prob2.model.add_constraint('constraint_1', equals=1.)
    prob2.model.add_constraint('constraint_2', lower=1.)
    prob2.setup()

    problem = mo.OpenMDAOProblem(problem_name='quartic', om_problem=prob2)
    optimizer = mo.SLSQP(problem, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(problem.x0)
    optimizer.solve()
    optimizer.print_results(all=True)
