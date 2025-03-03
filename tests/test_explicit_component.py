import openmdao.api as om
import csdl_alpha as csdl
import numpy as np
from csdl_om_connect import CSDLExplicitComponent
import pytest
import modopt as mo

def test_simple_operation():
    recorder = csdl.Recorder()
    recorder.start()

    x = csdl.Variable(name = 'x', value=1.0)
    y = csdl.Variable(name = 'y', value=1.0)

    f = (x-3)**2 + x*y + (y+4)**2 - 3
    f.add_name('f')

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder=recorder,
                                         additional_inputs=[x, y],
                                         additional_outputs=[f])

    # CSDLExplicitComponent from a CSDL simulator object
    paraboloid = CSDLExplicitComponent(sim,
                                       in_names=['x', 'y'],
                                       out_names=['f'])
    
    with pytest.raises(ValueError) as excinfo:
        paraboloid = CSDLExplicitComponent(sim, [], ['f'])
    assert '"in_names" cannot be an empty list.' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        paraboloid = CSDLExplicitComponent(sim, ['x', 'y'], [])
    assert '"out_names" cannot be an empty list.' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        paraboloid = CSDLExplicitComponent(sim, ['x', 'z'], ['f'])
    assert 'The following input names are not available input variable names in the CSDL JaxSimulator:' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        paraboloid = CSDLExplicitComponent(sim, ['x', 'y'], ['f', 'g'])
    assert 'The following output names are not available output variable names in the CSDL JaxSimulator:' in str(excinfo.value)

    om_prob = om.Problem()
    om_prob.model.add_subsystem('paraboloid', paraboloid, promotes=['*'])
    # As we promoted all the inputs and outputs, we must use promoted names
    om_prob.model.add_design_var('x', lower=0.0)
    om_prob.model.add_design_var('y')
    om_prob.model.add_objective('f')
    om_prob.setup()

    optimized_solution = {'x' : np.array([ 6.66665446, -7.33332915]), 'objective': -27.33333333321794, 'nfev': 7, 'njev': 7, 'nit':7}

    prob = mo.OpenMDAOProblem(problem_name='paraboloid', om_problem=om_prob)
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results(all=True)

    assert np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
    assert np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
    assert optimizer.results['nfev'] == optimized_solution['nfev']
    assert optimizer.results['njev'] == optimized_solution['njev']
    assert optimizer.results['nit'] == optimized_solution['nit']

def test_operation():

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

    sim = csdl.experimental.JaxSimulator(recorder=recorder,
                                         additional_inputs=[x, y],
                                         additional_outputs=[objective, constraint_1, constraint_2])

    # CSDLExplicitComponent from a CSDL simulator object
    quartic = CSDLExplicitComponent(sim, 
                                     in_names=['x', 'y'],
                                     out_names=['objective', 'constraint_1', 'constraint_2'])

    om_prob = om.Problem()
    om_prob.model.add_subsystem('quartic', quartic, promotes=['*'])
    # As we promoted all the inputs and outputs, we must use promoted names
    om_prob.model.add_design_var('x', lower=0.0)
    om_prob.model.add_design_var('y')
    om_prob.model.add_objective('objective')
    om_prob.model.add_constraint('constraint_1', equals=1.0)
    om_prob.model.add_constraint('constraint_2', lower =1.0)
    om_prob.setup()

    optimized_solution = {'x' : np.array([1.00000000e+00, 1.11022302e-16]), 'objective': 1.0, 'nfev': 2, 'njev': 2, 'nit':2}

    prob = mo.OpenMDAOProblem(problem_name='quartic', om_problem=om_prob)
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.solve()
    optimizer.print_results(all=True)

    assert np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
    assert np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
    assert optimizer.results['nfev'] == optimized_solution['nfev']
    assert optimizer.results['njev'] == optimized_solution['njev']
    assert optimizer.results['nit'] == optimized_solution['nit']

if __name__ == '__main__':
    test_simple_operation()
    test_operation()

    print('All tests passed!')

    # The following section is for testing the numerical accuracy of the results

    # 1. Paraboloid
    recorder = csdl.Recorder()
    recorder.start()

    x = csdl.Variable(name = 'x', value=1.0)
    y = csdl.Variable(name = 'y', value=1.0)

    f = (x-3)**2 + x*y + (y+4)**2 - 3
    f.add_name('f')

    x.set_as_design_variable(lower=0.0)
    y.set_as_design_variable()
    f.set_as_objective()

    recorder.stop()

    sim1 = csdl.experimental.JaxSimulator(recorder=recorder,
                                          additional_inputs=[x, y],
                                          additional_outputs=[f])
    
    problem = mo.CSDLAlphaProblem(problem_name='paraboloid', simulator=sim1)
    optimizer = mo.SLSQP(problem, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(problem.x0)
    optimizer.solve()
    optimizer.print_results(all=True)
    
    # 2. Quartic
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

    x.set_as_design_variable(lower=0.0)
    y.set_as_design_variable()
    objective.set_as_objective()
    constraint_1.set_as_constraint(equals=1.0)
    constraint_2.set_as_constraint(lower =1.0)

    recorder.stop()

    sim2 = csdl.experimental.JaxSimulator(recorder=recorder,
                                          additional_inputs=[x, y],
                                          additional_outputs=[objective, constraint_1, constraint_2])

    problem = mo.CSDLAlphaProblem(problem_name='quartic', simulator=sim2)
    optimizer = mo.SLSQP(problem, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(problem.x0)
    optimizer.solve()
    optimizer.print_results(all=True)