import numpy as np
import warnings
import openmdao.api as om
from openmdao.core.constants import _SetupStatus
import csdl_alpha as csdl
from typing import List, Dict, Union
import copy
import modopt as mo

# Notes:
# 1. PySimulator is not supported.
# 2. Only full Jacobians (no JVPs) are supported for now.

class CSDLExplicitComponent(om.ExplicitComponent):
    '''
    Class for creating an OpenMDAO ExplicitComponent from a CSDL Simulator object.

    Parameters
    ----------
    sim : csdl.Simulator
        A CSDL JaxSimulator object to be transformed into an OpenMDAO ExplicitComponent object with
        the inputs and outputs defined in the ``in_names`` and ``out_names`` lists.
    in_names : List[str]
        A list of CSDL input variable names as strings.
        Must be valid names for the CSDL variables in the ``additional_inputs`` list.
    out_names : List[str]
        A list of CSDL output variable names as strings.
        Must be valid names for the CSDL variables in the ``additional_outputs`` list.

    Attributes
    ----------
    sim : csdl.Simulator
        The CSDL JaxSimulator object to be transformed into an OpenMDAO ExplicitComponent object.
    in_names : List[str]
        A list of CSDL input variable names that define the inputs of the OpenMDAO ExplicitComponent.
    out_names : List[str]
        A list of CSDL output variable names that define the outputs of the OpenMDAO ExplicitComponent.
    input_manager : csdl.InputManager
        The input manager attribute of the CSDL JaxSimulator object.
    output_manager : csdl.OutputManager
        The output manager attribute of the CSDL JaxSimulator object.
    available_in_names : List[str]
        A list of available input variable names in the CSDL JaxSimulator.
    available_out_names : List[str]
        A list of available output variable names in the CSDL JaxSimulator.
    in_names_map : Dict[str, csdl.Variable]
        A dictionary mapping the input variable names to the input variable objects in the CSDL JaxSimulator.
    out_names_map : Dict[str, csdl.Variable]
        A dictionary mapping the output variable names to the output variable objects in the CSDL JaxSimulator.
    '''
    def __init__(self, sim: csdl.experimental.JaxSimulator, in_names: list[str], out_names: list[str]):
        """
        Instantiate the custom operation object with a CSDL JaxSimulator object initialized with the correct options.
        """

        # define any checks for the parameters
        if not isinstance(sim, csdl.experimental.JaxSimulator):
            raise TypeError('"sim" must be an instance of csdl.experimental.JaxSimulator')
        
        self.sim = sim
        self.input_manager = sim.input_manager
        self.output_manager = sim.output_manager
        self.in_names  = in_names
        self.out_names = out_names

        for name_type in ['in_names', 'out_names']:
            names = getattr(self, name_type)
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                raise TypeError(f'"{names}" must be a list of strings.')
        
        if len(in_names) == 0:
            raise ValueError('"in_names" cannot be an empty list.')
        if len(out_names) == 0:
            raise ValueError('"out_names" cannot be an empty list.')

        self.available_in_names = []
        for in_var in self.input_manager.list:
            self.available_in_names += in_var.names

        self.available_out_names = []
        for out_var in self.output_manager.list:
            self.available_out_names += out_var.names

        unavailable_in_names  = [ in_name for  in_name in  in_names if  in_name not in self.available_in_names ]
        unavailable_out_names = [out_name for out_name in out_names if out_name not in self.available_out_names]

        if len(unavailable_in_names) > 0:
            raise ValueError(f'The following input names are not available input variable names in the CSDL JaxSimulator: {unavailable_in_names}. '
                             f'Available input names are: {self.available_in_names}.')
        if len(unavailable_out_names) > 0:
            raise ValueError(f'The following output names are not available output variable names in the CSDL JaxSimulator: {unavailable_out_names}. '
                             f'Available output names are: {self.available_out_names}.')

        # Map the in_names to input variables
        self.in_names_map = {}
        for in_name in self.in_names:
            for in_var in self.input_manager.list:
                if in_name in in_var.names:
                    self.in_names_map[in_name] = in_var
                    break
        
        # Map the out_names to output variables
        self.out_names_map = {}
        for out_name in self.out_names:
            for out_var in self.output_manager.list:
                if out_name in out_var.names:
                    self.out_names_map[out_name] = out_var
                    break

        # call the base class constructor
        super().__init__()

    def setup(self):
        '''
        Add inputs, outputs, and declare partials for the OpenMDAO component.
        '''
        # add inputs
        for in_name in self.in_names:
            in_var = self.in_names_map[in_name]
            self.add_input(in_name, val=in_var.value)

        # add outputs
        for out_name in self.out_names:
            out_var = self.out_names_map[out_name]
            self.add_output(out_name, shape=out_var.shape)

        # declare all partials
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        # in_names do not work as keys in __setitem__ of JaxSimulator
        for in_name in self.in_names:
            in_var = self.in_names_map[in_name]
            self.sim[in_var] = inputs[in_name]

        self.sim.run()

        # out_names work as keys in __getitem__ of JaxSimulator
        for out_name in self.out_names:
            outputs[out_name] = self.sim[out_name]

    def compute_partials(self, inputs, partials):
        for in_name in self.in_names:
            in_var = self.in_names_map[in_name]
            self.sim[in_var] = inputs[in_name]

        # returns a dictionary {(out_var: csdl.Variable, in_var: csdl.Variable): val: np.ndarray}
        totals = self.sim.compute_totals()

        for out_name in self.out_names:
            out_var = self.out_names_map[out_name]
            for in_name in self.in_names:
                in_var = self.in_names_map[in_name]
                partials[out_name, in_name] = totals[out_var, in_var]

if __name__ == "__main__":
    
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

    # CSDLExplicitComponent from a CSDL simulator object
    quartic_comp = CSDLExplicitComponent(quartic_sim,
                                         in_names=['x', 'y'],
                                         out_names=['objective', 'constraint_1', 'constraint_2'])

    om_prob = om.Problem()
    om_prob.model.add_subsystem('quartic', quartic_comp, promotes=['*'])

    # Note that only promoted names can be used for 
    # adding design variables, objectives, and constraints
    om_prob.model.add_design_var('x', lower=0.)
    om_prob.model.add_design_var('y')
    om_prob.model.add_objective('objective')
    om_prob.model.add_constraint('constraint_1', equals=1.0)
    om_prob.model.add_constraint('constraint_2', lower=1.0)

    om_prob.setup()

    prob = mo.OpenMDAOProblem(problem_name='quartic', om_problem=om_prob)
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results(all=True)

    optimized_solution = {'x' : np.array([1.00000000e+00, 1.11022302e-16]), 'objective': 1.0, 'nfev': 2, 'njev': 2, 'nit':2}

    assert np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
    assert np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
    assert optimizer.results['nfev'] == optimized_solution['nfev']
    assert optimizer.results['njev'] == optimized_solution['njev']
    assert optimizer.results['nit'] == optimized_solution['nit']

    print('Tests passed!')