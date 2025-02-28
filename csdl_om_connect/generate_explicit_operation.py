import numpy as np
import warnings
import openmdao.api as om
from openmdao.core.constants import _SetupStatus
import csdl_alpha as csdl
from typing import List, Dict, Union
import copy

# Notes:
# 1. CSDL CustomExplicitOperation only supports full Jacobian computation.
# 2. Names of input/output variable keys in the evaluate method for the CSDL CustomExplicitOperation
#    can be the PROMOTED/UNPROMOTED names of the variables in the OpenMDAO model.
# 3. Problem objects need to call the setup() method before running the model.
class OpenMDAOExplicitOperation(csdl.CustomExplicitOperation):
    '''
    Class for creating a CSDL CustomExplicitOperation from an OpenMDAO model
    in the form of an ExplicitComponent, Group, or Problem object.
    The OpenMDAO model will be transformed into a CSDL CustomExplicitOperation object with 
    the option for user to expose the model inputs and outputs as CSDL variables using
    the PROMOTED/UNPROMOTED names of the variables in the OpenMDAO model.

    Parameters
    ----------
    model : Union[om.ExplicitComponent, om.Group, om.Problem]
        The OpenMDAO model in the form of an ExplicitComponent, Group, or Problem object.
        This model will be transformed into a CSDL CustomExplicitOperation object.
    solver_print_level : int, default=0
        The print level for solvers in the OpenMDAO model.

    Attributes
    ----------
    model : Union[om.ExplicitComponent, om.Group]
        The OpenMDAO model in the form of an ExplicitComponent or Group object.
    problem : om.Problem
        The OpenMDAO Problem object initialized with the given OpenMDAO model.
        If the class is instantiated with an OpenMDAO Problem object, 
        this attribute will be the same input Problem object.
    all_input_meta : Dict[str, Dict]
        A dictionary of all input metadata from the OpenMDAO model with unpromoted names as keys.
    all_output_meta : Dict[str, Dict]
        A dictionary of all output metadata from the OpenMDAO model with unpromoted names as keys.
    input_name_map : Dict[str, List[str]]
        A dictionary that maps promoted names to the list of all unpromoted names, for all input variables in the OpenMDAO model.
    output_name_map : Dict[str, List[str]]
        A dictionary that maps promoted names to the list of all unpromoted names, for all output variables in the OpenMDAO model.
    available_input_names : List[str]
        A list of all available input names for the user in CSDL,
        includes promoted and unpromoted names of all independent inputs in the model
        These inputs can be independently assigned values through csdl.Variable objects, if exposed by the user.
    available_output_names : List[str]
        A list of all available output names for the user in CSDL,
        includes promoted and unpromoted names of all non-independent outputs in the model
        These outputs can be accessed and used as CSDL variables, if exposed by the user.
    all_input_names : List[str]
        A list of all input names from the OpenMDAO model with promoted/unpromoted names.
    all_output_names : List[str]
        A list of all output names from the OpenMDAO model with promoted/unpromoted names.
    in_names : List[str]
        A list of input names that are exposed by the user as CSDL variables and can be assigned values.
    out_names : List[str]
        A list of output names that are exposed by the user as CSDL variables and can be used as regular CSDL variables.
    prom_in_names : List[str]
        A list of promoted input names in the order corresponding to the ``in_names`` list to be used for compute_totals.
    prom_out_names : List[str]
        A list of promoted output names in the order corresponding to the ``out_names`` list to be used for compute_totals.
    '''

    def __init__(self, model: Union[om.ExplicitComponent, om.Group, om.Problem], solver_print_level: int = 0):
            """
            Instantiate the custom operation object with an OpenMDAO model initialized with the correct options.
            """
            super().__init__()

            # define any checks for the parameters
            csdl.check_parameter(model, 'model', types=(om.ExplicitComponent, om.Group, om.Problem))

            # create an OpenMDAO problem object
            self.model   = model
            self.problem = om.Problem()
            if isinstance(model, om.ExplicitComponent):
                self.problem.model.add_subsystem('model', model, promotes=['*'])
            elif isinstance(model, om.Group):
                self.problem.model = self.model
            else: # Replace the model and problem attributes if the argument is an OpenMDAO Problem object
                self.problem = model
                self.model   = self.problem.model
                # Raise an error if the setup() method is not called by the user before passing the Problem object
                if (self.problem._metadata is None) or (not self.problem._metadata['setup_status'] >= _SetupStatus.POST_SETUP):
                        raise ValueError('The setup() method has not been called for the provided OpenMDAO Problem object. '
                                        'Please call the setup() before passing in the OpenMDAO problem object.')
                                          
            # Call setup for the problem if the model is an ExplicitComponent or a Group
            if not isinstance(model, om.Problem):
                self.problem.setup()

            # Set the solver print level for the OpenMDAO Problem object
            self.problem.set_solver_print(level=solver_print_level)
            
            # list_indep_vars requires that final_setup has been run for the Problem.
            self.problem.final_setup()

            # Get all input and output names from the model along with their promoted names
            self.all_input_meta  = self.problem.model.list_inputs(return_format='dict', out_stream=None, shape=True)
            self.all_output_meta = self.problem.model.list_outputs(return_format='dict', out_stream=None, shape=True)

            # # Replace all input and output names from the model as promoted names - no longer used
            # self.all_input_meta  = {item['prom_name']: self.all_input_meta[key] for key, item in self.all_input_meta.items()}
            # self.all_output_meta = {item['prom_name']: self.all_output_meta[key] for key, item in self.all_output_meta.items()}

            # All input and output unpromoted names
            self.all_input_names  = list(self.all_input_meta.keys())
            self.all_output_names = list(self.all_output_meta.keys())

            # All available input names for the user in CSDL, includes promoted and unpromoted names of all independent inputs in the model
            self.available_input_names  = [item[0] for item in self.problem.list_indep_vars(out_stream=None)] # returns unique promoted names

            # For all input variables, map promoted names to the list of all corresponding unpromoted names 
            self.input_name_map = {} 
            for key, item in self.all_input_meta.items():
                # Generate a dictionary that does the name mapping for all input variables
                if item['prom_name'] in self.input_name_map:
                    self.input_name_map[item['prom_name']] += [key] 
                else:
                    self.input_name_map[item['prom_name']]  = [key]
                
                # Include all unpromoted names of the input variables in the current available_input_names list of only promoted names
                if item['prom_name'] in self.available_input_names:
                    if key not in self.available_input_names:
                        self.available_input_names.append(key)

            # All available output names for the user in CSDL, includes promoted and unpromoted names of all non-independent outputs in the model
            self.available_output_names = copy.deepcopy(self.all_output_names)

            # Remove the names of the available input variables from the available_output_names list
            self.available_output_names = list(set(self.available_output_names) - set(self.available_input_names))

            # For all output variables, map promoted names to the list of all corresponding unpromoted names
            self.output_name_map = {}
            for key, item in self.all_output_meta.items():
                # Generate a dictionary that does the name mapping for all output variables
                if item['prom_name'] in self.output_name_map:
                    self.output_name_map[item['prom_name']] += [key]
                else:
                    self.output_name_map[item['prom_name']]  = [key]

                # Include all promoted names of the output variables in the current available_output_names list of only unpromoted names
                if item['prom_name'] not in self.available_output_names:
                    self.available_output_names.append(item['prom_name'])

    def evaluate(self, inputs: Dict[str, csdl.Variable], out_names: list):
        """
        Connect the provided input variables in CSDL to the corresponding inputs in the OpenMDAO model 
        and return the listed outputs as CSDL variables from the OpenMDAO model.

        Parameters
        ----------
        inputs : Dict[str, csdl.Variable]
            A dictionary of input variables with promoted/unpromoted names as keys and csdl.Variable objects as values.
        out_names : list[str]
            A list of output names that the user wants to expose as CSDL variables.

        Returns
        -------
        Dict[str, csdl.Variable]
            A dictionary of output variables with `out_names` as keys and csdl.Variable objects as values.
        """
        # Raise error if any provided input is not an available input in the OpenMDAO model
        unavailable_inputs = set(inputs.keys()) - set(self.available_input_names)
        if len(unavailable_inputs) > 0:
            raise ValueError(f'The following input variables are not independent variables in the OpenMDAO model: {unavailable_inputs}. '
                             f'\nPlease provide valid input names from: {self.available_input_names}.')
        
        # Assign the method inputs to an input dictionary
        for in_name in inputs.keys():
            self.declare_input(in_name, inputs[in_name])

        # Set in_names as the keys of the user's inputs dictionary (only a subset of available_input_names)
        self.in_names = list(inputs.keys())

        # List of corresponding promoted names for the in_names to be used for compute_totals
        self.prom_in_names = copy.deepcopy(self.in_names)
        for i, prom_in_name in enumerate(self.prom_in_names):
            if prom_in_name not in self.input_name_map:
                self.prom_in_names[i] = self.all_input_meta[prom_in_name]['prom_name']

        # Raise error if any provided out_name is not an available output in the OpenMDAO model
        unavailable_outputs = set(out_names) - set(self.available_output_names)
        if len(unavailable_outputs) > 0:
            raise ValueError(f'The following output variables are not available in the OpenMDAO model: {unavailable_outputs}. '
                             f'\nPlease provide valid output names from: {self.available_output_names}.')
        
        # Set outputs names as the list is provided by the user (only a subset of available_output_names)
        self.out_names = out_names

        # List of corresponding promoted names for the out_names to be used for compute_totals
        self.prom_out_names = copy.deepcopy(self.out_names)
        for i, prom_out_name in enumerate(self.prom_out_names):
            if prom_out_name not in self.output_name_map:
                self.prom_out_names[i] = self.all_output_meta[prom_out_name]['prom_name']
    
        # Construct output variables for the CustomExplicitOperation
        outputs = {}
        for out_name in self.out_names:
            # Note that all_output_meta contains the metadata with full names
            if out_name not in self.all_output_meta:    # in case user provides a promoted name
                full_out_name = self.output_name_map[out_name][0]
            else:
                full_out_name = out_name
            outputs[out_name] = self.create_output(out_name, self.all_output_meta[full_out_name]['shape'])

        return outputs
    
    def compute(self, input_vals, output_vals):
        """
        Compute the output values of the CustomExplicitOperation using the provided input values.
        The ``input_vals`` dictionary contains the values of the input variables as numpy arrays, 
        and the ``output_vals`` dictionary is used to store the output values. 
        The ``compute`` method should assign the output values to the ``output_vals`` dictionary. 
        The keys of these dictionaries correspond to the strings given in 
        the ``declare_input()`` and ``create_output()`` methods.
        """
        # Set values of the input variables in the OpenMDAO model
        for in_name in self.in_names:
            self.problem.set_val(in_name, input_vals[in_name]) # or self.problem[in_name] = input_vals[in_name]

        # Run the OpenMDAO model
        self.problem.run_model()

        # Assign the outputs from the OpenMDAO model to the output values for the CustomExplicitOperation
        for out_name in self.out_names:
            output_vals[out_name] = self.problem[out_name]

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        """
        Compute the derivatives of the output variables with respect to the input variables.
        The ``input_vals`` dictionary contains the values of the input variables as numpy arrays, 
        the ``output_vals`` dictionary contains the values of the output variables,
        and the ``derivatives`` dictionary is used to store the derivative values.
        The ``compute_derivatives`` method should assign the computed derivatives to the ``derivatives`` dictionary. 
        The keys of the ``derivatives`` dictionary are tuples of the form ``(output_name, input_name)``,
        where ``output_name`` and ``input_name`` are the strings given in 
        the ``create_output()`` and ``declare_input()`` methods.
        """
        # Set values of the input variables in the OpenMDAO model
        for in_name in self.in_names:
            self.problem.set_val(in_name, input_vals[in_name]) # or self.problem[in_name] = input_vals[in_name]

        # Use compute_totals() since CSDL's CustomExplicitOperation currently only supports full Jacobian for the derivatives.
        # TODO: Add support for derivative vector products in the future when CSDL supports it.
        # Note: Only promoted names can be used to compute the total derivatives in OpenMDAO
        totals = self.problem.compute_totals(of=self.prom_out_names, wrt=self.prom_in_names, return_format='flat_dict')
        
        # Assign the totals computed by the OpenMDAO model to the derivatives for the CustomExplicitOperation
        for (prom_out_name, out_name) in zip(self.prom_out_names, self.out_names):
            for (prom_in_name, in_name) in zip(self.prom_in_names, self.in_names):
                derivatives[out_name, in_name] = totals[prom_out_name, prom_in_name]

if __name__ == '__main__':

    para_comp = om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3')

    para_group = om.Group()
    # It is only optional to promote the variables to the top level group
    para_group.add_subsystem('paraboloid', para_comp, promotes=['*'])

    para_prob = om.Problem()
    para_prob.model = para_group
    para_prob.setup()

    # The following code should run without any errors.
    class QuarticComp(om.ExplicitComponent):
        def setup(self): 
            # add_inputs
            self.add_input('x', 1.)
            self.add_input('y', 1.)
            
            # add_outputs
            self.add_output('objective')
            self.add_output('constraint_1')
            self.add_output('constraint_2')

            # declare_partials
            self.declare_partials(of='objective', wrt='*')
            self.declare_partials(of='constraint_1', wrt='x', val=1.)
            self.declare_partials(of='constraint_1', wrt='y', val=1.)
            self.declare_partials(of='constraint_2', wrt='x', val=1.)
            self.declare_partials(of='constraint_2', wrt='y', val=-1.)

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

    quartic_group = om.Group()
    # It is only optional to promote the variables to the top level group
    quartic_group.add_subsystem('quartic', QuarticComp(), promotes=['*'])

    quartic_prob = om.Problem()
    quartic_prob.model = quartic_group
    quartic_prob.setup()

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inputs = {}

    inputs['x'] = csdl.Variable(value=1.0, name='x')
    inputs['y'] = csdl.Variable(value=1.0, name='y')

    inputs['x'].set_as_design_variable(lower=0.0)
    inputs['y'].set_as_design_variable()

    # 1.1  OpenMDAOExplicitOperation from an ExecComp/ExplicitComponent object 
    # paraboloid = OpenMDAOExplicitOperation(para_comp)

    # 1.2  OpenMDAOExplicitOperation from an OpenMDAO group
    # paraboloid = OpenMDAOExplicitOperation(para_group)

    # 1.3  OpenMDAOExplicitOperation from an OpenMDAO problem
    # paraboloid = OpenMDAOExplicitOperation(para_prob)
    
    # outputs = paraboloid.evaluate(inputs, ['f'])
    # outputs['f'].set_as_objective()
    # optimized_solution = {'x' : np.array([ 6.66665446, -7.33332915]), 'objective': 1.0, 'nfev': 7, 'njev': 7, 'nit':7}

    # 2.1 OpenMDAOExplicitOperation from an ExplicitComponent object
    # quartic = OpenMDAOExplicitOperation(QuarticComp())

    # 2.2 OpenMDAOExplicitOperation from an OpenMDAO group
    # quartic = OpenMDAOExplicitOperation(quartic_group)

    # 2.3 OpenMDAOExplicitOperation from an OpenMDAO problem
    quartic = OpenMDAOExplicitOperation(quartic_prob)

    outputs = quartic.evaluate(inputs, ['quartic.objective', 'quartic.constraint_1', 'quartic.constraint_2'])
    outputs['quartic.objective'].set_as_objective()
    outputs['quartic.constraint_1'].set_as_constraint(equals=1.0)
    outputs['quartic.constraint_2'].set_as_constraint(lower =1.0)
    optimized_solution = {'x' : np.array([1.00000000e+00, 1.11022302e-16]), 'objective': -27.33333333321794, 'nfev': 2, 'njev': 2, 'nit':2}


    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder)

    import modopt as mo
    prob = mo.CSDLAlphaProblem(problem_name='quartic', simulator=sim)
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter':20})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results(all=True)

    np.allclose(optimizer.results['x'], optimized_solution['x'], atol=1e-6)
    np.allclose(optimizer.results['fun'], optimized_solution['objective'], atol=1e-6)
    assert optimizer.results['nfev'] == optimized_solution['nfev']
    assert optimizer.results['njev'] == optimized_solution['njev']
    assert optimizer.results['nit'] == optimized_solution['nit']

    print('Tests passed!')