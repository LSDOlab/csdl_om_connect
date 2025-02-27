import numpy as np
import warnings
import openmdao.api as om
from openmdao.core.constants import _SetupStatus
import csdl_alpha as csdl
from typing import List, Dict, Union
import copy

# Notes:
# 1. The CSDL CustomExplicitOperation only supports full Jacobian computation.
# 2. The names of input/output variable keys in the evaluate method for the CSDL CustomExplicitOperation
#    are the PROMOTED/UNPROMOTED names of the variables in the OpenMDAO model.
# 3. Problem objects need to call the setup() method before running the model.
class OpenMDAOExplicitOperation(csdl.CustomExplicitOperation):
    '''
    Class for creating a CSDL CustomExplicitOperation from an OpenMDAO ExplicitComponent, Group, or Problem object.

    Parameters
    ----------
    model : Union[om.ExplicitComponent, om.Group]
        An OpenMDAO model in the form of an ExplicitComponent, Group, or Problem object.
        This model will be transformed into a CSDL CustomExplicitOperation object with 
        the option for user to expose the model inputs and outputs as CSDL variables using
        the PROMOTED/UNPROMOTED names of the variables in the OpenMDAO model.
    solver_print_level : int, default=0
        The print level for the solver in the OpenMDAO Problem object.

    Attributes
    ----------
    model : Union[om.ExplicitComponent, om.Group]
        The OpenMDAO model in the form of an ExplicitComponent, or Group object.
    problem : om.Problem
        The OpenMDAO Problem object initialized with the given OpenMDAO model.
        If instantiated with an OpenMDAO Problem object, this attribute will be the same as the input.
    all_input_meta : Dict[str, Dict]
        A dictionary of all input metadata from the OpenMDAO model with promoted/unpromoted names as keys.
    all_output_meta : Dict[str, Dict]
        A dictionary of all output metadata from the OpenMDAO model with promoted/unpromoted names as keys.
    input_name_map : Dict[str, List[str]]
        A dictionary that maps the promoted names to the unpromoted names of the input variables in the OpenMDAO model.
    output_name_map : Dict[str, List[str]]
        A dictionary that maps the promoted names to the unpromoted names of the output variables in the OpenMDAO model.
    all_input_names : List[str]
        A list of all input names from the OpenMDAO model with promoted/unpromoted names.
    all_output_names : List[str]
        A list of all output names from the OpenMDAO model with promoted/unpromoted names.
    available_input_names : List[str]
        A list of all available input names from the OpenMDAO model with promoted/unpromoted names
        that can be independently assigned values through csdl.Variable objects, if exposed by the user.
    available_output_names : List[str]
        A list of all available output names from the OpenMDAO model with promoted/unpromoted names
        that can be accessed and used as CSDL variables, if exposed by the user.
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

            # raise an error if the input is an OpenMDAO ExecComp object # not required if we create a Problem and run the model
            # if isinstance(model, om.ExecComp):
            #     raise ValueError('The input cannot be an OpenMDAO ExecComp object. Please provide a fully defined OpenMDAO ExplicitComponent object.')
            
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
                        raise ValueError('The setup() method has not been called for provided the OpenMDAO Problem object. '
                                        'Please call the setup() before passing in the OpenMDAO problem object.')
                                          
            # Call setup for the problem if the model is an ExplicitComponent or a Group
            if not isinstance(model, om.Problem):
                self.problem.setup()
            
            # list_indep_vars requires that final_setup has been run for the Problem.
            self.problem.final_setup()

            # Set the solver print level for the OpenMDAO Problem object
            self.problem.set_solver_print(level=solver_print_level)

            # Get all input and output names from the model along with their promoted names
            self.all_input_meta  = self.problem.model.list_inputs(return_format='dict', out_stream=None, shape=True)
            self.all_output_meta = self.problem.model.list_outputs(return_format='dict', out_stream=None, shape=True) 

            # # # Replace all input and output names from the model as promoted names - no longer used
            # self.all_input_meta  = {item['prom_name']: self.all_input_meta[key] for key, item in self.all_input_meta.items()}
            # self.all_output_meta = {item['prom_name']: self.all_output_meta[key] for key, item in self.all_output_meta.items()}

            self.all_input_names  = list(self.all_input_meta.keys())
            self.all_output_names = list(self.all_output_meta.keys())

            self.available_input_names  = [item[0] for item in self.problem.list_indep_vars(out_stream=None)] # returns unique promoted names
            self.input_name_map = {}
            for key, item in self.all_input_meta.items():
                # Generate a dictionary that maps the promoted names to all corresponding unpromoted names of the input variables.
                if item['prom_name'] in self.input_name_map:
                    self.input_name_map[item['prom_name']] += [key] 
                else:
                    self.input_name_map[item['prom_name']]  = [key]
                
                # Include all unpromoted names of the input variables in the available_input_names list
                if item['prom_name'] in self.available_input_names:
                    if key not in self.available_input_names:
                        self.available_input_names.append(key)
            
            self.available_output_names = copy.deepcopy(self.all_output_names)
            self.output_name_map = {}
            for key, item in self.all_output_meta.items():
                # Generate a dictionary that maps the promoted names to all corresponding unpromoted names of the output variables.
                if item['prom_name'] in self.output_name_map:
                    self.output_name_map[item['prom_name']] += [key]
                else:
                    self.output_name_map[item['prom_name']]  = [key]
                if item['prom_name'] not in self.available_output_names:
                    self.available_output_names.append(item['prom_name'])

            self.available_output_names = list(set(self.available_output_names) - set(self.available_input_names))

            # if isinstance(model, om.ExplicitComponent):
            #     self.in_names  = model._var_rel_names['input']
            #     self.out_names = model._var_rel_names['output']

    def evaluate(self, inputs: Dict[str, csdl.Variable], outputs: list):
        """
        Assign the input values to the OpenMDAO model, run the model, and return the output values as a dictionary.

        Parameters
        ----------
        inputs : Dict[str, csdl.Variable]
            A dictionary of input variables with promoted/unpromoted names as keys and csdl.Variable objects as values.
            The input variables are assigned to the OpenMDAO model before running the model.
        outputs : list, optional
            A list of output names that the user wants to expose as CSDL variables.
            # If not provided, all outputs from the OpenMDAO model will be exposed as CSDL variables.

        Returns
        -------
        Dict[str, csdl.Variable]
            A dictionary of output variables with promoted/unpromoted names as keys and csdl.Variable objects as values.
        """
        unavailable_inputs = set(inputs.keys()) - set(self.available_input_names)
        if len(unavailable_inputs) > 0:
            raise ValueError(f'The following input variables are not independent variables in the OpenMDAO model: {unavailable_inputs}. '
                             f'\nPlease provide valid input names from: {self.available_input_names}.')
        # for in_name in self.in_names: # only if user declares all inputs in the OpenMDAO model
        for in_name in inputs.keys():
            # if in_name not in self.available_input_names:
            #     raise ValueError(f'The input variable "{in_name}" is not an independent variable input in the OpenMDAO model. '
            #                      f'Please provide a valid input name from: {self.available_input_names}.')
            self.declare_input(in_name, inputs[in_name])

        # Set in_names to the keys of the inputs dictionary which is only a subset of available_input_names
        self.in_names = list(inputs.keys())

        # List of corresponding promoted names for the inputs to be used for compute_totals
        self.prom_in_names = copy.deepcopy(self.in_names)
        for i, prom_in_name in enumerate(self.prom_in_names):
            if prom_in_name not in self.input_name_map:
                self.prom_in_names[i] = self.all_input_meta[prom_in_name]['prom_name']

        unavailable_outputs = set(outputs) - set(self.available_output_names)
        if len(unavailable_outputs) > 0:
            raise ValueError(f'The following output variables are not available in the OpenMDAO model: {unavailable_outputs}. '
                             f'\nPlease provide valid output names from: {self.available_output_names}.')
        # for out_name in outputs:
        #     if out_name not in self.available_output_names:
        #         raise ValueError(f'The output variable "{out_name}" is not an available output in the OpenMDAO model. '
        #                          f'Please provide a valid output name from: {self.available_output_names}.')
        
        # Set outputs names as the list is provided by the user
        self.out_names = outputs

        # List of corresponding promoted names for the outputs to be used for compute_totals
        self.prom_out_names = copy.deepcopy(self.out_names)
        for i, prom_out_name in enumerate(self.prom_out_names):
            if prom_out_name not in self.output_name_map:
                self.prom_out_names[i] = self.all_output_meta[prom_out_name]['prom_name']
    
        # declare output variables and construct output of the model
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
        # The following does not work for OpenMDAO ExecComp objects without calling problem setup.
        # self.model.compute(input_vals, output_vals)

        for in_name in self.in_names:
            # self.problem[in_name] = input_vals[in_name]
            self.problem.set_val(in_name, input_vals[in_name])

        self.problem.run_model()

        for out_name in self.out_names:
            output_vals[out_name] = self.problem[out_name]

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        # The following does not work correctly as csdl provided derivatives dict contains the declared partials as zero.
        # self.model.compute_partials(input_vals, derivatives)

        # Use the following since CSDL's CustomExplicitOperation only supports full Jacobian for derivatives.
        for in_name in self.in_names:
            # self.problem[in_name] = input_vals[in_name]
            self.problem.set_val(in_name, input_vals[in_name])

        # Note: Only promoted names can be used to compute the derivatives in OpenMDAO
        totals = self.problem.compute_totals(of=self.prom_out_names, wrt=self.prom_in_names, return_format='flat_dict')
        
        for (prom_out_name, out_name) in zip(self.prom_out_names, self.out_names):
            for (prom_in_name, in_name) in zip(self.prom_in_names, self.in_names):
                derivatives[out_name, in_name] = totals[prom_out_name, prom_in_name]

if __name__ == '__main__':

    # The following code should raise an error as the input is an OpenMDAO ExecComp object.
    exec_comp = om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3')

    prob = om.Problem()
    prob.model.add_subsystem('paraboloid', exec_comp)
    exec_comp_model = prob.model

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

    prob = om.Problem()
    # optional to promote the variables to the top level group
    prob.model.add_subsystem('quartic', QuarticComp())
    quartic_model = prob.model

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inputs = {}

    inputs['quartic.x'] = csdl.Variable(value=1.0, name='x')
    inputs['quartic.y'] = csdl.Variable(value=1.0, name='y')

    inputs['quartic.x'].set_as_design_variable(lower=0.0)
    inputs['quartic.y'].set_as_design_variable()

    # 1.1  OpenMDAOExplicitOperation from an ExecComp object 
    # paraboloid = OpenMDAOExplicitOperation(exec_comp)

    # 1.2  OpenMDAOExplicitOperation from an OpenMDAO model with an ExecComp object
    # paraboloid = OpenMDAOExplicitOperation(exec_comp_model)
    
    # outputs = paraboloid.evaluate(inputs)
    # outputs['f'].set_as_objective()
    # optimized_solution = {'x' : np.array([ 6.66665446, -7.33332915]), 'objective': 1.0, 'nfev': 7, 'njev': 7, 'nit':7}

    # 2.1 OpenMDAOExplicitOperation from a fully explicitly defined ExplicitComponent object
    # quartic = OpenMDAOExplicitOperation(QuarticComp())

    # 2.2 OpenMDAOExplicitOperation from an OpenMDAO model with a fully explicitly defined ExplicitComponent object
    quartic = OpenMDAOExplicitOperation(quartic_model)

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