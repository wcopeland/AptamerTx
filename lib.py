import numpy as np
import pandas as pd


class Optimization(object):
    def __init__(self, model, observed, relation_frame, model_param_frame, param_range, t_0=0, t_max=100,
                 population_size=10, max_generations=10, F=0.8, CR=0.6):
        self.model = model
        self.observed = observed
        self.relation_frame = relation_frame
        self.model_param_frame = model_param_frame
        self.param_range = param_range

        self.t_0 = t_0
        self.t_max = t_max

        self.population = np.zeros(shape=(population_size, self.param_range.shape[0]))
        self.trial_population = np.zeros(shape=(population_size, self.param_range.shape[0]))

        self.fitness = np.zeros(population_size)
        self.trial_fitness = np.ones(population_size) * 1e12

        self.max_generations = max_generations
        self.F = F
        self.CR = CR

        # Initialize population and evaluate initial fitness
        self.initialize_random_population()
        self.population = np.apply_along_axis(self.verify_dependent_parameters, 1, self.population)
        self.fitness = self.get_fitness(chunk=np.arange(population_size), verbose=True)
        return

    def initialize_random_population(self):
        # Initialize parameter vector for optimization
        # We must convert range from log scale back to linear scale.
        for i in range(self.population.shape[0]):
            self.population[i] = np.power(10., [np.random.uniform(low, high) for (low, high) in self.param_range])
        return

    def verify_dependent_parameters(self, vector, verbose=False):
        '''
        :param vector:
        :param verbose:
        @rtype: np.ndarray
        '''
        changes = []
        for i in self.relation_frame.index:
            target_idx = int(self.relation_frame.loc[i, 'target'].replace('{', '').replace('}', ''))

            # Get range for the first operand.
            if isinstance(self.relation_frame.loc[i, 'operand_1_min'], (str, unicode)):
                min_1 = vector[int(self.relation_frame.loc[i, 'operand_1_min'].replace('{', '').replace('}', ''))]
            else:
                min_1 = self.relation_frame.loc[i, 'operand_1_min']

            if isinstance(self.relation_frame.loc[i, 'operand_1_max'], (str, unicode)):
                max_1 = vector[int(self.relation_frame.loc[i, 'operand_1_max'].replace('{', '').replace('}', ''))]
            else:
                max_1 = self.relation_frame.loc[i, 'operand_1_max']

            # Get range for the second operand.
            if isinstance(self.relation_frame.loc[i, 'operand_2_min'], (str, unicode)):
                min_2 = vector[int(self.relation_frame.loc[i, 'operand_2_min'].replace('{', '').replace('}', ''))]
            else:
                min_2 = self.relation_frame.loc[i, 'operand_2_min']

            if isinstance(self.relation_frame.loc[i, 'operand_2_max'], (str, unicode)):
                max_2 = vector[int(self.relation_frame.loc[i, 'operand_2_max'].replace('{', '').replace('}', ''))]
            else:
                max_2 = self.relation_frame.loc[i, 'operand_2_max']

            # If the value does not adhere to the dependent relationship, randomly update the value.
            op_func = self.relation_frame.loc[i, 'operator']
            if not op_func(min_1, min_2) <= vector[target_idx] <= op_func(max_1, max_2):
                if min_1 == max_1:
                    op_1 = min_1
                else:
                    op_1 = 10. ** np.random.uniform(np.log10(min_1), np.log10(max_1))

                if min_2 == max_2:
                    op_2 = min_2
                else:
                    op_2 = 10. ** np.random.uniform(np.log10(min_2), np.log10(max_2))

                vector[target_idx] = op_func(op_1, op_2)
                changes.append((target_idx, vector[target_idx]))

        if verbose:
            print(changes)
        return vector

    def run(self, num_cpus=1):

        generation_count = 0
        while generation_count < self.max_generations:

            # Create trial vectors
            if num_cpus == 1:
                chunk = np.arange(self.population.shape[0])
                self.evolve_generation(chunk)
            else:
                chunks = []
                raise NotImplementedError

            # Replace population vector and fitness values with trial results where appropriate.
            replace_idx = (self.trial_fitness < self.fitness)
            self.fitness = self.trial_fitness * replace_idx + self.fitness * ~replace_idx

            replace_idx = replace_idx[:, np.newaxis]
            self.population = self.trial_population * replace_idx + self.population * ~replace_idx

            generation_count += 1
        return

    def evolve_generation(self, chunk):
        # Generate random seed within the function. This helps multi CPU runs avoid
        # choosing the same random values.
        np.random.seed()

        self.create_challengers(chunk)
        return

    def simulate_model(self, parameter_vector):
        # Iteratively generate simulation results for each model.
        simulation_results = pd.DataFrame(columns=['Time [s]', 'Model', 'Simulated FL'])
        for model_name in self.model_param_frame.index:

            # Parameterize the model: Draw values from shared parameter vector.
            for param_id, param_value in self.model_param_frame.loc[model_name, :].iteritems():
                if isinstance(param_value, (str, unicode)):
                    value_idx = int(param_value.replace('{', '').replace('}', ''))
                    self.model.setValue(param_id, parameter_vector[value_idx])
                else:
                    self.model.setValue(param_id, param_value)

            # Initialize species concentrations
            self.model.setValue('rna', 0.)
            self.model.setValue('dye', 0.)
            self.model.setValue('bound', 0.)
            self.model.setValue('dye_ext', 0.)

            # Solve for steady state concentrations prior to malachite green induction.
            self.model.steadyState()

            # Modify external dye concentration to match the amount of malachite green that is pulsed into the well.
            self.model.setValue('dye_ext', self.model_param_frame.loc[model_name, 'dye_ext'])

            # Simulate the transient concentration of dye-bound rna (ie. bound) after malachite green exposure.
            model_result = self.model.simulate(self.t_0, self.t_max, int(self.t_max + 1),
                                               selections=['time', 'bound'])
            model_result = pd.DataFrame(model_result, columns=['Time [s]', 'bound'])

            # Get approximate FL value.
            model_result['Simulated FL'] = model_result['bound'] * self.model.getValue('ci')

            # Annotate which model this corresponds to.
            model_result['Model'] = model_name

            # Add model results to all simulation results.
            simulation_results = pd.concat(
                (simulation_results, model_result[['Time [s]', 'Model', 'Simulated FL']]))

        return simulation_results

    def get_fitness(self, chunk=None, group=None, verbose=False):
        '''
        @param chunk: row indices corresponding to a subset of the population
        @type chunk: np.ndarray or list or tuple
        @param group: two-dimensional array of parameter values
        @type group: np.ndarray
        :return:
        '''

        if group is None:
            if not chunk is None:
                group = self.population[chunk, :]
            else:
                msg = 'Either group or chunk argument must be specified.'
                raise ValueError(msg)

        fitness_values = np.ones(shape=group.shape[0])

        for i, vector in enumerate(group):
            # Generate simulation results for each model and store in a single DataFrame.
            simulation_results = self.simulate_model(vector)

            # Merged simulated and observed data.
            obs_exp_frame = pd.merge(simulation_results, self.observed, on=['Time [s]', 'Model'])

            # chi_squared = np.power((obs_exp_frame['MGA5S FL'] - obs_exp_frame['Simulated FL']), 2.0) / obs_exp_frame['Simulated FL']
            diff_squared = np.power((obs_exp_frame['MGA5S FL'] - obs_exp_frame['Simulated FL']), 2.0)

            # Get weights
            weights = obs_exp_frame['Time [s]'].apply(lambda o: np.power(1. - o / 240., 0.4))

            # Sum of weighted differences between observed and expected.
            sum_weighted_diff_squared = (diff_squared *weights).sum()

            # Update fitness values.
            fitness_values[i] = sum_weighted_diff_squared
        return fitness_values

    def create_challengers(self, chunk):
        '''
        @param chunk:
        @type chunk: np.ndarray
        :return:
        '''
        selected_parents = []
        for i in range(chunk.shape[0]):
            selected_parents.append(
                [x for x in np.random.choice(self.population.shape[0], 4, replace=False) if x != chunk[i]][:3]
            )
        selected_parents = np.asarray(selected_parents)

        O = self.population[chunk]
        A = self.population[selected_parents[:, 0]]
        B = self.population[selected_parents[:, 1]]
        C = self.population[selected_parents[:, 2]]

        # Create a new set of vectors based on a combination of 3 randomly selected vectors.
        N = A + self.F * (B - C)

        # Create a mask for crossover events. Note that a value of 1 in the crossover matrix will
        # signify using the value from N and a value of 0 in the crossover matrix will signify
        # using the value from O.
        CRM = np.random.choice([0, 1], O.shape, p=[1-self.CR, self.CR])

        # Exclude negative values.
        # If a combined value yields a negative number, divide the original vector value by 2
        # and then set the crossover flag to choose the modified original value.
        neg_locs = np.where(N < 0)
        O[neg_locs] /= 2.
        CRM[neg_locs] = 0

        # Create new trial members
        T = CRM * N + np.abs(CRM - 1) * O

        # Store the vector and fitness for new members.
        # Note: It is important to check if all parameter values are within the specified
        # range. If they are not, they will be randomly reassigned within the fitness function.
        assert isinstance(T, np.ndarray)
        self.trial_population[chunk, :] = T
        T = np.apply_along_axis(self.verify_dependent_parameters, 1, T)

        self.trial_fitness[chunk] = self.get_fitness(group=T)
        return

