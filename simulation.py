import itertools
import tqdm


class Simulation:
    def __init__(self, parameters, function, finish=None, max_iterations=10000):
        self.parameters = parameters
        self.function = function
        self.finish = finish
        self.max_iterations = max_iterations
        self.context = {}

    def run(self):
        parameter_sets = itertools.product(*self.parameters.values())

        for parameter_set in parameter_sets:
            # Sanitize parameter set, including variable names
            parameter_set = {list(self.parameters.keys())[i]: v for i, v in enumerate(parameter_set)}
            # Create a string explaining the current parameter set
            parameter_string = ', '.join(map(lambda kv: "{}: {}".format(kv[0], kv[1]), parameter_set.items()))

            for iteration in tqdm.trange(self.max_iterations, postfix=parameter_string):
                self.context["iteration"] = iteration
                self.run_one(parameter_set)

            if self.finish:
                self.finish(parameter_set, self.context)

    def run_one(self, parameters):
        return self.function(parameters, self.context)
