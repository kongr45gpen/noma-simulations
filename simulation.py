import itertools
import tqdm
from tabulate import tabulate


class Simulation:
    def __init__(self, parameters, function, finish=None, initialize=None, max_iterations=10000, interesting_fields=None):
        if interesting_fields is None:
            interesting_fields = ["iteration"]
        self.parameters = parameters
        self.function = function
        self.initialize = initialize
        self.finish = finish
        self.max_iterations = max_iterations
        self.interesting_fields = interesting_fields
        self.context = {}

    def run(self):
        parameter_sets = itertools.product(*self.parameters.values())

        for parameter_tuple in parameter_sets:
            # Sanitize parameter set, including variable names
            parameter_set = {list(self.parameters.keys())[i]: v for i, v in enumerate(parameter_tuple)}
            # Create a string explaining the current parameter set
            parameter_string = ', '.join(map(lambda kv: "{}: {}".format(kv[0], kv[1]), parameter_set.items()))

            context = self.context[hash(parameter_tuple)] = {}

            if self.initialize:
                self.initialize(parameter_set, context)

            for iteration in tqdm.trange(self.max_iterations, postfix=parameter_string):
                context["iteration"] = iteration
                self.run_one(parameter_set, context)

            if self.finish:
                self.finish(parameter_set, context)

            self.print_table()

    def run_one(self, parameters, context):
        return self.function(parameters, context)

    def print_table(self):
        parameter_sets = itertools.product(*self.parameters.values())

        headers = list(self.parameters.keys()) + self.interesting_fields
        table = []

        for parameter_tuple in parameter_sets:
            if hash(parameter_tuple) not in self.context:
                continue

            # Sanitize parameter set, including variable names
            parameter_set = {list(self.parameters.keys())[i]: v for i, v in enumerate(parameter_tuple)}

            context = self.context[hash(parameter_tuple)]
            table.append(list(parameter_tuple) + list(context[k] for k in self.interesting_fields))

        print(headers)
        print(tabulate(table, headers=headers, showindex="always"))
