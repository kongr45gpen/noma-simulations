import itertools
import datetime
import tqdm
import tabulate


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
        self.__last_table_lines = -1

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

            last_update_time = datetime.datetime.now()

            #for iteration in tqdm.trange(self.max_iterations, postfix=parameter_string):
            for iteration in range(self.max_iterations):
                context["iteration"] = iteration
                self.run_one(parameter_set, context)

                if datetime.datetime.now() - last_update_time > datetime.timedelta(milliseconds=100):
                    self.print_table()
                    last_update_time = datetime.datetime.now()

            if self.finish:
                self.finish(parameter_set, context)

            self.print_table()

    def run_one(self, parameters, context):
        return self.function(parameters, context)

    def print_table(self):
        parameter_sets = itertools.product(*self.parameters.values())

        headers = list(self.parameters.keys()) + ["Progress"] + self.interesting_fields
        table = []

        for parameter_tuple in parameter_sets:
            if hash(parameter_tuple) not in self.context:
                continue

            # Sanitize parameter set, including variable names
            parameter_set = {list(self.parameters.keys())[i]: v for i, v in enumerate(parameter_tuple)}

            context = self.context[hash(parameter_tuple)]
            percentage = "{:8.3f}".format(100 * context["iteration"] / (self.max_iterations - 1))
            table.append(list(parameter_tuple) + [percentage]
                         + list(context[k] for k in self.interesting_fields))

        tabulate.PRESERVE_WHITESPACE = True
        table_text = tabulate.tabulate(table, headers=headers, showindex="always", tablefmt="orgtbl", disable_numparse=True, stralign="right")

        if self.__last_table_lines >= 0:
            print("\x1B[{}A".format(self.__last_table_lines + 1), end='')
            print("\x1B[0K\r", end='')
        print(table_text)
        self.__last_table_lines = table_text.count('\n')


