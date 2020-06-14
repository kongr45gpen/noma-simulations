import itertools
import datetime
import tqdm
import tabulate
import multiprocessing
import concurrent.futures
import time
import logging
import copy
import traceback

log = logging.getLogger(name='python-simulation')
log.setLevel(logging.DEBUG)


class Simulation:
    def __init__(self, parameters, function, finish=None, initialize=None, max_iterations=10000,
                 interesting_fields=None):
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
        self.__lock = multiprocessing.Lock()

    def run(self):
        parameter_sets = itertools.product(*self.parameters.values())

        futures = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            for parameter_tuple in parameter_sets:
                # Sanitize parameter set, including variable names
                parameter_set = {list(self.parameters.keys())[i]: v for i, v in enumerate(parameter_tuple)}
                # Create a string explaining the current parameter set
                parameter_string = ', '.join(map(lambda kv: "{}: {}".format(kv[0], kv[1]), parameter_set.items()))

                context = self.context[hash(parameter_tuple)] = {
                    "max_iterations": self.max_iterations
                }

                future = executor.submit(self.run_parameter_set, parameter_set, context)
                future.add_done_callback(Simulation.__run__callback)

                futures.append(future)
                log.debug("Submitting job {}".format(parameter_tuple))

            while True:
                finished, pending = concurrent.futures.wait(futures, timeout=0.1,
                                                            return_when=concurrent.futures.ALL_COMPLETED)

                log.debug("[{}, {}]".format(len(finished), len(pending)))

                # self.print_table()

                if len(pending) == 0:
                    break

        # self.print_table()

    @staticmethod
    def __run__callback(future):
        if future.exception() is not None:
            err = future.exception()
            log.error("Task ended in error: {}".format(err))

        else:
            log.debug("Task done")

    def run_parameter_set(self, parameter_set, context):
        with self.__lock:
            if self.initialize:
                self.initialize(parameter_set, context)
            local_context = copy.copy(context)

        # for iteration in tqdm.trange(self.max_iterations, postfix=parameter_string):
        for iteration in range(local_context["max_iterations"]):
            print("iter")
            context["iteration"] = iteration
            self.run_one(parameter_set, context)

            # if iteration % 10 == 0:
            # context = copy.copy(local_context)

        with self.__lock:
            # context = copy.copy(local_context)
            if self.finish:
                self.finish(parameter_set, context)

        return 5

    def run_one(self, parameters, context):
        return self.function(parameters, context)

    def print_table(self):
        parameter_sets = itertools.product(*self.parameters.values())

        headers = list(self.parameters.keys()) + ["Progress"] + self.interesting_fields
        table = []

        for parameter_tuple in parameter_sets:
            if hash(parameter_tuple) not in self.context or self.context[hash(parameter_tuple)] == {}:
                continue

            # Sanitize parameter set, including variable names
            parameter_set = {list(self.parameters.keys())[i]: v for i, v in enumerate(parameter_tuple)}

            with self.__lock:
                context = self.context[hash(parameter_tuple)]

                if 'iteration' not in context:
                    continue

                percentage = "{:8.3f}".format(100 * context["iteration"] / (context["max_iterations"] - 1))
                table.append(list(parameter_tuple) + [percentage] + list(
                    context[k] if k in context else '' for k in self.interesting_fields))
            # context = {
            #     "Outage rate (u)": 5,
            #     "Outage rate (v)": 6,
            #     "iteration": 1000
            # }

        # tabulate.PRESERVE_WHITESPACE = True
        table_text = tabulate.tabulate(table, headers=headers, showindex="always", tablefmt="orgtbl",
                                       disable_numparse=True, stralign="right")

        if self.__last_table_lines >= 0:
            print("\x1B[{}A".format(self.__last_table_lines + 1), end='')
            print("\x1B[0K\r", end='')
        print(table_text)
        self.__last_table_lines = table_text.count('\n')
