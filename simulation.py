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
                 interesting_fields=None, initial_global_context={}):
        if interesting_fields is None:
            interesting_fields = ["iteration"]
        self.parameters = parameters
        self.function = function
        self.initialize = initialize
        self.finish = finish
        self.max_iterations = max_iterations
        self.interesting_fields = interesting_fields
        self.context = {
            "global": initial_global_context
        }
        self.__last_table_lines = -1

    def run(self, display_table=True):
        parameter_sets = itertools.product(*self.parameters.values())

        futures = []

        manager = multiprocessing.Manager()

        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            self.context["global"] = manager.dict(self.context["global"])

            for parameter_tuple in parameter_sets:
                # Sanitize parameter set, including variable names
                parameter_set = {list(self.parameters.keys())[i]: v for i, v in enumerate(parameter_tuple)}
                # Create a string explaining the current parameter set
                parameter_string = ', '.join(map(lambda kv: "{}: {}".format(kv[0], kv[1]), parameter_set.items()))

                context = self.context[hash(parameter_tuple)] = manager.dict({
                    "max_iterations": self.max_iterations
                })

                future = executor.submit(self.run_parameter_set, parameter_set, context, self.context["global"], manager.Lock())
                future.add_done_callback(Simulation.__run__callback)

                futures.append(future)
                log.debug("Submitting job {}".format(parameter_tuple))

            while True:
                finished, pending = concurrent.futures.wait(futures, timeout=0.1,
                                                            return_when=concurrent.futures.ALL_COMPLETED)

                if display_table:
                    self.print_table(len(pending), len(finished))

                if len(pending) == 0:
                    break

        # self.print_table()

    @staticmethod
    def __run__callback(future):
        if future.exception() is not None:
            err = future.exception()
            log.error("Task ended in error: {}".format(err))
            future.result()

    def run_parameter_set(self, parameter_set, context, global_context, gc_lock):
        if self.initialize:
            self.initialize(parameter_set, context)
        # Create a local context that can be edited without annoying the parent context
        local_context = copy.deepcopy(context)

        # Run each iteration
        for iteration in range(local_context["max_iterations"]):
            local_context["iteration"] = iteration
            self.run_one(parameter_set, local_context)

            # After some time has passed, update the global context (which acquires the lock and slows down the other threads)
            if iteration % 100 == 0:
                for key, value in local_context.items():
                    context[key] = value

        # For the final iteration, copy the context once again
        for key, value in local_context.items():
            context[key] = value

        if self.finish:
            with gc_lock:
                local_global_context = copy.deepcopy(global_context)
                self.finish(parameter_set, context, local_global_context)
                for key, value in local_global_context.items():
                    global_context[key] = value

    def run_one(self, parameters, context):
        return self.function(parameters, context)

    def print_table(self, pending, finished):
        parameter_sets = itertools.product(*self.parameters.values())

        headers = list(self.parameters.keys()) + ["Progress"] + self.interesting_fields
        table = []

        for parameter_tuple in parameter_sets:
            if hash(parameter_tuple) not in self.context or self.context[hash(parameter_tuple)] == {}:
                continue

            # Sanitize parameter set, including variable names
            parameter_set = {list(self.parameters.keys())[i]: v for i, v in enumerate(parameter_tuple)}

            context = copy.deepcopy(self.context[hash(parameter_tuple)])

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
            print("\x1B[{}A".format(self.__last_table_lines + 2), end='')
            print("\x1B[0K\r", end='')
        print("Task progress: {}/{}".format(finished, finished + pending))
        print(table_text)
        self.__last_table_lines = table_text.count('\n')

    def get_global_context(self):
        return self.context["global"]
