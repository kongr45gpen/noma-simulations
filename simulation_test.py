from simulation import Simulation


def simulator(parameters):
    pass
    #print(parameters)


simulation = Simulation(parameters={
    'SNR': [0, 10, 20, 30],
    'τ': [0.6, 1]
}, function=simulator)
simulation.run()
