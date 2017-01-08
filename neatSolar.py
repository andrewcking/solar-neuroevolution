""" Using NEAT to predict Solar Radiation """

from neat import nn, population, statistics
import numpy as np
from numpy import genfromtxt

import visualize

#number of instances
M = 40880
#number of test instances
T = 20440
#number of attributes
N = 8


""" Training Matrix """
#pull in csv (prescaled)
csv = genfromtxt('solardata.csv', delimiter=',')

# create our data matrix (the first N rows)
a = csv[:,0:N]
sr_inputs = a.tolist()
print sr_inputs
# create output matrix (column M)
b = csv[:,N:N+1]
c = np.ravel(b)
sr_outputs = c.tolist()

""" Testing Matrix """
#pull in csv (prescaled)
testcsv = genfromtxt('test.csv', delimiter=',')

# create our data matrix (the first N rows)
ta = csv[:,0:N]
test_inputs = ta.tolist()

# create output matrix (column M)
tb = csv[:,N:N+1]
tc = np.ravel(tb)
test_outputs = tc.tolist()


def eval_fitness(genomes):
    for g in genomes:
        net = nn.create_feed_forward_phenotype(g)

        mean_absolute_error = 0.0
        for inputs, expected in zip(sr_inputs, sr_outputs):
            # Serial activation propagates the inputs through the entire network.
            output = net.serial_activate(inputs)
            mean_absolute_error += (abs(output[0] - expected))/M

        # When the output matches expected for all inputs, fitness will reach
        # its maximum value of 1.0.
        g.fitness = 0 - mean_absolute_error


pop = population.Population('solarrad_config')
pop.run(eval_fitness, 5)

print('Number of evaluations: {0}'.format(pop.total_evaluations))

# Display the most fit genome.
winner = pop.statistics.best_genome()
print('\nBest genome:\n{!s}'.format(winner))

# Verify network output against training data.
print('\nOutput:')
winner_net = nn.create_feed_forward_phenotype(winner)
for inputs, expected in zip(test_inputs, test_outputs):
    output = winner_net.serial_activate(inputs)
    print("expected {0:1.5f} got {1:1.5f}".format(expected, output[0]))


statistics.save_stats(pop.statistics)
statistics.save_species_count(pop.statistics)
statistics.save_species_fitness(pop.statistics)
