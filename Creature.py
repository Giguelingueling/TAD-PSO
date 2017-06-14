import numpy as np
import math


class Creature(object):
    def __init__(self, random, upper_bound, lower_bound, swarm_confidence=2.0, position=None,
                 fitness=float('Inf')):
        self._upper_bound = np.array(upper_bound, dtype='float64')
        self._lower_bound = np.array(lower_bound, dtype='float64')
        # Array containing the min and max possible for each position
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        # Get the length of each bound
        bound_distance = []
        for i in range(len(self._lower_bound)):
            bound_distance.append(self._upper_bound[i] - self._lower_bound[i])
        self._bound_distance = np.array(bound_distance)


        self._number_dimensions = len(self._upper_bound)
        self._random = random

        self._fitness = fitness
        if position is None:
            self._position = np.array(self.generate_vector_random(), dtype='float64')
        else:
            self._position = np.array(position, dtype='float64')
        self._velocity = np.array(self.generate_vector_random(), dtype='float64')
        self._max_velocity = .5*self._bound_distance

        self._best_memory_fitness = self._fitness
        self._best_memory_position = np.copy(self._position)

        self._refreshing_gap = 5
        self._stall_iteration = self._refreshing_gap+1

        self._max_weight_velocity = .9
        self._min_weight_velocity = .4

        self._swarm_confidence = swarm_confidence

        self._got_reset = False

        self._current_examplar = np.copy(self._best_memory_position)
        self._growth_weight_velocity = 1.

    # Generate the position or the velocity of the creature randomly
    def generate_vector_random(self):
        return (self._random.uniform(size=self._number_dimensions) *
                (self._upper_bound - self._lower_bound)) + self._lower_bound

    def get_best_memory_fitness(self):
        return self._best_memory_fitness

    def get_best_memory_position(self):
        return np.copy(self._best_memory_position)

    def get_position(self):
        position = []
        for i in range(len(self._position)):
            position.append(self._best_memory_position[i])
        return np.array(position)

    def get_velocity(self):
        return np.copy(self._velocity)

    def reset_fitness(self):
        self._fitness = float('Inf')

    def reset_memory(self, fitness_function, index_dimension_to_keep):
        self._best_memory_fitness = float('Inf')
        nmbr_eval = self.soft_reset_memory(fitness_function=fitness_function,
                                           index_dimension_to_keep=index_dimension_to_keep)
        self._best_memory_position = np.array(np.copy(self._position), dtype='float64')
        self._stall_iteration = self._refreshing_gap+1
        return nmbr_eval

    def soft_reset_memory(self, fitness_function, index_dimension_to_keep):
        self._got_reset = True
        random_vector = self.generate_vector_random()
        new_position = []
        for i in random_vector:
            new_position.append(i)
        if index_dimension_to_keep is not None:
            new_position[index_dimension_to_keep] = self._best_memory_position[index_dimension_to_keep]
        self._position = np.array(new_position, dtype='float64')
        self._fitness = float('Inf')
        self._stall_iteration = self._refreshing_gap + 1
        self.update_fitness(fitness_function=fitness_function)
        return 1.0

    def set_random_velocity_except_for_this_dimension(self, index_dimension_to_keep):
        random_vector = self.generate_vector_random()
        self._velocity = np.array(random_vector)
        self._velocity[index_dimension_to_keep] = 1.

    def need_new_examplar(self):
        return self._stall_iteration > self._refreshing_gap

    def new_examplar_created(self):
        self._stall_iteration = 0

    def update_velocity(self, current_gen, max_iter, examplar=None, fast_convergence=False):
        weight_velocity = self._max_weight_velocity - (((self._max_weight_velocity - self._min_weight_velocity) *
                                                        current_gen) / float(max_iter))

        inertia = weight_velocity * self._velocity

        if fast_convergence:

            c1 = (.5 - 2.5) * float(current_gen) / float(max_iter) + 2.5
            cognitive_component = np.ones(self._number_dimensions)

            #if self._random.uniform() > -.5:
            for i in range(self._number_dimensions):
                cognitive_component[i] = c1 * self._random.uniform() * (self._best_memory_position[i] -
                                                                        self._position[i])

            c2 = (2.5 - .5) * float(current_gen) / float(max_iter) + .5
            social_component = np.ones(self._number_dimensions)
            for i in range(self._number_dimensions):
                social_component[i] = c2 * self._random.uniform() * (examplar[i] - self._position[i])
            '''else:
                random = self._random.uniform()
                for i in range(self._number_dimensions):
                    cognitive_component[i] = c1 * random * (self._best_memory_position[i] -
                                                            self._position[i])

                c2 = (2.5 - .5) * float(current_gen) / float(max_iter) + .5
                social_component = np.ones(self._number_dimensions)
                random = self._random.uniform()
                for i in range(self._number_dimensions):
                    social_component[i] = c2 * random * (examplar[i] - self._position[i])'''
            velocity = inertia + np.copy(cognitive_component) + np.copy(social_component)
            for i in range(len(velocity)):
                #reinitialization_velocity = (
                #                            (.1 * self._max_velocity[i] - self._max_velocity[i]) * (float(current_gen) /
                #                                                                                    float(max_iter))) + \
                #                            self._max_velocity[i]
                #if abs(velocity[i]) <= 1e-14:
                #    if .5 < self._random.uniform():
                #        velocity[i] = reinitialization_velocity * self._random.uniform()
                #    else:
                #        velocity[i] = -1. * reinitialization_velocity * self._random.uniform()
                velocity[i] = cmp(velocity[i], 0.0) * min(abs(self._max_velocity[i]), abs(velocity[i]))
            #velocity = inertia + cognitive_component + social_component
            '''
            c1 = (.5 - 2.5) * float(current_gen) / float(max_iter) + 2.5
            cognitive_component = np.ones(self._number_dimensions)

            if self._random.uniform() > .5:
                for i in range(self._number_dimensions):
                    cognitive_component[i] = c1 * self._random.uniform() * (self._best_memory_position[i] -
                                                                            self._position[i])

                c2 = (2.5 - .5) * float(current_gen) / float(max_iter) + .5
                social_component = np.ones(self._number_dimensions)
                for i in range(self._number_dimensions):
                    social_component[i] = c2 * self._random.uniform() * (examplar[i] - self._position[i])
            else:
                random = self._random.uniform()
                for i in range(self._number_dimensions):
                    cognitive_component[i] = c1 * random * (self._best_memory_position[i] -
                                                                            self._position[i])

                c2 = (2.5 - .5) * float(current_gen) / float(max_iter) + .5
                social_component = np.ones(self._number_dimensions)
                random = self._random.uniform()
                for i in range(self._number_dimensions):
                    social_component[i] = c2 * random * (examplar[i] - self._position[i])

            velocity = inertia + cognitive_component + social_component
            '''
        else:
            random_value = self._random.rand()

            if examplar is None:
                swarm_influence = np.array(self._swarm_confidence * random_value *
                                           (np.array(self._current_examplar, dtype='float64') -
                                            np.array(self._position, dtype='float64')))
            else:
                swarm_influence = np.array(self._swarm_confidence*random_value *
                                           (np.array(examplar, dtype='float64')-np.array(self._position, dtype='float64')))
                self._current_examplar = np.copy(examplar)
            velocity = inertia + swarm_influence

            for i in range(len(velocity)):
                if abs(velocity[i]) > self._max_velocity[i]:
                    velocity[i] = cmp(velocity[i], 0.0)*self._max_velocity[i]
                #elif abs(velocity[i]) < 1e-14:
                #    if .5 < self._random.uniform():
                #        velocity[i] = self._max_velocity[i] * self._random.uniform()
                #    else:
                #        velocity[i] = -1. * self._max_velocity[i] * self._random.uniform()
        self._velocity = velocity

    def update_position(self):
        '''
        new_position = np.copy(self._position) + np.copy(self._velocity)
        # Verify if the new position is out of bound. If it's the case put the creature back in the function research
        # domain by using the reflect method (act as if the boundary are mirrors and the creature photons
        # and put inertia to 0.0 on this dimension. We use this method because it was the one shown to perform the best
        # on average by Helwig et al. (2013)
        for i in range(self._number_dimensions):
            # Make sure we don't go out of bound
            if new_position[i] > self._upper_bound[i]:
                new_position[i] = self._upper_bound[i] - (new_position[i] - self._upper_bound[i])
                self._velocity[i] = 0.0
                # Verify the edge case (extremely unlikely) that the creature goes over the other bound
                # If that's the case. Clamp the creature back to the domain
                if new_position[i] < self._lower_bound[i]:
                    new_position[i] = self._lower_bound[i]
            elif new_position[i] < self._lower_bound[i]:
                new_position[i] = self._lower_bound[i] + (self._lower_bound[i] - new_position[i])
                self._velocity[i] = 0.0
                # Verify the edge case (extremely unlikely) that the creature goes over the other bound
                # If that's the case. Clamp the creature back to the domain
                if new_position[i] > self._upper_bound[i]:
                    new_position[i] = self._upper_bound[i]
        '''
        self._position += np.copy(self._velocity)

    def update_fitness(self, fitness_function, examplar=None):
        for xi, upper_bound_i, lower_bound_i in zip(self._position, self._upper_bound, self._lower_bound):
            if xi > upper_bound_i or xi < lower_bound_i:
                #print "OUTSIDE"
                #print self._position, "       Velocity: ", self._velocity, "   EXAMPLAR: ", examplar
                return 0
        self._fitness = fitness_function(self._position)
        if self._fitness < self._best_memory_fitness:
            self._stall_iteration = 0
            self._best_memory_fitness = self._fitness
            new_best_position = []
            for i in self._position:
                new_best_position.append(i)
            self._best_memory_position = np.array(new_best_position)
        else:
            self._stall_iteration += 1

        return 1

    def update(self, current_eval, max_evals, fitness_function, examplar, fast_convergence):
        self._current_examplar = np.copy(examplar)
        self.update_velocity(current_eval, max_evals, examplar, fast_convergence)
        self.update_position()
        return self.update_fitness(fitness_function=fitness_function)
