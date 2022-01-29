import numpy as np
import math
import random
from matplotlib import pyplot as plt
# from IPython.display import clear_output


class MonteCarloTraining:

    def __init__(self):
        self._test_lamdas = [i * 0.05 for i in range(1, 61)]
        self._variances = []

    def get_test_lamdas(self):
        return self._test_lamdas

    def get_variances(self):
        return self._variances

    def get_rand_number(self, min_value, max_value):
        """
        This function gets a random number from a uniform distribution between
        the two input values [min_value, max_value] inclusively
        Args:
        - min_value (float)
        - max_value (float)
        Return:
        - Random number between this range (float)
        """
        range = max_value - min_value
        choice = random.uniform(0, 1)
        return min_value + range * choice

    def f_of_x(self, x):
        """
        This is the main function we want to integrate over.
        Args:
        - x (float) : input to function; must be in radians
        Return:
        - output of function f(x) (float)
        """
        return (math.e ** (-1 * x)) / (1 + (x - 1) ** 2)

    # def crude_monte_carlo(self, num_samples=1000):
    #     """
    #     This function performs the Crude Monte Carlo for our
    #     specific function f(x) on the range x=0 to x=5.
    #     Notice that this bound is sufficient because f(x)
    #     approaches 0 at around PI.
    #     Args:
    #     - num_samples (float) : number of samples
    #     Return:
    #     - Crude Monte Carlo estimation (float)
    #     """
    #     lower_bound = 0
    #     upper_bound = 5
    #     sum_of_samples = 0
    #     for i in range(num_samples):
    #         x = self.get_rand_number(lower_bound, upper_bound)
    #         sum_of_samples += self.f_of_x(x)
    #     return (upper_bound - lower_bound) * float(sum_of_samples / num_samples)
    #
    # def get_crude_mc_variance(self, num_samples):
    #     """
    #     This function returns the variance fo the Crude Monte Carlo.
    #     Note that the entered number of samples does not necessarily
    #     need to correspond to number of samples used in the Monte
    #     Carlo Simulation.
    #     Args:
    #     - num_samples (int)
    #     Return:
    #     - Variance for Crude Monte Carlo approximation of f(x) (float)
    #     """
    #     int_max = 5  # this is the max of our integration range
    #     # get the average of squares
    #     running_total = 0
    #     for i in range(num_samples):
    #         x = self.get_rand_number(0, int_max)
    #         running_total += self.f_of_x(x) ** 2
    #     sum_of_sqs = running_total * int_max / num_samples
    #     # get square of average
    #     running_total = 0
    #     for i in range(num_samples):
    #         x = self.get_rand_number(0, int_max)
    #         running_total = self.f_of_x(x)
    #     sq_ave = (int_max * running_total / num_samples) ** 2
    #     return sum_of_sqs - sq_ave

    def g_of_x(self, x, a, lamda):
        """template of weight function g(x)"""
        math.e = 2.71828
        return a * math.pow(math.e, -1 * lamda * x)

    def inverse_g_of_r(self, r, lamda):
        return (-1 * math.log(float(r))) / lamda

    def get_variance(self, lamda, num_samples):
        """
        This function calculates the variance if a Monte Carlo
        using importance sampling.
        Args:
        - lamda (float) : lamdba value of g(x) being tested
        Return:
        - Variance
        """
        a = lamda
        int_max = 5
        # get sum of squares
        running_total = 0
        for i in range(num_samples):
            x = self.get_rand_number(0, int_max)
            running_total += (self.f_of_x(x) / self.g_of_x(x, a, lamda)) ** 2
        sum_of_sqs = running_total / num_samples
        # get squared average
        running_total = 0
        for i in range(num_samples):
            x = self.get_rand_number(0, int_max)
            running_total += self.f_of_x(x) / self.g_of_x(x, a, lamda)
        sq_ave = (running_total / num_samples) ** 2
        return sum_of_sqs - sq_ave

    def set_variences(self):
        """get variance as a function of lambda by dev many different lambdas"""
        for i, lamda in enumerate(self._test_lamdas):
            variance = self.get_variance(lamda, 10000)
            self._variances.append(variance)

    def get_optimal_lamda(self):
        self.set_variences()
        return self._test_lamdas[np.argmin(np.asarray(self._variances))]

    def importance_sampling_mc(self, lamda, num_samples):
        a = lamda
        running_total = 0
        for i in range(num_samples):
            r = self.get_rand_number(0, 1)
            running_total += self.f_of_x(self.inverse_g_of_r(r, lamda)) / self.g_of_x(self.inverse_g_of_r(r, lamda), a, lamda)
        approximation = float(running_total / num_samples)
        return approximation

    def run_simulation(self, optimal_lamda, num_samples):
        approx = self.importance_sampling_mc(optimal_lamda, num_samples)
        variance = self.get_variance(optimal_lamda, num_samples)
        error = (variance / num_samples) ** 0.5
        # display results
        print(f"Importance Sampling Approximation: {approx}")
        print(f"Variance: {variance}")
        print(f"Error: {error}")


class MonteCarloPlotterTraining:

    def __init__(self, monte_carlo):
        self._monte_carlow = monte_carlo

    def plot_fx(self):
        xs = [float(i/50) for i in range(int(50*math.pi*2))]
        ys = [self._monte_carlow.f_of_x(x) for x in xs]
        plt.plot(xs, ys, 'b')
        plt.title("f(x)")
        plt.show()

    def plot_fx_gx(self):
        xs = [float(i / 50) for i in range(int(50 * math.pi))]
        fs = [self._monte_carlow.f_of_x(x) for x in xs]
        gs = [self._monte_carlow.g_of_x(x, a=1.4, lamda=1.4) for x in xs]
        plt.plot(xs, fs, 'r')
        plt.plot(xs, gs, 'g')
        plt.title("f(x) and g(x)")
        plt.show()

    def plot_varaince_vs_lamda(self):
        lamdas = self._monte_carlow.get_test_lamdas()
        variances = self._monte_carlow.get_variances()
        plt.plot(lamdas[5:40], variances[5:40])
        plt.title("Variance of MC at Different Lambda Values")
        plt.show()

mc = MonteCarloTraining()
print("Optimal Lamda = " + str(mc.get_optimal_lamda()))
mc.run_simulation(mc.get_optimal_lamda(), 10000)
mcp = MonteCarloPlotterTraining(mc)
mcp.plot_fx()
mcp.plot_fx_gx()
mcp.plot_varaince_vs_lamda()
