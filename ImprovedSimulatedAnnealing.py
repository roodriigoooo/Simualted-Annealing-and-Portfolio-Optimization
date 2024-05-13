#!/usr/bin/env python
# coding: utf-8


import numpy as np
import random



def load_data():
    file = 'https://raw.githubusercontent.com/jnin/information-systems/main/data/SP500_data.csv'
    all_data = np.genfromtxt(file, delimiter=",")
    data = all_data[1:, 1:]
    return data
    # Implement a function to load all data except the first row and first column in your code.
    # We strongly encourage you to store your data in a NumPy array.

    # return the data


def aggregate_data(data):
    # Aggregate the stock value of each company in blocks of 5 days (approx. a week).
    # Note that The stock markets close during weekends.

    # The aggregation consists of computing the LogRatio (either a gain or a loss)
    # of the stock value of each company between two days, with a time lag of 5 days.
    # For instance, the first entry of the aggregate data is the LogRatio of the stock
    # market price of each company between October 13, 2006, and October 6, 2006.
    # The second entry will be the LogRatio between October 20 and October 13, and so on.

    # return the aggregated data called weekly_data
    aggregated_data = []
    for i in range(5, len(data), 5):
        log_ratio = np.log(data[i] / data[i - 5])
        aggregated_data.append(log_ratio)
    weekly_data = np.array(aggregated_data)
    return weekly_data


def calculate_mean_std(data):
    # Compute the aggregated data mean and the standard deviation for each company
    # Store them in a matrix called mean_std containing column-wise.
    # Column 0 should contain all companies' means,
    # and column 1 its corresponding standard deviation
    mean = np.mean(data, axis=0)
    deviation = np.std(data, axis=0)
    mean_std = np.vstack((mean, deviation)).T
    # Return this matrix mean_std
    return mean_std


def initial_solution(size, amount):
    # Compute the initial solution
    # Giving the 100 units of money, generate the initial asset allocation
    # assigning 1 unit to random companies

    # return the initial solution
    solution = np.zeros(size)
    solution[:amount] = 1
    np.random.shuffle(solution)
    return solution

def initial_solution_optimized(size, amount, mean_std):
    mean = mean_std[:,0]
    attractive_companies = np.argsort(mean)[-amount:]
    solution = np.zeros(size)
    solution[attractive_companies] = 1
    return solution

def initial_solution_optimized2():
    pass

def objective_function(distributions, sol):
    num_trials = 100
    total_gains = np.zeros(num_trials)


    # Simulate total gains over 100 trials
    for i in range(num_trials):
        random_values = np.random.normal(distributions[:, 0], distributions[:, 1]) + 1
        total_gain = np.sum(random_values * sol)
        total_gains[i] = total_gain

        # Compute the 95% one-tailed Value at Risk (VaR) as the 5th percentile of total gains
    VaR = np.percentile(total_gains, 5)

    return VaR
    # Giving the mean and standard deviation of each company computed over the aggregated data
    # and the current solution sol, compute:
    # 1. For each company, draw a random value following a normal distribution with its mean and standard deviation.
    # 2. Use these random values and the current assets allocated to each company to compute the total gain
    # 3. Repeat 100 times points 1 and 2.
    # Compute the 95% one-tailed Value at Risk over the 100 total gains


def objective_function_sharp(distributions, sol):
    """

    :param normaliser:
    :param sol:
    :return:
    """
    # Giving the mean and standard deviation of each company computed over the aggregated data
    # and the current solution sol, compute:
    # 1. For each company, draw a random value following a normal distribution with its mean and standard deviation.
    # 2. Use these random values and the current assets allocated to each company to compute the total gain
    # 3. Repeat 100 times points 1 and 2.
    # Compute the (adapted) Sharp Ratio over the 100 total gains
    # 1. Compute the mean of the gains
    # 2. Compute the standard deviation of the gains
    # 3. Apply the adapted formula for the Sharp Ratio

    # Return the sharp ratio
    num_trials = 100
    total_gains = np.zeros(num_trials)
    init_investment = np.sum(sol)

    for i in range(num_trials):
        random_values = np.random.normal(distributions[:, 0], distributions[:, 1]) + 1
        total_gain = np.sum(random_values * sol)
        total_gains[i] = total_gain

    gains_mean = np.mean(total_gains)
    gains_std = np.std(total_gains - init_investment)

    sharp = (gains_mean - init_investment) / gains_std if gains_std != 0 else 0
    return sharp


def objective_function_mdd(distributions, sol):
    # Giving the mean and standard deviation of each company computed over the aggregated data
    # and the current solution sol, compute:
    # 1. For each company, draw a random value following a normal distribution with its mean and standard deviation.
    # 2. Use these random values and the current assets allocated to each company to compute the total gain
    # 3. Repeat 100 times points 1 and 2.
    # Compute the (adapted) maximum drawdow (mdd) over the 100 total gains
    # 1. Find the minimum gain
    # 2. Find the maximum gain
    # 3. Apply the adapted formula for mdd

    # Return the mdd
    num_trials = 100
    total_gains = np.zeros(num_trials)

    for i in range(num_trials):
        random_values = np.random.normal(distributions[:, 0], distributions[:, 1]) + 1
        total_gain = np.sum(random_values * sol)
        total_gains[i] = total_gain

    max_gain = np.max(total_gains)
    min_gain = np.min(total_gains)

    mdd = (min_gain - max_gain) / max_gain if max_gain > 0 else 0
    return mdd

def get_neighbor(sol):
    # generate a new neighbor solution
    # move one unit from one of the companies in the asset to another whatever company
    # the company receiving the unit has to be different than the company moving the unit
    # but it could be either a company already having some units or a company with 0 units

    # return the neighbor solution
    companies_with_investments = np.where(sol > 0)[0]
    company_moving = np.random.choice(companies_with_investments)
    company_pool = np.arange(len(sol))[np.arange(len(sol)) != company_moving]
    company_receiving = np.random.choice(company_pool)

    neig = sol.copy()
    neig[company_moving] -= 1
    neig[company_receiving] += 1

    return neig

def get_neighbor_swap(sol):
    companies_with_investments = np.where(sol > 0)[0]
    company_moving, company_receiving = np.random.choice(companies_with_investments, size = 2, replace = False)
    neig = sol.copy()
    neig[company_moving], neig[company_receiving] = neig[company_receiving], neig[company_moving]
    return neig

def get_neighbor_stochastic(sol, distributions, current_eval, num_neighbors = 5):
    neighbors = []
    for neig in range(num_neighbors):
        neighbor = get_neighbor(sol)
        eval = objective_function(distributions, neighbor)
        if eval > current_eval:
            neighbors.append(neighbor)
    if neighbors:
        return random.choice(neighbors)
    else:
        return sol



def evaluate_sol(distributions, neig_sol, current_eval, best_eval, temperature):
    neig_eval = objective_function(distributions, neig_sol)
    diff = neig_eval - current_eval
    if diff > 0:
        # If the neighbor solution is better than the current solution, accept it.
        accept_neig = True
        current_eval = neig_eval
    else:
        # If the neighbor solution is worse, accept it with a probability according to the Metropolis criterion.
        p = np.random.rand()
        if p < np.exp(-diff / temperature):
            accept_neig = True
            current_eval = neig_eval
        else:
            accept_neig = False
    return accept_neig, current_eval


def simulated_annealing_optimize(data, distributions, sol, num_iter, temperature):
    init_eval = objective_function(distributions, sol)
    best = sol.copy()
    best_eval = init_eval
    current_sol = sol.copy()
    current_eval = init_eval

    for i in range(num_iter):
        temperature = temperature * 0.98

        if i % 10 == 0:
            neighbor_sol = get_neighbor_swap(current_sol)
        elif temperature < 50:
            neighbor_sol = get_neighbor_stochastic(current_sol, distributions, current_eval)
        else:
            neighbor_sol = get_neighbor(current_sol)

        accept_neig, updated_eval = evaluate_sol(distributions, neighbor_sol, current_eval, best_eval, temperature)

        if accept_neig:
            current_sol = neighbor_sol
            current_eval = updated_eval
            # Update best solution if the current solution is better than the best solution found so far.
            if current_eval > best_eval:
                best = current_sol.copy()
                best_eval = current_eval

    return best, best_eval


# Constants
initial_amount = 100
max_iters = 100  # Number of generated random values for the calculation
# of the 95% VaR in objective function
temperature = 1000

repetitions = 100  # Main loop of the simulated annealing

np.random.seed(0)  # For reproducibility, i.e.,
# different executions of the same code
# will generate the same random numbers
# Keep it to check if the code works
# Comment it to check it the code is able
# to optimise any combination of randon numbers

# MAIN
data = load_data()
print("Data shape ", data.shape)

# MAIN
data = load_data()
print("Data shape ", data.shape)

data_week = aggregate_data(data)
print("Weekly data shape ", data_week.shape)

data_mean_std = calculate_mean_std(data_week)

isolution = initial_solution_optimized(data_week.shape[1], initial_amount, data_mean_std)
print("Selected companies: ", isolution)

VaR_95 = objective_function(data_mean_std, isolution)

print(f"\nValue at Risk (VaR) at 95% confidence level: ${VaR_95:,.2f}")

best, best_eval = simulated_annealing_optimize(data_week, data_mean_std, isolution, repetitions, temperature)

print("Selected companies: ", best)

print(f"\nValue at Risk (VaR) at 95% confidence level: ${best_eval:,.2f}")