import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def expected_payoff(all_lambdas, n_teams, n_sites, team_sizes):
    alpha, beta, gamma, delta = 1.451, 0.138, -0.287, 0.057
    #Same Shape as SiteTeamMatrix
    expected_rewards = np.empty((n_teams, n_sites)) * 0.0

    k = 5

    for i in range(n_teams):
        team_size = team_sizes[i]
        for j in range(n_sites):
            # FIX THIS: 
            # Furthermore, STsji denotes the estimated starting time (in hours) of team i at site sj, which Van Rijn et al. (2020) estimate as 7 AM plus the driving time
            expected_rewards[i][j] = all_lambdas[j] * (alpha + beta * (np.sqrt(12 * k) + np.sqrt(12 * (k - 1))) + gamma * np.sqrt(12) + delta * team_size)
    
    return expected_rewards



def inititial_sol(SiteTeam: pd.DataFrame, n_teams: int, n_sites: int) -> dict:
    initial_solution = {}
    shape_params = SiteTeam.shape  
    
    for i in range(1, shape_params[0] + 1):
        for j in range(1, shape_params[1] + 1):
            initial_solution[i, j] = SiteTeam.iloc[i-1, j-1]
    initial_solution1 = {(j, i): initial_solution[i, j] for i in range(1, n_sites + 1) for j in range(1, n_teams + 1)}

    return initial_solution1


def concat_df(df, k):
    df_repeat = pd.concat([df] * k, ignore_index=True)
    return df_repeat



def calculate_incentive(z, z_ij, num_teams, num_sites, c, a, b):
    min_z = min(z_ij.values())
    incentive_z = (z-min_z)/z
    
    incentive = {}
    
    for i in range(1, num_teams):
        for j in range(1, num_sites):
            incentive[i, j] = c[i]/(a[i, j] + b[i]) * (incentive_z - (z - z[i, j])/z)
            
    return incentive