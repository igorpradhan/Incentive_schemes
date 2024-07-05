import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB



def method_optimal(payoffs: dict, max_tasks: dict, initial_solution: dict, num_teams = int, num_sites = int):
    with gp.Env() as env, gp.Model(env=env) as model:
        m = gp.Model("IC")

        
        
        N =  [i for i in range(1, num_teams + 1)]
        J = [j for j in range(1, num_sites + 1)]
        
        x = {}
        a = payoffs
        
        
        for i in N:
            for j in J:
                    x[i, j] = m.addVar(vtype=GRB.BINARY, name="x_%s_%s" % (i, j))
        
                    
        objective = gp.quicksum(a[i, j] * x[i, j] for i in N for j in J)
        m.setObjective(objective, GRB.MAXIMIZE)
        

        
        for j in J:
            m.addConstr(gp.quicksum(x[i, j] for i in N) <= 1, name=f"c_{j}")
            
        for i in N:
            m.addConstr(gp.quicksum(x[i, j] for j in J) <= max_tasks[i-1], name=f"c_{i}")
            m.addConstr(x[i, j] * a[i,j] >= initial_solution[i, j] * a[i,j], name=f"c_{i}_{j}")    
            
        
        m.optimize()

        x_values = None
        if m.status == GRB.OPTIMAL:
            m.getAttr('x', x)   
            print('Obj: %g' % m.objVal)
            x_values = {(i,j): x[i,j].x for i in N for j in J}

        return m.objVal, x_values
    
    
# FINDING z*
    
def method_z(payoffs: dict, max_tasks: dict, num_teams: int, num_sites: int):
    with gp.Env() as env, gp.Model(env=env) as model:
        m = gp.Model("IC")

        
        
        N =  [i for i in range(1, num_teams + 1)]
        J = [j for j in range(1, num_sites + 1)]
        
        x = {}
        a = payoffs
        
        
        for i in N:
            for j in J:
                    x[i, j] = m.addVar(vtype=GRB.BINARY, name="x_%s_%s" % (i, j))
        
                    
        objective = gp.quicksum(a[i, j] * x[i, j] for i in N for j in J)
        m.setObjective(objective, GRB.MAXIMIZE)
        

        
        for j in J:
            m.addConstr(gp.quicksum(x[i, j] for i in N) <= 1, name=f"c_{j}")
            
        for i in N:
            m.addConstr(gp.quicksum(x[i, j] for j in J) <= max_tasks[i-1], name=f"c_{i}") 
        
        m.addConstr(x[i, j] == 1, name=f"c_x_{i}_x_{j}")

        
        m.optimize()

        x_values = None
        if m.status == GRB.OPTIMAL:
            m.getAttr('x', x)   
            print('Obj: %g' % m.objVal)
            x_values = {(i,j): x[i,j].x for i in N for j in J}

        return m.objVal, x_values
    
# FINDING z_{ij}*
    
def method_contrained(payoffs: dict, max_tasks: dict, num_teams: int, num_sites: int, extra_constraint: tuple, x_opt: dict):
    with gp.Env() as env, gp.Model(env=env) as model:
        m = gp.Model("IC")

        
        
        N =  [i for i in range(1, num_teams + 1)]
        J = [j for j in range(1, num_sites + 1)]
        
        x = x_opt
        a = payoffs
        
        
        for i in N:
            for j in J:
                    x[i, j] = m.addVar(vtype=GRB.BINARY, name="x_%s_%s" % (i, j))
        
                    
        objective = gp.quicksum(a[i, j] * x[i, j] for i in N for j in J)
        m.setObjective(objective, GRB.MAXIMIZE)
        

        
        for j in J:
            m.addConstr(gp.quicksum(x[i, j] for i in N) <= 1, name=f"c_{j}")
            
        for i in N:
            m.addConstr(gp.quicksum(x[i, j] for j in J) <= max_tasks[i-1], name=f"c_{i}") 
        
        #Team i fixed to site j
        i, j = extra_constraint
        
        m.addConstr(x[i, j] == 1, name=f"c_x_{i}_x_{j}")

        
        m.optimize()

                
        if m.status == GRB.OPTIMAL:
            m.getAttr('x', x)   
            print('Obj: %g' % m.objVal)
            

        return m.objVal
    
    
    
def method_dual(max_tasks, num_sites, num_teams, payoff):
    with gp.Env() as env, gp.Model(env=env) as model:
        m = gp.Model("IC")
        
        J = [j for j in range(1, num_sites + 1)]
        N = [i for i in range(1, num_teams + 1)]
        
        v = {}
        w = {}
        
        
        for i in N:
            v[i] = m.addVar(vtype = GRB.CONTINUOUS)

        for j in J:
            w[j] = m.addVar(vtype = GRB.CONTINUOUS)
            
        m.setObjective(gp.quicksum(max_tasks[i - 1] * v[i] for i in N) + gp.quicksum(w[j] for j in J), GRB.MINIMIZE)
        
        
        for i in N:
            for j in J:
                m.addConstr(v[i] + w[j] >= payoff[i, j], name=f"c_{i}_{j}")
                
                
        for i in N:
            m.addConstr(v[i] >= 0, name=f"c_{i}")
        for j in J:
            m.addConstr(w[j] >= 0, name=f"c_{j}")
        
        
        m.optimize()
        
        
        if m.status == GRB.OPTIMAL:
            print('Obj: %g' % m.objVal)
            v_values = {i: v[i].x for i in N}
            w_values = {j: w[j].x for j in J}


        return m.objVal, v_values, w_values
    
    
def method_incentive(payoffs: dict, max_tasks: dict, initial_solution: dict, num_teams: int, num_sites: int, incentive_scheme):
    with gp.Env() as env, gp.Model(env=env) as model:
        m = gp.Model("IC")

        
        
        N =  [i for i in range(1, num_teams + 1)]
        J = [j for j in range(1, num_sites + 1)]
        
        x = {}
        a = payoffs
        p = incentive_scheme
        
        for i in N:
            for j in J:
                    x[i, j] = m.addVar(vtype=GRB.BINARY, name="x_%s_%s" % (i, j))
        
                    
        objective = gp.quicksum(a[i, j] * x[i, j] * p[i, j] for i in N for j in J)
        m.setObjective(objective, GRB.MAXIMIZE)
        

        
        for j in J:
            m.addConstr(gp.quicksum(x[i, j] for i in N) <= 1, name=f"c_{j}")
            
        for i in N:
            m.addConstr(gp.quicksum(x[i, j] for j in J) <= max_tasks[i-1], name=f"c_{i}")
            m.addConstr(x[i, j] * a[i,j] * p[i, j]>= initial_solution[i, j] * a[i,j] * p[i, j], name=f"c_{i}_{j}")    
            

        
        m.optimize()

                
        if m.status == GRB.OPTIMAL:
            m.getAttr('x', x)   
            print('Obj: %g' % m.objVal)
            

        return m.objVal