
import os
import time
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


def generateILP(
        ILP_name: str = None,
        ILP_type: str = None,
        ILP_path: str = None,
        fraction: float = 0.2,
        num_instance: int = 50,
        random_seed: int = 10):
    """
    Generate children ILP instances
    """
    if not (ILP_name and ILP_type and ILP_path):
        raise ValueError("No parent ILP instance specified!")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load .mps file
    ILP = gp.read(ILP_path) # parent_ILP_path is defined in main func

    # Max iteration for generating children ILPs
    # max_iterations = num_instance ** 2 if num_instance <= 100 else num_instance * 10
    max_iterations = num_instance * 10
    
     # Set number of assigned variables
    num_assigned_vars = int(fraction * ILP.NumVars)

    # Where results are stored
    assigned_vars_index = []
    sub_opt_sol = []
    sub_opt_obj = []
    # Index of vars in constrs (CANNOT USE [[]] * ILP.NumConstrs) * num_instance
    vars_in_constrs = [[[] for _ in range(ILP.NumConstrs)] for _ in range(num_instance)]   
    # RHS of constrs (CANNOT USE [[1] * ILP.NumCosntrs] * num_instance)
    RHS = [[1] * ILP.NumConstrs for _ in range(num_instance)]   

    # Run Gurobi solver
    i = 0
    iterations = 0
    while i < num_instance and iterations < max_iterations:
        # Increase the iteration
        iterations += 1

        # Copy the original problem
        child_ILP = ILP.copy()

        # Solution 
        solution = []

        # var: var_idx map
        var_idx_map = {var: idx for idx, var in enumerate(child_ILP.getVars())}

        # Randomly select the variables to be assigned (no repetition)
        var_range = range(child_ILP.NumVars)
        assigned_vars = random.sample(var_range, num_assigned_vars)

        # Assign selected variables
        match ILP_type:
            case "SC":
                for var in assigned_vars:
                    child_ILP.getVars()[var].LB = 1
            case "SP":
                for var in assigned_vars:
                    child_ILP.getVars()[var].UB = 0
            case "MIS":
                for var in assigned_vars:
                    child_ILP.getVars()[var].UB = 0
            case "MVC":
                for var in assigned_vars:
                    child_ILP.getVars()[var].LB = 1
            case "CA":
                for var in assigned_vars:
                    child_ILP.getVars()[var].UB = 0

        # Update problem
        child_ILP.update()

        # Run Gurobi
        child_ILP.setParam("OutputFlag", 0)   # No output during solving process
        child_ILP.Params.TimeLimit = 60      # Time limit for solving ILP
        child_ILP.optimize()

        # Continue to next child ILP if no optimal solution is found
        # The counter will not be increased
        if child_ILP.Status != GRB.OPTIMAL:
            print(f"Failed to generate child ILP for {ILP_name} at fraction {fraction} "
                  f"in the {iterations}-the iteration.")
            continue

        # Get the solution
        for j in range(child_ILP.NumVars):
            solution.append(child_ILP.getVars()[j].X)
        assigned_vars_index.append(assigned_vars[:])
        sub_opt_sol.append(solution[:])
        sub_opt_obj.append(child_ILP.ObjVal)

        # Get var index in each constr
        for constr_idx, constr in enumerate(child_ILP.getConstrs()):
            coeffs = child_ILP.getRow(constr)
            for idx in range(coeffs.size()):
                var = coeffs.getVar(idx)
                var_idx = var_idx_map[var]
                if var_idx not in assigned_vars:
                    vars_in_constrs[i][constr_idx].append(var_idx)
                else:
                    RHS[i][constr_idx] -= 1.
        
        print(f"Generated {i+1}-th child ILP for {ILP_name} fraction={fraction}")

        # Increase the counter
        i += 1
    
    return assigned_vars_index, sub_opt_sol, sub_opt_obj, vars_in_constrs, RHS




def generateChildrenILPs(
        parent_ILPs_dir: str = None,    # Directory of parent ILPs
        parent_ILPs: list = None,       # Names of parent ILPs
        fractions: list = [0.1, 0.15, 0.2, 0.25],   # Fraction of assigned vars
        num_instances: list = [2, 2, 2, 2],         # Number of children ILPs
        random_seed: int = 10):    
    """
    Generate children ILPs for the input parent ILPs following required fractions and numbers
    """
    if not (parent_ILPs_dir or parent_ILPs):
        raise ValueError("Parent ILPs not specified.")
    
    # Generate chileren ILPs
    for parent_ILP in parent_ILPs:
        # Get parent ILP type
        parent_ILP_type = parent_ILP[:3].replace("_", "")

        # Get dir and names of parent ILPs
        parent_ILP_path = os.path.realpath(
            os.path.join(parent_ILPs_dir, f"./{parent_ILP}")
        )
        
        # Generate all children ILPs
        for fraction, num_instance in zip(fractions, num_instances):
            # Generate children ILP given parent ILP, num_instance and fraction
            assigned_vars_index, sub_opt_sol, sub_opt_obj, \
            vars_in_constrs, RHS = generateILP(ILP_name = parent_ILP,
                                        ILP_type = parent_ILP_type, 
                                        ILP_path = parent_ILP_path,
                                        fraction = fraction, 
                                        num_instance = num_instance, 
                                        random_seed = random_seed)

            # In case no children ILP generated
            if len(sub_opt_obj) == 0:
                print(f"\nNo children ILP generated for {parent_ILP}\n")
                continue
            
            # Sufficient/insufficient children ILPs 
            if len(sub_opt_obj) < num_instance:
                print(f"\nNot sufficient children ILPs generated for {parent_ILP}"
                      f" at fraction={fraction}\n")
            else:
                print(f"\nSuccessfully generated all children ILPs for {parent_ILP}"
                      f" at fraction={fraction}\n")
                
            # Directory where the results will be saved
            children_ILP_dir = (Path(parent_ILPs_dir) / f"./../children/{parent_ILP[:-4]}_children_ILP_{fraction}").resolve()
            children_ILP_dir.mkdir(parents=True, exist_ok=True)

            # Save results, use pathlib to deal with the "\" and "/" difference 
            # on Windows and Unix-like system
            file_path = Path(children_ILP_dir) / "assigned_vars_index.txt"
            with open(file_path, "w") as f:
                for line in assigned_vars_index:
                    f.write(",".join(map(str, line)) + "\n")

            file_path = Path(children_ILP_dir) / "sub_opt_sol.txt"
            with open(file_path, "w") as f:
                for line in sub_opt_sol:
                    f.write(",".join(map(str, line)) + "\n")

            file_path = Path(children_ILP_dir) / "sub_opt_obj.txt"
            with open(file_path, "w") as f:
                for value in sub_opt_obj:
                    f.write(f"{value}\n")

            file_path = Path(children_ILP_dir) / "RHS.txt"
            with open(file_path, "w") as f:
                for line in RHS:
                    f.write(",".join(map(str, line)) + "\n")

            file_path = Path(children_ILP_dir) / "vars_in_constrs.txt"
            with open(file_path, "w") as f:
                for block in vars_in_constrs:
                    for line in block:
                        f.write(",".join(map(str, line)) + "\n")
                    f.write("---End of instance---\n")
                    


# Get inputs from command line
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions",
                        type = list,
                        default = [0.25, 0.22, 0.20, 0.18, 0.15],
                        help = "Fraction of assigned variables of each parent ILP")
    parser.add_argument("--num_instances",
                        type = list,
                        default = [50] * 5,
                        help = "Number of children ILP instances at each fraction")
    parser.add_argument("--random_seed",
                        type = int,
                        default = 10,
                        help = "Random seed for reproducibility")
    return parser.parse_args()
    






if __name__ == "__main__":
    # Terminal inputs
    args = parseArgs()
    print(f"\n\nYour input is:\n {vars(args)}\n\n")

    # Read parent ILPs
    parent_ILPs_dir = r"./../instances/training/parents/"
    parent_ILPs = os.listdir(parent_ILPs_dir)
    # The order of items in parentILPs can be different 
    # if running the script on different operating systems
    parent_ILPs.sort()

    # Generate children ILPs
    generateChildrenILPs(parent_ILPs_dir, parent_ILPs, **vars(args))