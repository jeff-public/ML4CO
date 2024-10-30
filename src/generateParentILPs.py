import os
import time
import random
import argparse
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path

"""
This script generates five types benchmark integer linear programming (ILP) instances:
set covering (SC), set partitioning (SP), maximum independent set (MIS), minimum vertices
cover (MVC) and combinatorial auction (CA)
"""



def generateSCInstances(
        num_vars: int = 10,
        num_instances: int = 5,
        random_seed: int = 10):
    """
    Generate N set covering instances with num_vars variables each.
    The number of constraints is randomly set between [0.8*num_vars, 1.5*num_vars].
    The number of variables in each constraint is set randomly, ensuring that every 
    variable appears in at least one constraint.
    Save the instances in .mps format using Gurobi.
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    instance_num = 0
    while instance_num < num_instances:
        try:
            # Create a new model
            model = gp.Model(f"SC_{instance_num + 1}")

            # Set num_vars 
            num_vars = random.randint(int(0.5 * num_vars), int(1.5 * num_vars))

            # Add binary variables for the set covering problem
            variables = model.addVars(num_vars, vtype=GRB.BINARY, name="x")

            # Determine the number of constraints randomly
            num_constraints = random.randint(int(0.8 * num_vars), int(1.5 * num_vars))

            constraints = []

            # 2-10 variables appear in each constraint
            for _ in range(num_constraints):
                num_vars_in_constraint = min(num_vars, random.randint(5, 15))
                selected_vars = random.sample(range(num_vars), num_vars_in_constraint)
                constraints.append(selected_vars)

            # Add constraints to the model
            for j, selected_vars in enumerate(constraints):
                model.addConstr(
                    gp.quicksum(variables[i] for i in selected_vars) >= 1, 
                    name=f"Constraint_{j+1}")

            # Set objective to minimize the sum of variables
            model.setObjective(
                gp.quicksum(variables[i] for i in range(num_vars)), 
                GRB.MINIMIZE)


        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e.message}")
        except AttributeError as e:
            print(f"Attribute error: {e}")


        # Check whether the instance is feasible
        model.Params.OutputFlag = 0 # No log during the solving process
        model.Params.TimeLimit = 5
        model.optimize()

        # Check model.Status
        OPTIMAL = True if model.Status == GRB.OPTIMAL else False
        SUBOPTIMAL = True if model.Status == GRB.SUBOPTIMAL else False
        TIMELIMIT = True if model.Status == GRB.TIME_LIMIT else False
        if OPTIMAL or SUBOPTIMAL or TIMELIMIT:
            # Save the model in MPS format
            file_path = Path(f"../instances/training/parents/SC_{instance_num + 1}.mps")
            # In case there is directory missing
            file_path.parent.mkdir(parents = True, exist_ok = True)
            # Input to Gurobi.model.write() should be a str
            model.write(str(file_path))
            print(f"Instance {instance_num + 1} generated and saved as "
                  f"SC_{instance_num + 1}.mps")
        
            instance_num += 1


def parseArgs():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ILP_type",
    #                     type = str,
    #                     choices = ["SC", "SP", "MIS", "MVC", "CA"]
    #                     default = "SC",
    #                     help = "ILP type")
    parser.add_argument("--num_vars",
                        type = int,
                        default = 500,
                        help = "Number of variables in each ILP instance. \
                            The actual number of variables is between [0.5, 1.5]*input.")
    parser.add_argument("--num_instances",
                        type = int,
                        default = 10,
                        help = "The number of ILP instances")
    parser.add_argument("--random_seed",
                        type = int,
                        default = 10,
                        help = "Random seed")
    return parser.parse_args()



if __name__ == "__main__":
    args = parseArgs()
    print(vars(args))
    generateSCInstances(**vars(args))