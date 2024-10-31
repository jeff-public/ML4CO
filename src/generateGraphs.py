import os
import random
import argparse
import glob
from pathlib import Path
import shutil
import gurobipy as gp
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from GraphDataset import GraphDataset

def parentILP2Graph(
        ILP_name: list = None,
        ILP_dir: list = None,
        sub_opt_sol_list: list = None,
        graph_list: list = None,
        random_seed: int = 10
):
    if not (ILP_name and ILP_dir):
        raise ValueError("\nNo input ILPs\n")
    
    # Set random seed
    random.seed(random_seed)

    # Load ILP problem
    ILP_path = os.path.realpath(os.path.join(ILP_dir, f"./{ILP_name}"))
    ILP = gp.read(ILP_path)

    # Variable nodes, feature vector = [c, p(x), random feature]
    var_nodes = [[1, 0.5, random.uniform(0, 1)] for _ in range(ILP.NumVars)]

    # Constrain nodes,
    # feature vector = [RHS, "=", ">", "<", ">=", "<=", random feature]
    constr_nodes = [[constr.RHS, 0, 0, 0, 1, 0, random.uniform(0, 1)] for constr in ILP.getConstrs()]

    # Edges
    edges = []
    constr_idx_map = {constr: idx for idx, constr in enumerate(ILP.getConstrs())}
    for node_idx, node in enumerate(ILP.getVars()):
        coeffs = ILP.getCol(node)
        for idx in range(coeffs.size()):
            constr = coeffs.getConstr(idx)
            constr_idx = constr_idx_map[constr]
            edges.append([node_idx, constr_idx])

    # Edge features: TBD
    edge_attr = [[1] for _ in range(len(edges))]

    ## Assemble the graph
    graph = HeteroData()

    # Variable nodes
    graph["var_nodes"].x = torch.tensor(var_nodes).float()
    graph["var_nodes"].mask = torch.ones(ILP.NumVars).bool()
    x_list = torch.tensor(sub_opt_sol_list)
    best_obj_found = x_list.sum(axis = 1).min()    # The best feasible solution so far

    ## Mean or sum????
    graph["var_nodes"].best_obj_found = torch.tensor(best_obj_found)
    graph["var_nodes"].dividend = torch.exp(-x_list.sum(axis = 1) / best_obj_found).mean()
    graph["var_nodes"].ILP = ILP_name

    # Constraint nodes
    graph["constr_nodes"].x = torch.tensor(constr_nodes).float()

    # Edges
    edge_index = torch.tensor(edges).long().t().contiguous()
    graph["var_nodes", "in", "constr_nodes"].edge_index = edge_index
    graph["var_nodes", "in", "constr_nodes"].edge_attr = torch.tensor(edge_attr).float()

    graph["constr_nodes", "rev_in", "var_nodes"].edge_index = edge_index.flip(dims=(0,))
    graph["constr_nodes", "rev_in", "var_nodes"].edge_attr = torch.tensor(edge_attr).float()


    ## Collect the graph
    graph_list.append(graph)

    return


def generateParentGraphDataset(
        parent_ILPs_dir: list = None,
        parent_ILPs: list = None,
        graph_list: list = None,
        random_seed: int = 10):
    
    for parent_ILP in parent_ILPs:
        # Get children dir of the parent_ILP
        children_ILPs_dirs = glob.glob(os.path.realpath(
            os.path.join(parent_ILPs_dir, f"./../children/{parent_ILP[:-4]}_*")))

        # Get sub_opt_sol_list
        sub_opt_sol_list = []
        for dir in children_ILPs_dirs:
            file_path = Path(dir) / "sub_opt_sol.txt"
            with open(file_path, "r") as f:
                for line in f:
                    sub_opt_sol_list.append([float(var) for var in line.strip().split(",")])
                
        # Generate parent ILP graph
        parentILP2Graph(
            ILP_name = parent_ILP,
            ILP_dir = parent_ILPs_dir,
            sub_opt_sol_list = sub_opt_sol_list,
            graph_list = graph_list,
            random_seed = random_seed)
    
    # Generate graph dataset
    dataset_dir = Path(f"./../dataset/training/parent_graphs")
    dataset_dir.mkdir(parents = True, exist_ok = True)

    shutil.rmtree(dataset_dir)
    print(f'The old dataset has been deleted!')
    # Save the dataset
    GraphDataset(root=dataset_dir, data_list=graph_list)
    print(f'The new dataset has been saved!')
    
    return



def childrenILP2Graph(
        assigned_vars_index: list = [],
        # vars_in_constrs: list = [],
        sub_opt_sol: list = [],
        sub_opt_obj: float = 0,
        # RHS: list = [],
        parent_mps: gp.Model = None,
        children_ILP: list = [],
        graph_list: list = [],
        random_seed = 10):
    # # In case there is empty/wrong input
    # if not (assigned_vars_index and 
    #         # vars_in_constrs and 
    #         sub_opt_sol and 
    #         sub_opt_obj and 
    #         RHS):
    #     raise ValueError("\nEmpty or wrong input.\n")

    # Set random seed
    random.seed(random_seed)
    
    # Number of graphs
    num_graphs = len(sub_opt_sol)

    # Number of variables
    num_vars = parent_mps.NumVars

    # Create individual graph
    for i in range(num_graphs):        
        # Variable nodes, feature vector = [c, p(x), random feature]
        var_nodes = []
        for var in range(num_vars):
            if var not in assigned_vars_index[i]:
                # Unassigned variable x = 0.5
                var_nodes.append([1, 0.5, random.uniform(0, 1)])
            else:
                # Assigned variable x = 1 (ONLY IN THIS CASE)
                var_nodes.append([1, 1, random.uniform(0, 1)])

        # # Constraint nodes, feature vector = [RHS, "=", ">", "<", ">=", "<="]
        # constr_nodes = []
        # for constr in range(len(vars_in_constrs[i])):
        #     # For SC problem: ">="
        #     constr_nodes.append([RHS[i][constr], 0, 0, 0, 1, 0])  
        # # Edges
        # edges = []
        # for j in vars_in_constrs[i]:
        #     edges.append([[k, j] for k in var_nodes if k in vars_in_constrs[i]])


        # Constraint nodes, feature vector = [RHS, "=", ">", "<", ">=", "<=", random feature]
        constr_nodes = [[1, 0, 0, 0, 1, 0, random.uniform(0, 1)] for _ in range(parent_mps.NumConstrs)]

        # Edges
        edges = []
        constr_idx_map = {constr: idx for idx, constr in enumerate(parent_mps.getConstrs())}
        for var_idx, var in enumerate(parent_mps.getVars()):
            coeffs = parent_mps.getCol(var)
            for idx in range(coeffs.size()):
                constr = coeffs.getConstr(idx)
                constr_idx = constr_idx_map[constr]
                edges.append([var_idx, constr_idx])
        # Edge features: TBD
        edge_attr = [[1] for _ in range(len(edges))]

        ## Assemble the graph
        graph = HeteroData()

        # Variable nodes
        graph["var_nodes"].x = torch.tensor(var_nodes).float()
        graph["var_nodes"].y = torch.tensor(sub_opt_sol[i]).float()
        graph["var_nodes"].mask = ~torch.isin(torch.arange(num_vars), 
                                              torch.tensor(assigned_vars_index[i]))
        graph["var_nodes"].ILP = children_ILP
        
        # Constraint nodes
        graph["constr_nodes"].x = torch.tensor(constr_nodes).float()

        # Edges
        edge_index = torch.tensor(edges).long().t().contiguous()
        graph["var_nodes", "in", "constr_nodes"].edge_index = edge_index
        graph["var_nodes", "in", "constr_nodes"].edge_attr = torch.tensor(edge_attr).float()

        graph["constr_nodes", "rev_in", "var_nodes"].edge_index = edge_index.flip(dims=(0,))
        graph["constr_nodes", "rev_in", "var_nodes"].edge_attr = torch.tensor(edge_attr).float()

        ## Collect the graph
        graph_list.append(graph)

    return


def generateChildrenGraphDataset(
        children_ILPs_dir: list = [],
        children_ILPs: list = [],
        parent_ILPs: list = [],
        graph_list: list = [],
        random_seed: int = 0):
    
    for ILP, parent_ILP in zip(children_ILPs, parent_ILPs):
        ILP_dir = os.path.realpath(
            os.path.join(children_ILPs_dir, f"./{ILP}"))

        # Load assigned_vars_index.txt
        assigned_vars_index = []
        file_path = Path(ILP_dir) / "assigned_vars_index.txt"
        with open(file_path, "r") as f:
            for line in f:
                assigned_vars_index.append([float(var) for var in line.strip().split(",")])

        # # Load vars_in_constrs.txt
        # vars_in_constrs = []
        # block = []
        # file_path = Path(ILP_dir) / "vars_in_constrs.txt"
        # with open(file_path, "r") as f:
        #     for line in f:
        #         if line.strip() != "---End of instance---":
        #             block.append([int(var) for var in line.strip().split(",")])
        #         else:
        #             vars_in_constrs.append(block)
        #             block = []

        # Load sub_opt_sol.txt
        sub_opt_sol = []
        file_path = Path(ILP_dir) / "sub_opt_sol.txt"
        with open(file_path, "r") as f:
            for line in f:
                sub_opt_sol.append([float(var) for var in line.strip().split(",")])

        # Load sub_opt_obj.txt
        file_path = Path(ILP_dir) / "sub_opt_obj.txt"
        with open(file_path, "r") as f:
            sub_opt_obj = [float(line.strip()) for line in f]

        # Load RHS.txt
        RHS = []
        file_path = Path(ILP_dir) / "RHS.txt"
        with open(file_path, 'r') as f:
            for line in f:
                RHS.append([float(var) for var in line.strip().split(",")])

        # Load parent ILP
        file_path = os.path.realpath(
            os.path.join("./../instances/training/parents/", f"{parent_ILP}.mps"))
        parent_mps = gp.read(str(file_path))

        # Call function
        childrenILP2Graph(
            assigned_vars_index = assigned_vars_index,
            # vars_in_constrs = vars_in_constrs,
            sub_opt_sol = sub_opt_sol,
            sub_opt_obj = sub_opt_obj,
            # RHS = RHS,
            parent_mps = parent_mps,
            children_ILP = ILP,
            graph_list = graph_list,
            random_seed = random_seed)
        
    # Generate graph dataset
    dataset_dir = Path(f"./../dataset/training/children_graphs")
    dataset_dir.mkdir(parents = True, exist_ok = True)

    shutil.rmtree(dataset_dir)
    print(f'The old dataset has been deleted!')
    # Save the dataset
    GraphDataset(root=dataset_dir, data_list=graph_list)
    print(f'The new dataset has been saved!')

    return



def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed",
                        type = int,
                        default = 10,
                        help = "Random seed")
    
    return parser.parse_args()






if __name__ == "__main__":

    # Terminal inputs
    args = parseArgs()
    print(f"\n\nYour input is:\n {vars(args)}\n\n")
    
    ## Parent ILP to graph
    parent_ILPs_dir = f"./../instances/training/parents/"
    parent_ILPs = os.listdir(parent_ILPs_dir)
    parent_ILPs = [var for var in parent_ILPs if var[-4:] == ".mps"]
    parent_ILPs.sort()
    graph_list = []
    generateParentGraphDataset(
        parent_ILPs_dir = parent_ILPs_dir,
        parent_ILPs = parent_ILPs,
        graph_list = graph_list,
        **vars(args))
    

    ## Children ILP to graph
    children_ILPs_dir = f"./../instances/training/children/"
    children_ILPs = os.listdir(children_ILPs_dir)
    mps_types = ["SC", "SP", "MIS", "MVC", "CA"]
    children_ILPs = [var for var in children_ILPs 
                     if var[:3].replace("_", "") in mps_types]
    parent_ILPs = [item.split("_", 2)[0] + "_" +item.split("_", 2)[1] 
                   for item in children_ILPs]
    # Sort the list so that they are in the same order 
    # for differene toperating systems
    children_ILPs.sort()
    graph_list = []
    generateChildrenGraphDataset(
        children_ILPs_dir = children_ILPs_dir,
        children_ILPs = children_ILPs,
        parent_ILPs = parent_ILPs,
        graph_list = graph_list,
        **vars(args))