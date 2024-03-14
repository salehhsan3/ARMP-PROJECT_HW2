import numpy as np
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import Building_Blocks
from visualizer import Visualize_UR
from itertools import product
import os
import matplotlib.pyplot as plt
import time

def main():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)
    
    # Create directory for plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    
    step_sizes = [0.1, 0.3, 0.5, 0.8, 1.2, 1.7, 2.0]
    p_biases = [0.05, 0.2]
    num_runs = 5
    max_iterations = 2000
    all_costs = np.zeros((len(step_sizes), len(p_biases)))
    all_times = np.zeros((len(step_sizes), len(p_biases)))
    fig_ind = 0
    
    # for (stepsize, bias) in product(step_sizes, p_biases):
    
    for i, bias in enumerate(p_biases):
        direc_name = os.path.join('plots', f'bias={bias}')
        if not os.path.exists(direc_name):
            os.makedirs(direc_name)
        for j, stepsize in enumerate(step_sizes):
            costs = []
            times = []
            direc_name = os.path.join(os.path.join('plots', f'bias={bias}'), f'stepsize={stepsize}')
            for ind in range(num_runs):
                if not os.path.exists(direc_name):
                    os.makedirs(direc_name)
                if not os.path.exists(os.path.join(direc_name,'paths')):
                        os.makedirs(os.path.join(direc_name,'paths'))

                bb = Building_Blocks(transform=transform, ur_params=ur_params, env=env, resolution=0.1, p_bias=bias,)
                rrt_star_planner = RRT_STAR(max_step_size=stepsize, max_itr=max_iterations, bb=bb)
                
                # visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)
                
                # --------- configurations-------------
                env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
                env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
                # ---------------------------------------
                filename = f'task2_run_{ind}'
                start_time = time.time()
                path = rrt_star_planner.find_path(start_conf=env2_start, goal_conf=env2_goal, filename=filename)
                computation_time = time.time() - start_time
                print("finished", path)    
                
                if len(path) > 0:
                    cost = rrt_star_planner.compute_cost(path)
                    np.save(os.path.join(os.path.join(direc_name,'paths'),filename), path)
                    costs.append(cost)
                    times.append(computation_time)
                else:
                    # print("No path found for ", stepsize, bias, "run: ", ind)
                    costs.append(0) # found no path / empty path
                    times.append(computation_time)
                
            avg_cost = np.mean(costs)
            avg_time = np.mean(times)
            # all_costs = np.append(all_costs, avg_cost)
            # all_times = np.append(all_times, avg_time)
            all_costs[j, i] = avg_cost
            all_times[j, i] = avg_time
            print("Average cost for stepsize = ", stepsize, ",bias = ", bias, ":", avg_cost)
            print("Average time forstepsize = ", stepsize, ",bias = ", bias, ":", avg_time)
            
            # Plotting the Cost vs Time plot for these recents runs for given bias, stepsize
            plt.figure(fig_ind)
            fig_ind += 1
            plt.plot(times, costs)
            plt.xlabel('Computation Time')
            plt.ylabel('Cost')
            plt.title(f'Cost vs Computation Time (Max Step Size: {stepsize}, p_bias: {bias})')
            
            # Save plot
            plt.savefig(os.path.join(direc_name, f'plot_step_{stepsize}_bias_{bias}.png'))
            plt.close()
            
    
        # Plotting all runs
        plt.figure(fig_ind)
        fig_ind += 1
        for k, stepsize in enumerate(step_sizes):
            plt.scatter(all_times[k, i], all_costs[k, i], label=f'step: {stepsize}')
        plt.xlabel('Computation Time')
        plt.ylabel('Cost')
        plt.title(f'Cost vs Computation Time (Averaged over {num_runs} Runs)')
        plt.legend()
        direc_name = os.path.join('plots', f'bias={bias}')
        plt.savefig(os.path.join(direc_name, 'all_runs_plot.png'))
        plt.close()
    
if __name__ == '__main__':
    main()



