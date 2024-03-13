import numpy as np
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import Building_Blocks
from visualizer import Visualize_UR
from itertools import product

def main():
    # ur_params = UR5e_PARAMS(inflation_factor=1)
    # env = Environment(env_idx=2)
    # transform = Transform(ur_params)
    
    # for stepsize,bias in product([0.1, 0.3, 0.5, 0.8, 1.2, 1.7, 2.0], [0.05, 0.2]):

    #     bb = Building_Blocks(transform=transform, 
    #                         ur_params=ur_params, 
    #                         env=env,
    #                         resolution=0.1, 
    #                         p_bias=bias,)
        
    #     rrt_star_planner = RRT_STAR(max_step_size=stepsize,
    #                                 max_itr=2000, 
    #                                 bb=bb)
        
    #     #visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)
        
    #     # --------- configurations-------------
    #     env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    #     env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    #     # ---------------------------------------
    #     filename = 'task2'
    #     path = rrt_star_planner.find_path(start_conf=env2_start,
    #                                         goal_conf=env2_goal,
    #                                         filename=filename)
    #     cost = rrt_star_planner.compute_cost(path)
    #     shortest_path, shortest_cost = None, None

    #     print("finished ", stepsize, ":",len(path),path)       
    #     try:
    #         np.save(filename+'_path'+str(stepsize)+'_'+str(bias), path)
    #         #path = np.load(filename+'_path'+str(stepsize)+'_'+str(bias)+".npy")
    #         #visualizer.show_path(path)
    #         #visualizer.show_conf(env2_goal)
    #         if shortest_path == None and cost > 0: # make sure we found a path
    #             shortest_path = path
    #             shortest_cost = cost
    #             shortest_step = stepsize
    #             shortest_bias = bias
    #         elif cost > 0 and cost < shortest_cost:
    #             shortest_path = path
    #             shortest_cost = cost
    #             shortest_step = stepsize
    #             shortest_bias = bias
            
    #     except:
    #         print('No Path Found')

    # try:
    #     np.save(filename+'shortest_path_stepsize='+str(shortest_step)+'_bias='+str(shortest_bias)+'cost='+str(shortest_cost), shortest_path)
    # except:
    #     print('shortest path computation error')
    
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)
    

    bb = Building_Blocks(transform=transform, 
                        ur_params=ur_params, 
                        env=env,
                        resolution=0.1, 
                        p_bias=0.05,)
    rrt_star_planner = RRT_STAR(max_step_size=2,
                                    max_itr=2000, 
                                    bb=bb)
    computed_path = np.array([[1.9198621771937625, -1.2217304763960306, 1.5707963267948966, -1.5707963267948966, -1.5707963267948966, 0],
                                [1.512619413494538, -2.7247946225869515, -0.15320880086596445, -2.687868216527126, -1.2370550483122993, -0.30693003886979797],
                                [1.35325956023292, -2.208541643872767, 0.8678777971879921, -0.927072230137759, -2.9689847784393164, 0.11446748814539243],
                                [0.8726646259971648, -1.3962634015954636, 1.5707963267948966, -1.5707963267948966, -1.5707963267948966, 0],
                                [0.8726646259971648, -1.3962634015954636, 1.5707963267948966, -1.5707963267948966, -1.5707963267948966, 0]])
    computed_cost = rrt_star_planner.compute_cost(computed_path)
    print("The cost of the best path we found is: ", computed_cost)
if __name__ == '__main__':
    main()



