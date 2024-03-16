import numpy as np
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import Building_Blocks
from visualizer import Visualize_UR
from itertools import product

def main():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)
    
    for stepsize,bias in product([0.1, 0.3, 0.5, 0.8, 1.2, 1.7, 2.0], [0.05, 0.2]):

        bb = Building_Blocks(transform=transform, 
                            ur_params=ur_params, 
                            env=env,
                            resolution=0.1, 
                            p_bias=bias,)
        
        rrt_star_planner = RRT_STAR(max_step_size=stepsize,
                                    max_itr=2000, 
                                    bb=bb)
        
        #visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)
        
        # --------- configurations-------------
        env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
        env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
        # ---------------------------------------
        filename = 'task2'
        path = rrt_star_planner.find_path(start_conf=env2_start,
                                           goal_conf=env2_goal,
                                           filename=filename)

        print("finished ", stepsize, ":",len(path),path)       
        try:
            #np.save(filename+'_path'+str(stepsize)+'_'+str(bias), path)
            path = np.load(filename+'_path'+str(stepsize)+'_'+str(bias)+".npy")
            cost = 0
            for c1, c2 in zip(path,path[1:]):
                cost += bb.edge_cost(c1,c2)
                
            print(str(stepsize)+' '+str(bias), cost, len(path))
            #visualizer.show_path(path)
            #visualizer.show_conf(env2_goal)
            
        except:
            print('No Path Found')
    
    
   

if __name__ == '__main__':
    main()



