'''
What EVO requires are the quaternion and translation vector of the pose transformation 
from the world coordinate system (ID 0) to the viewing perspective coordinate system (ID).
'''

import os
import sys
import numpy as np
import transforms3d as tf

if __name__ == '__main__':
    file_path = sys.argv[1]
    save_file_name = sys.argv[2]
    parent_path = os.path.dirname(file_path)
    print(parent_path)
    if save_file_name == "L":
        save_file = open(os.path.join(parent_path, "evo_right_to_left.log"), 'w')
    elif save_file_name == "N":
        save_file = open(os.path.join(parent_path, "evo_left_to_right.log"), 'w')
    gtlogname = os.path.join(parent_path, save_file_name)
    with open(file_path, 'r') as file:
        datas = file.readlines()
        i = 0
        iter = 0
        while i < len(datas):
            if datas[i] == '' or datas[i] == ' ':
                break
            if gtlogname.endswith('gtLo.log'):
                ids = datas[i].split('\t')
                h1 = [float(item) for item in datas[i + 1].split('\t')]
                h2 = [float(item) for item in datas[i + 2].split('\t')]
                h3 = [float(item) for item in datas[i + 3].split('\t')]
                R = [h1[:3], h2[:3], h3[:3]]
                t = [h1[3], h2[3], h3[3]]
                R = np.array(R)
            else :
                ids = datas[i].split(' ')
                h1 = []
                for item in datas[i+1].split(' '):
                    if item != '':
                        h1.append(float(item))
                h2 = []
                for item in datas[i+2].split(' '):
                    if item != '':
                        h2.append(float(item))
                h3 = []
                for item in datas[i+3].split(' '):
                    if item != '':
                        h3.append(float(item))

                R = [h1[:3], h2[:3], h3[:3]]
                t = [h1[3], h2[3], h3[3]]
                R = np.array(R)
            q = tf.quaternions.mat2quat(R)
            t_str = [str(item) for item in t]
            t_str = ' '.join(t_str)
            write_result = str(iter) + " " + t_str + " " + str(q[1]) + " " + str(q[2]) + " " + str(q[3]) + " " + str(q[0])
            save_file.write(write_result + '\n')
            i += 5
            iter += 1
