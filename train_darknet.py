import numpy as np
import visdom
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./', help='path to darknet binary file')
parser.add_argument('--task', required=True, help='classifier or detector')
parser.add_argument('--data', required=True, help='path to .data config file')
parser.add_argument('--cfg', required=True, help='path to .cfg config file')
parser.add_argument('--weights', required=False, help='path to weights file')
parser.add_argument('--gpus', default='0', help='ids of gpus to use, ex: "0,1,2,3"')

opt = parser.parse_args()

print(opt)

if opt.task not in ['classifier', 'detector']:
    raise Exception('Task must be classifier or detector')

# Create list of args for subprocess that is going to call Darknet
list_arg = [opt.path+"darknet", opt.task, 'train', opt.data, opt.cfg]
if opt.weights is not None:
    list_arg.append(opt.weights)
list_arg.extend(["-gpus", opt.gpus])

print(list_arg)

# Keep a log of training process
log_list = []
log_file = open('log.txt', 'w')

# Run Darknet training
darknet = subprocess.Popen(list_arg, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,bufsize=0)
with darknet.stdout:
    for line in iter(darknet.stdout.readline, b''):
        print line,
        if line[-7:] == 'images\n':
            log_file.write(line)
            iteration, rest = line.split(':')
            values = [value[1:].split(' ')[0] for value in rest.split(',')]
            total_loss, average_loss, learning_rate, wall_time, total_images = values
            #print('[Python wrapper] This is the {} iteration\n'.format(iteration)),
            log_list.append({
                'iteration': int(iteration),
                'total_loss': float(total_loss),
                'average_loss': float(average_loss),
                'learning_rate': float(learning_rate),
                'wall_time': float(wall_time),
                'total_images': int(total_images)
            })
            if int(iteration) > 50:
                darknet.terminate()
darknet.wait()
log_file.close()
