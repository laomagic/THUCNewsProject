import os


current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(current_path)[0])[0]

data_path = root_path + '/data/THUCNews/'
