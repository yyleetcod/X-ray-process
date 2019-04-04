import os
path = './xray'
sep = '\n'
fl=open('label_list.txt', 'w')

def get_file(path):          #获取文件路径
    for root, dirs, files in os.walk(path):
        for file in files:
            dir_path = os.path.dirname(os.path.abspath(path))   
            #print(os.path.join(dir_path,file))
            fl.write(os.path.join(dir_path,file))
            fl.write('\n')
get_file(path)
