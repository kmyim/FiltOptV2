import os                                                                       
from multiprocessing import Pool                                                
                                                                                
                                                                                
processes = []
slices = [1, 2, 3, 4]
m0 = [3,4,5]
do = [0.25, 0.5]

for s in slices:
    for m in m0:
        for d in do:
            processes.append('--s ' + str(s) + ' --m0 ' + str(m) + ' --do ' + str(d))
                                                  
print(processes[0])                                                            
def run_process(process):                                                             
    os.system('python3 command_line_test.py {}'.format(process))                                       
                                                                                
                                                                                
pool = Pool(processes=len(slices)*len(m0)*len(do))                                                        
pool.map(run_process, processes)  