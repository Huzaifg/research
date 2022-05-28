import os
import glob
import pandas as pd
import matplotlib.pyplot as mpl



path = './sens/'
extension = 'csv'

os.chdir(path)
data_files = glob.glob('*.{}'.format(extension))

cxf = ['']*len(data_files)
vx = [0]*len(data_files)
x = [0]*len(data_files)

for i,file_name in enumerate(data_files):
    cxf[i] = int(file_name.split('_')[0] + '000')
    data = pd.read_csv(file_name,sep=',',header='infer')
    vx[i] = data['vx'].iloc[-1]
    x[i] = data['x'].iloc[-1]


os.chdir('../')
# Longitudinal Velocity sensitivity
mpl.figure(figsize=(10,10))
mpl.scatter(cxf,vx)
mpl.title('Longitudinal Velocity Sensitivity - Chrono')
mpl.xlabel('Cxf/Cxr')
mpl.ylabel('Velocity at end of 8.5 (s) (m/s)')
mpl.xticks(cxf)
mpl.savefig('./images/s_lov.png',facecolor = 'w')
mpl.show()


# Distance travelled along X
mpl.figure(figsize=(10,10))
mpl.scatter(cxf,x)
mpl.title('X - Distance Sensitivity - Chrono')
mpl.xlabel('Cxf/Cxr')
mpl.ylabel('Distance at end of 8.5 (s) (m)')
mpl.xticks(cxf)
mpl.savefig("./images/s_dx.png",facecolor = 'w')
mpl.show()



