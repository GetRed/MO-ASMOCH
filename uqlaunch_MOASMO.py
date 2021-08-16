#%%
# launch uq workflow with this script

import sys
sys.path.append('./src')
import UQ
from cfg_testf import modelcfg
from cfg_MOASMO import uqcfg
# from cfg_NSGAm import uqcfg
import numpy as np
import matplotlib.pyplot as plt


uqres = UQ.workflow(modelcfg, uqcfg)

y = uqres['uqres']['optimization']['result']['y']
besty = uqres['uqres']['optimization']['result']['besty']
bestx = uqres['uqres']['optimization']['result']['bestx']


y_true = np.loadtxt('./reference_points_RE22.dat')

plt.plot(besty[:,0],besty[:,1],'r.',label='MO-ASMOCH optimal')
plt.plot(y_true[:,0],y_true[:,1],'k.',markersize=2.,label='True Pareto')

plt.xlim([0, 400])
plt.xlabel('y1')
plt.ylabel('y2')
plt.legend()
plt.show()


