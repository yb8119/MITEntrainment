from Utilities import findcLceta, ulambda_sq
from numpy import zeros, linspace, sqrt, pi, insert, savez, load
from Model import Fr2_crit_getter, get_rise_speed, int_seg_find
import matplotlib.pyplot as plt
Table = load("Ent_table_NP.npz")
FrXcoefs = Table['FrXcoefs']
fig = plt.figure(figsize=(8,6),dpi=300); ax1 = fig.add_subplot(111)
nwz = 10000
zlam_lst = linspace(1, 4, nwz)
Fr2_crit_lst = zeros(nwz)

l = 0.65


for i in range(nwz):
	Fr2_crit_lst[i]=Fr2_crit_getter(l,zlam_lst[i]*l,FrXcoefs,Fr2_crt_PolyExtra=False)
# Local minimum
for i in range(1, nwz-1):
	if Fr2_crit_lst[i] <= Fr2_crit_lst[i-1] and Fr2_crit_lst[i] < Fr2_crit_lst[i+1]:
		ax1.plot([zlam_lst[i],zlam_lst[i]],[Fr2_crit_lst[i],Fr2_crit_lst[i]], marker = 'o',color='red')
		# print("local MIN found: z/lam = {:.8e}.".format(zlam_lst[i]))
	if Fr2_crit_lst[i] >= Fr2_crit_lst[i-1] and Fr2_crit_lst[i] > Fr2_crit_lst[i+1]:
		ax1.plot([zlam_lst[i],zlam_lst[i]],[Fr2_crit_lst[i],Fr2_crit_lst[i]], marker = 'o',color='red')
		# print("local MAX found: z/lam = {:.8e}.".format(zlam_lst[i]))


ax1.plot(zlam_lst,Fr2_crit_lst,color='black')
ax1.plot([2,2],[0,1],color='black',linestyle='--')
ax1.plot([3,3],[0,1],color='black',linestyle='--')
ax1.set_xlim(zlam_lst.min()-0.1,zlam_lst.max()+0.1)
ax1.set_ylim(0,0.6)

# Now calculate Fr2

g = 9.81; kt = 1.0; et = 4.0; nuc = 1.0e-6;
cL, cEta = findcLceta(kt, et, nuc, mode=1)
ul2 = ulambda_sq(l,kt,et,cL,cEta,nuc,pope_spec=1.01)
ulam  =sqrt(ul2)
circ_p=pi*pi/4*l*ulam
Fr2_lst=zeros(nwz)
for i in range(nwz):
	wz = get_rise_speed(l,2*zlam_lst[i]*l,kt,et,nuc,cL,cEta,method=2)
	Fr2_lst[i] = circ_p*wz/(l**2/4*g)
ax1.plot(zlam_lst,Fr2_lst,color="blue")
print("==================================================================")
num_seg, zp_seg = int_seg_find(l,zlam_lst.min(),zlam_lst.max(),
							   kt,et,nuc,cL,cEta,FrXcoefs,circ_p,g,
							   Fr2_crt_PolyExtra=False)
for i in range(num_seg):
	ax1.fill_between( zlam_lst,Fr2_lst,where=(zlam_lst*l>zp_seg[i][0])*(zlam_lst*l<zp_seg[i][1]) )