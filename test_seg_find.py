from numpy import array, linspace, sin
import matplotlib.pyplot as plt

# diff = array([-1,-0.5,0.4,0.2,0.5,0.01,-0.2,0.02,0.3,0.6,0.8,10,0.3,-0.2,-0.2])
nwz=50
zp_lst=linspace(0,5,nwz)
diff=sin(zp_lst*5)-0.6*sin(zp_lst*16+1)
# diff=zp_lst-2
num_seg=0; zp_seg=[]

if ~(diff.any()>0): # integrating over all domain, Fr2>Fr2_crit
	num_seg = 1
	zp_seg=[zp_lst.min(), zp_lst.max()]
else:
	if diff[0] > 0: #Rising edge
		num_seg = 1; zp_R_edge = zp_lst[0]
	i=0
	while i < nwz-1:
		if diff[i] > 0 and diff[i+1] <= 0: #Falling edge
			zp_F_edge=zp_lst[i]-diff[i]/((diff[i]-diff[i+1])/(zp_lst[i]-zp_lst[i+1]))
			zp_seg.append((zp_R_edge,zp_F_edge))
		elif diff[i] < 0 and diff[i+1] >= 0: #Rising edge
			zp_R_edge=zp_lst[i]-diff[i]/((diff[i]-diff[i+1])/(zp_lst[i]-zp_lst[i+1]))
			num_seg=num_seg+1
		i = i + 1
	if diff[nwz-1] > 0: #Rising edge
			zp_seg.append((zp_R_edge,zp_lst[nwz-1]))
plt.plot(zp_lst,diff); plt.plot([0,5],[0,0])