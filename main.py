from numpy import pi, zeros, linspace, loadtxt, array, insert, interp, ones, transpose
from scipy.sparse.linalg import spsolve
from scipy.sparse import dia_matrix, csc_matrix
from Utilities import get_ug, assemble_matrix, assemble_rhs
from Model import Jent_numerical_prev, Jent_numerical_New
import matplotlib.pyplot as plt
import time as t
from scipy.io import loadmat
t1=t.time()
####   EVERYTHING IS DIMENSIONAL   ####
# Input parameters
model = 2
Sl0 = 1.9e-2
L0  = 47 				#Athena Ship Length
U0  = 10.5 * 0.514444	# 10.5 knot speed
#######################################################
# Phyical properties    #Numerical parameters
rhoc = 1000;            zmax = 5.0
rhod = 1.204;           Tend = 10.0
nuc = 1.0e-6;           Nx = 30
sigma = 0.072;          Nt = 10
g = 9.81
z = linspace(0,zmax,Nx)   #Domain
dt = Tend/Nt
############################################################
#Read groups info
ds = loadtxt('groups.dat', skiprows=1)
D =ds[:,0]*L0*2	#Diameters
Dg=ds[:,1]		#Entrainment distribution
G = D.size
turb_data = loadtxt('stern_turbulence.dat',skiprows=18);
kt  = -1*interp(-1*z,-1*insert(turb_data[:,5]*L0,0,10), -1*insert(turb_data[:,6] ,0,turb_data[0,6] ));
et  = -1*interp(-1*z,-1*insert(turb_data[:,5]*L0,0,10), -1*insert(turb_data[:,11],0,turb_data[0,11]));
nut = -1*interp(-1*z,-1*insert(turb_data[:,5]*L0,0,10), -1*insert(turb_data[:,8] ,0,turb_data[0,8] ));
for i in range(Nx):
	if z[i] > 0.5 :
		nut[i] = nuc

#Compute group sizes
dDg=zeros(G)
dDg[0] = 0.5*(D[0]+D[1])
for ig in range(1,G-1):
	dDg[ig] = 0.5*(D[ig+1]-D[ig-1])

dDg[G-1] = 2*dDg[G-2]-dDg[G-3]
vg = pi/6*(D**3)	#Bubble volume
v0 = sum(Dg*vg)		#Mean entrainned bubble volume

#Terminal velocity
VT=zeros(G)
for ig in range(G):
	VT[ig] = get_ug(D[ig],rhoc,rhod,sigma,nuc,1.0)

# Calculate pz:
L11 = 0.43*(kt**1.5)/et
LD=zeros((Nx,G));   Lz=zeros((Nx,G));   pz=zeros((Nx,G)); n=zeros((Nx,G))
for ig in range(G):
	LD[:,ig]=nut/VT[ig]
	Lz[:,ig]=1/(1/L11+1/LD[:,ig])
	n[:,ig] = L11/LD[:,ig]
	#Variable turbulence
	arg = ((1-z/L11)+abs(1-z/L11))/2
	pz[:,ig]=(arg**n[:,ig])/Lz[:,ig];

#Find ph
phg = Dg/pz[0,:]
phg = phg/sum(phg)

################## MATRIX ##################
#Create sparse matrx
diags = ones((3,Nx))
alower = 1
adiag  = 2
aupper = 3
diags_idxs = array([-1,0,1])
A=[]
for ig in range(G):
	A_tmp = dia_matrix((diags,diags_idxs),shape=(Nx,Nx))
	A_tmp=csc_matrix(A_tmp)
	A.append(A_tmp)
	A[ig] = assemble_matrix(A[ig],z,nut,VT[ig],dt)
b = zeros((Nx,1)); S = zeros(Nx); alpha = zeros(Nx)
#Solution vectors
N  = zeros((Nx,G));	N0 = zeros((Nx,G));

#Compute the part of the source that does not change with time
for i in range(Nx):
	t2=t.time()
	if model == 1:
		# Previous model
		J=Jent_numerical_prev(kt[i],et[i])
		S[i] = Sl0*J/v0;
	if model == 2:
		# Previous model
		J=Jent_numerical_New(kt[i],et[i],nuc,g,rhoc,sigma)
		S[i] = J/v0;
	print ('>>>> i={:3d}, t={:.2f}s'.format(i,t.time()-t2))
print('Matrix build complete, time:{:.4f}s'.format(t.time()-t1))
time=zeros(Nt);	alpha0=zeros(Nt)

N_py = zeros((Nx,G));	N_ma = zeros((Nx,G));
for itime in range(Nt):
	print(itime)
	time[itime] = itime*dt;	N0 = N
	for ig in range(G):
		#################################################################
		#Compute source (using the previous void fraction)
		Sg=S*(0.3-alpha)*phg[ig]*pz[:,ig]
		#################################################################
		#Assemble RHS
		b=assemble_rhs(N0[:,ig],Sg,dt)
		N[:,ig] = spsolve(A[ig],b)
	#Void fraction
	alpha=zeros(Nx)
	for ig in range(G):
		alpha = alpha+vg[ig]*N[:,ig];
	alpha0[itime] = alpha[1]; #void fraction at the free surface
