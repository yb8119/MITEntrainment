import time as t
from numpy import pi, ones, array, zeros
from scipy.sparse.linalg import spsolve
from Utilities import get_ug, assemble_matrix, assemble_rhs
from scipy.sparse import dia_matrix, csc_matrix
from Model import Jent_numerical_prev, Jent_numerical_New
def Turb_entrainment(Nx,Nt,kt,et,g,rhoc,rhod,sigma,z,nuc,nut,G,D,Dg,dt,Sl0,Table,model,zlam_min,zlam_max,wmeth,Fr2_crt_PolyExtra,F_tab_NP):
	t1=t.time()
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
	# alower = 1
	# adiag  = 2
	# aupper = 3
	diags_idxs = array([-1,0,1])
	A=[]
	for ig in range(G):
		A_tmp = dia_matrix((diags,diags_idxs),shape=(Nx,Nx))
		A_tmp=csc_matrix(A_tmp)
		A.append(A_tmp)
		A[ig] = assemble_matrix(A[ig],z,nut,VT[ig],dt)
	b = zeros((Nx,1)); S = zeros(Nx); 
	#Solution vectors
	N  = zeros((Nx,G));	N0 = zeros((Nx,G)); alpha = zeros(Nx)

	#Compute the part of the source that does not change with time
	for i in range(Nx):
		t2=t.time()
		if model == 1:
			# Previous model
			J=Jent_numerical_prev(kt[i],et[i])
			S[i] = Sl0*J/v0;
		if model == 2:
			# New model
			J=Jent_numerical_New(kt[i],et[i],nuc,g,rhoc,sigma,Table,zlam_min,zlam_max,wmeth,Fr2_crt_PolyExtra,F_tab_NP)
			S[i] = J/v0;
	print('Matrix build complete, time:{:.4f}s'.format(t.time()-t1))
	time=zeros(Nt);	alpha0=zeros(Nt); alpha_mid=zeros(Nt); alpha_out=zeros((Nx,Nt))

	for itime in range(Nt):
		time[itime] = (itime+1)*dt;	N0 = N
		for ig in range(G):
			#################################################################
			#Compute source (using the previous void fraction)
			Sg=S*(0.3-alpha)*phg[ig]*pz[:,ig]
			#################################################################
			#Assemble RHS
			b=assemble_rhs(N0[:,ig],Sg,dt)
			N[:,ig] = spsolve(A[ig],b)
		#Void fraction
		alpha = zeros(Nx)
		for ig in range(G):
			alpha = alpha+vg[ig]*N[:,ig];
		alpha_out[:,itime] = alpha
		alpha0[itime] 		= alpha[0]; #void fraction at the free surface
		alpha_mid[itime] 	= alpha[int(Nx/2)];
	# calculate the total source
	Sg=zeros(Nx); S_raw=zeros(Nx)
	for ig in range(G):
		Sg=Sg+S*(0.3-alpha)*phg[ig]*pz[:,ig]*vg[ig]
		S_raw=S_raw+S*phg[ig]*pz[:,ig]*vg[ig]
	return time, alpha_out, alpha0, alpha_mid, Sg, VT, S_raw