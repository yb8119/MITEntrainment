from numpy import zeros, linspace, load, flip
import matplotlib.pyplot as plt
from matplotlib import cm
print("========  Using Original Table (no extrapolation)  ========")
Table = load("Ent_table_org.npz")
Refitcoefs=Table['Refitcoefs'];	FrXcoefs=Table['FrXcoefs']
Fr2_lst=Table['flxfr_data']; zcoa_lst=Table['z_a_data']; F_tab=Table['F_lookuptable']

from Model import Fr_contribution
from Utilities import F_func_table_ext
zcoa_test = linspace(-5,-2, 4)
Fr2_test = linspace(0, 7, 11)

isNP = False
if isNP:
	method = "Nearest Point"
else:
	method = "Linear Exrapolation"
Ftab_NP1 = F_func_table_ext(
	Fr2_lst, Fr2_test, zcoa_lst, zcoa_test, F_tab, method)
Ftab_NP2 = zeros((zcoa_test.size, Fr2_test.size))
for j in range(Fr2_test.size):
	for i in range(zcoa_test.size):
		Ftab_NP2[i, j] = Fr_contribution(
			zcoa_test[i], Fr2_test[j], isNP, zcoa_lst, F_tab, Fr2_lst)


fig_NP, ax_NP = plt.subplots(figsize=(5,4), dpi=200)
fig_LN, ax_LN = plt.subplots(figsize=(5,4), dpi=200)
