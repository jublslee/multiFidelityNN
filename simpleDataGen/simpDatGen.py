# Call RD simulator and plot contours
from datGenHelper import *
from funcZip import *

#---------specify input parameters--------#
# x_L       = [i / 10 for i in range(11)]
# print(x_L)
x_L = 1
x_H = [0,0.4,0.6,1]
#--------------------------------------------------------#


#--------------------plotting options---------------------# 
# range: [ (lower limit for c1, upper limit for c1), (lower limit for c2, upper limit for c2)]     
# cbar_range = [(-0.9, -0.6),(0, 0.2)] 
colorbar   = False # if False, remove colorbar
ticksoff   = False # if true, remove ticks
#--------------------------------------------------------#


# Call PDEbenCH solver
# Time intgration: RK4 
# Spatial discretization: 1st-order FV method
#---------------------------------------------------#
# Note: max_int+1 because of the initial condition
# Note: output 4D RD_sol tensor: 
		# num_Time_instance(max_int + 1) x number of cells in X-direc (dimx) x number of cells in Y-direc (dimy) x number of species (2)      
# RD_sol   = RD_simulator( Dc1_star, Dc2_star, kappa_star, t_star, max_int+1, dimx, dimy )
#---------------------------------------------------#


# plot solution at the simulation ending time
#-----------------------------------------------------#
# path     = 'Solutions/' # where to save
# fig_name = 'exact-sol-T=' + str(t_star) # name of the picture
# RD_plotter(path, fig_name , RD_sol, -1, colorbar = colorbar, ticksoff = ticksoff)
#-----------------------------------------------------#