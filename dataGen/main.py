# Call RD simulator and plot contours
from plotter import *
from sim_diff_react import *



#---------specify reaction diffusion coefficients--------#
Dc1_star       = 1e-3
Dc2_star       = 5e-2
kappa_star     = 1e-3
#--------------------------------------------------------#


#--------------------plotting options---------------------# 
# range: [ (lower limit for c1, upper limit for c1), (lower limit for c2, upper limit for c2)]     
# cbar_range = [(-0.9, -0.6),(0, 0.2)] 
colorbar   = False # if False, remove colorbar
ticksoff   = False # if true, remove ticks
#--------------------------------------------------------#


#---------specify space/time integration parameters-------------#
dt          = 1e-2 # time step size
max_int     = 400 # total time integration steps
t_star      = max_int*dt # simulation ending time
dimx        = 64  # number of FV-cells in x-direction
dimy        = 64  # number of FV-cells in y-direction
#---------------------------------------------------------#


# Call PDEbenCH solver
# Time intgration: RK4 
# Spatial discretization: 1st-order FV method
#---------------------------------------------------#
# Note: max_int+1 because of the initial condition
# Note: output 4D RD_sol tensor: 
		# num_Time_instance(max_int + 1) x number of cells in X-direc (dimx) x number of cells in Y-direc (dimy) x number of species (2)      
RD_sol   = RD_simulator( Dc1_star, Dc2_star, kappa_star, t_star, max_int+1, dimx, dimy )
#---------------------------------------------------#


# plot solution at the simulation ending time
#-----------------------------------------------------#
path     = 'Solutions/' # where to save
fig_name = 'exact-sol-T=' + str(t_star) # name of the picture
RD_plotter(path, fig_name , RD_sol, -1, colorbar = colorbar, ticksoff = ticksoff)
#-----------------------------------------------------#