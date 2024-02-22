#These imports deal with the file handeling and hdf5 file opening.
import h5py
import os,sys
import numpy as np
import ast
import matplotlib.pyplot as plt
#For creating a movie
import matplotlib.animation as animation

#Contants
pc = 3.086*10**(16) ##[m]
#Sets the directory to the current location
os.chdir(os.getcwd())
#Find all the files in the listed directory. 
files = os.listdir()
#Defines a filter that removes all none snapshot files from the list.
def filter_files(file):
    if (file[0:10] == "disc_patch"):
        return True
    else:
        return False
#Recieves filtered list
filtered_files = np.sort(list(filter(filter_files,files)))


#Code to create a grid from the HDF5 file (Not my code)
def get_tb_grid(grid,subgriddim,gridsize):
    result = np.zeros((gridsize[0],gridsize[1],gridsize[2]))
    cx = int(gridsize[0]/subgriddim[0])
    cy = int(gridsize[1]/subgriddim[1])
    cz = int(gridsize[2]/subgriddim[2])
    startchunk = 0
    endchunk = cx*cy*cz
    ix = 0
    iy = 0
    iz = 0
    while endchunk <= gridsize[0]*gridsize[1]*gridsize[2]:
        chunk = np.array(grid[startchunk:endchunk])
        result[ix:ix+cx,iy:iy+cy,iz:iz+cz] = chunk.reshape(cx,cy,cz)
        startchunk += cx*cy*cz
        endchunk += cx*cy*cz
        iz += cz
        if iz == gridsize[2]:
            iz = 0
            iy += cy
            if iy == gridsize[1]:
                iy = 0
                ix += cx
    return result
#Also mostly not my code but uses the hdf5 file to get relevant data.
def DataGetter(filename, keys = False):
    with h5py.File(filename) as file:
        #get simulation box dimension in SI and pc
        box = np.array(file["/Header"].attrs["BoxSize"])
        box_pc = box/pc
        #Get the time when the snapshot was saved in SI(S) and 
        time = (file["/Header"].attrs["Time"])
        time_Myr = time/(3600*24*365.25*1000000)
        #get gridcell dimensions 
        grid = ast.literal_eval(file["/Parameters"].attrs["DensityGrid:number of cells"].decode("utf-8") )
        grid = np.array(grid)
        #get task based subgrid dimensions
        subgrids = ast.literal_eval(file["/Parameters"].attrs["DensitySubGridCreator:number of subgrids"].decode("utf-8"))
        subgrids = np.array(subgrids)
        #voxel side length SI
        pix_size = box[0]/grid[0]
        #simulation data stored in here
        filepart = file['PartType0']
        #this should print all available datasets eg. 'Temperature' 'NumberDensity' etc..
        if keys:
            print("File groups:",file.keys())
            print("Attributes inside Header:",file["/Header"].attrs.keys())
            print("Measured quantiies:",filepart.keys())
        #then get the datasets by...
        ntot = get_tb_grid(filepart['NumberDensity'],subgrids,grid)
        #which gives the full Cartesian grid
        tempGrid = get_tb_grid(filepart["Temperature"],subgrids,grid)
        NeutralFractionGrid  = get_tb_grid(filepart["NeutralFractionH"],subgrids,grid)
        IonisedHydrogenGrid = ntot*(1-NeutralFractionGrid)
        return ntot, tempGrid,IonisedHydrogenGrid,box_pc,time_Myr
#This updates the animation. Pass the filename and it retrives the relevant data and updates the sidebars.
def Update(filename):
    NewData = DataGetter(filename)
    im1.set_data(NewData[0].sum(0))
    im2.set_data(NewData[1].sum(0))
    im3.set_data(NewData[2].sum(0))
    plt.suptitle(f"{NewData[4]:3f}Myr")
    im1.autoscale()
    im2.autoscale()
    im3.autoscale()
    return im1, im2,im3

fig, ax = plt.subplots(1,3,figsize=(15,5))

#Labels the python figures
ax[0].set_title("Total Number Density")
ax[1].set_title("Number Density of H+")
ax[2].set_title("Temperature")
ax[0].set_xlabel("Pc")
ax[0].set_ylabel("Pc")
ax[1].set_xlabel("Pc")
ax[1].set_ylabel("Pc")
ax[2].set_xlabel("Pc")
ax[2].set_ylabel("Pc")

data = DataGetter(filtered_files[0],False)
BoxDim = data[3]
TimeMyr = data[4]
plt.suptitle(f"{TimeMyr:.3f} Myr")
im1 = ax[0].imshow((data[0].sum(0)), extent = [-BoxDim[1]/2 , BoxDim[1]/2,-BoxDim[2]/2 , BoxDim[2]/2], 
                   cmap = "hot",aspect = "equal",norm = "linear") 
im2 = ax[2].imshow((data[1].sum(0)), extent = [-BoxDim[1]/2 , BoxDim[1]/2,-BoxDim[2]/2 , BoxDim[2]/2], 
                   cmap = "hot",aspect = "equal",norm = "log")
im3 = ax[1].imshow((data[2].sum(0)), extent = [-BoxDim[1]/2 , BoxDim[1]/2,-BoxDim[2]/2 , BoxDim[2]/2], 
                   cmap = "hot",aspect = "equal", norm = "linear")

bar = plt.colorbar(im1)
bar1 = plt.colorbar(im2)
bar2 = plt.colorbar(im3)


#This creates the animation using the Update function, and starts from 0th frame. Outputs the file as Movie.gif
print("Starting Movie Creation")
animation2 = animation.FuncAnimation(fig=fig, func = Update, frames = filtered_files) 
animation2.save("Movie.gif", fps = 15)
print("Created Movie: Done!")

data = DataGetter(filtered_files[0],False)
BoxDim = data[3]
TimeMyr = data[4]
plt.suptitle(f"{TimeMyr:.3f} Myr")
im1 = ax[0].imshow((data[0].sum(0)), extent = [-BoxDim[1]/2 , BoxDim[1]/2,-BoxDim[2]/2 , BoxDim[2]/2], 
                   cmap = "hot",aspect = "equal",norm = "log") 
im2 = ax[2].imshow((data[1].sum(0)), extent = [-BoxDim[1]/2 , BoxDim[1]/2,-BoxDim[2]/2 , BoxDim[2]/2], 
                   cmap = "hot",aspect = "equal",norm = "log")
im3 = ax[1].imshow((data[2].sum(0)), extent = [-BoxDim[1]/2 , BoxDim[1]/2,-BoxDim[2]/2 , BoxDim[2]/2], 
                   cmap = "hot",aspect = "equal", norm = "log")

bar = plt.colorbar(im1)
bar1 = plt.colorbar(im2)
bar2 = plt.colorbar(im3)

#This creates the animation using the Update function, and starts from 0th frame. Outputs the file as Movie.gif
print("Starting Movie Creation")
animation2 = animation.FuncAnimation(fig=fig, func = Update, frames = filtered_files) 
animation2.save("Movie.gif", fps = 15)
print("Created Movie: Done!")
