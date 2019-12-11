from fluidsimulation import *


# How many timesteps in the simulation
timesteps = 1000


# Define the simulation
# n - size of the simulation
# windSpeed - speed of the wind
# windDirection - direction of the wind (in degrees)
# gasLocationX/Y - location of the source of the leak
# gasRelease - how much "particles" are released at the source
# diffusion - diffusion of the gas
# viscosity - viscosity of the gas
# windNoise - how much the wind speed direction is changed as the simulation is unrolled (in degrees)
# windNoiseTimestep - how ofteh the wind speed direction is changed as the simulation is unrolled
fs = FluidSimulation(64, windSpeed=0.1, windDirection = 60.0, gasLocationX=32, gasLocationY=32, gasRelease=300, diffusion=0.0001, viscosity=0, windNoise=180, windNoiseTimestep=30)


# Save the data to a file
f = h5py.File("test.hdf5", "w")


# Run the simulation
for i in range(timesteps):
    fs.update(f)
    print ("Sim: "+str(i))

# Load the data from the saved file
file = h5py.File("test.hdf5", "r")

# Visualize the data - every 10th image is shown
for i in range(0,timesteps,10):
    plt.clf()
    # Show the gas concetration as an image
    plt.imshow(file['readings'+str(i)], vmin=0, vmax=10)
    plt.colorbar()
    plt.pause(0.000000001)
    print ("Render: "+str(i))
