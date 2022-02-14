import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import math
import random
import time

class Simulation:
    def __init__(self):
        self.time = 0
        self.total_time = 1e-10
        self.dt = 1e-13 #seconds
        self.time_steps = int(self.total_time/self.dt)
        self.temperature = 300 #kelvin
        self.particle_mass = 5.3e-26 #kg
        self.particle_radius = 1.5e-10 #m
        self.boltzmann_constant = 1.381e-23 #m^2kg/s^2K
        self.avg_MB_velocity = math.sqrt((2*self.boltzmann_constant*self.temperature)/self.particle_mass)
        self.box_size_x = 50e-10
        self.box_size_y = 50e-10
        self.particle_count = 100
        self.initialise_arrays()
        self.run_simulation()
        return None

    def initialise_arrays(self):
        self.all_position_data = np.zeros((self.particle_count*self.time_steps,2),dtype=np.float)
        self.particle_positions = np.random.rand(self.particle_count, 2)
        self.particle_velocities = np.full((self.particle_count, 2), self.avg_MB_velocity)
        self.particle_velocities = np.multiply(self.particle_velocities,[([-1,1][random.randrange(2)],[-1,1][random.randrange(2)]) for i in range(self.particle_count)])
        self.particle_positions = np.subtract(self.particle_positions, 0.5)
        self.particle_positions[:,0] = np.multiply(self.particle_positions[:,0], self.box_size_x)
        self.particle_positions[:,1] = np.multiply(self.particle_positions[:,1], self.box_size_y)
        self.collision_matrix = np.zeros((self.particle_count,self.particle_count))

    def run_simulation(self):
        for step in range(self.time_steps):
            self.save_arrays(step)
            self.particle_positions += self.particle_velocities * self.dt
            self.calculate_collisions()
            if step%5 == 0: self.collision_matrix.fill(0) # change mod value to change timesteps between duplicate collisions
            self.time+=self.dt

    def calculate_collisions(self):
        #wall collisions
        for i, (p_x, p_y) in enumerate(zip(self.particle_positions[:,0],self.particle_positions[:,1])):
            if (p_x >= self.box_size_x/2 - self.particle_radius and self.particle_velocities[i,0] == abs(self.particle_velocities[i,0])):
                self.particle_velocities[i,0]=-self.particle_velocities[i,0]
            if (p_x <= -self.box_size_x/2+self.particle_radius and self.particle_velocities[i,0] != abs(self.particle_velocities[i,0])):
                self.particle_velocities[i,0]=-self.particle_velocities[i,0]
            if (p_y >= self.box_size_y/2 - self.particle_radius and self.particle_velocities[i,1] == abs(self.particle_velocities[i,1])):
                self.particle_velocities[i,1]=-self.particle_velocities[i,1]
            if (p_y <= -self.box_size_y/2 + self.particle_radius and self.particle_velocities[i,1] != abs(self.particle_velocities[i,1])):
                self.particle_velocities[i,1]=-self.particle_velocities[i,1]

        #particle collisions
        for n, (p_x, p_y) in enumerate(zip(self.particle_positions[:,0],self.particle_positions[:,1])):
            for m, (p2_x, p2_y) in enumerate(zip(self.particle_positions[:, 0], self.particle_positions[:, 1])):
                if m>n and self.collision_matrix[n,m] != 1: #stops calculating collisions with self and double collisions
                    if math.sqrt(pow(p_x-p2_x,2) + pow(p_y-p2_y,2))<2*self.particle_radius:
                        print("collision!")
                        self.collision_matrix[n,m] = 1
                        temp_nx = self.particle_velocities[n, 0]
                        temp_ny = self.particle_velocities[n, 1]
                        self.particle_velocities[n, 0] = self.particle_velocities[n, 0] + self.particle_velocities[n, 0]
                        self.particle_velocities[n, 1] = self.particle_velocities[m, 1]
                        self.particle_velocities[m, 0] = temp_nx
                        self.particle_velocities[m, 1] = temp_ny
                        #self.stop_coupling(p_x, p2_x, p_y, p2_y, n, m)


    # def stop_coupling(self,p_x,p2_x,p_y,p2_y,n,m):
    #     this is suppost to stop particle clipping by teleporting them out of eachother
    #     overlap_x = (self.particle_radius*2) - abs(abs(p_x)-abs(p2_x))
    #     overlap_y = (self.particle_radius*2) - abs(abs(p_y)-abs(p2_y))
    #     self.particle_positions[n, 0] += overlap_x/2 * self.get_sign(self.particle_velocities[n, 0])
    #     self.particle_positions[n, 1] += overlap_y/2 * self.get_sign(self.particle_velocities[n, 1])
    #     self.particle_positions[m, 0] += overlap_x/2 * self.get_sign(self.particle_velocities[m, 0])
    #     self.particle_positions[m, 1] += overlap_y/2 * self.get_sign(self.particle_velocities[m, 1])

    def get_sign(self, x):
        if x <0:
            return -1
        elif x>=0:
            return 1

    def save_arrays(self, step):
        self.all_position_data[step*self.particle_count:(step+1)*self.particle_count,:] = self.particle_positions

class Plotting():
    def __init__(self, sim):
        self.fig = plt.figure()
        self.ax = plt.subplot()
        self.ax.set_xlim(-0.5e-8, 0.5e-8)
        self.ax.set_ylim(-0.5e-8, 0.5e-8)
        self.box = patches.Rectangle((-sim.box_size_x/2,-sim.box_size_y/2),sim.box_size_x,sim.box_size_y,fill=False)
        self.ax.add_patch(self.box)
        self.timestamp = self.ax.text(.03, .94, 'Time: ', color='b', transform=self.ax.transAxes, fontsize='x-large')
        self.lines = []
        self.plots = []
        self.initiate_plot(sim)
        self.ani = animation.FuncAnimation(self.fig, self.animate, repeat=True, frames=int(sim.time_steps/1),
                                       fargs=(sim,), blit=True, interval=10, )
        #self.ani.save('C:/Users/melti/desktop/animation' + str(time.time()) + '.gif', writer='imagemagick', fps=60)
        plt.show()

    def initiate_plot(self, sim):
        for i in range(sim.particle_count):
            plot = self.ax.scatter([],[],color='black',s=60)
            line, = self.ax.plot([],[],color='blue',linewidth=1)
            self.lines.append(line)
            self.plots.append(plot)


    def animate(self, i, sim):
        for p in range(sim.particle_count):
           #self.lines[p].set_xdata(sim.all_position_data[0:p*i +p, 0])
            #self.lines[p].set_ydata(sim.all_position_data[0:p*i, 1])
            self.plots[p].set_offsets(sim.all_position_data[p+(i)*sim.particle_count, :])
        self.timestamp.set_text(f"Timestep: {i}")
        return self.plots + [self.timestamp]


sim = Simulation()
plot = Plotting(sim)
#plt.show()
print("done")