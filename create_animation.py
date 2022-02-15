import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import math
import random

class Simulation:
    def __init__(self, sim_input):
        self.time = 0
        self.total_time = sim_input["total_time"]
        self.dt = 1e-13 #seconds
        self.time_steps = int(self.total_time/self.dt)
        self.temperature = sim_input["temperature"] #kelvin
        self.particle_mass = 5.3e-26 #kg
        self.particle_radius = 1.5e-10 #m
        self.boltzmann_constant = 1.381e-23 #m^2kg/s^2K
        self.avg_MB_velocity = math.sqrt((2*self.boltzmann_constant*self.temperature)/self.particle_mass)
        self.box_size_x = 50e-10
        self.box_size_y = 50e-10
        self.particle_count = sim_input["particle_count"]
        return None

    def initialise_arrays(self):
        self.all_position_data = np.zeros((self.particle_count*self.time_steps,2),dtype=np.float)
        self.particle_positions = np.random.rand(self.particle_count, 2)
        self.particle_velocities = np.full((self.particle_count, 2), self.avg_MB_velocity)
        self.particle_velocities = np.multiply(self.particle_velocities,[([-1,1][random.randrange(2)],[-1,1][random.randrange(2)]) for i in range(self.particle_count)])
        self.particle_positions = np.subtract(self.particle_positions, 0.5)
        self.particle_positions[:,0] = np.multiply(self.particle_positions[:,0], self.box_size_x)
        self.particle_positions[:,1] = np.multiply(self.particle_positions[:,1], self.box_size_y)

    def run_simulation(self):
        for step in range(self.time_steps):
            self.save_arrays(step)
            self.particle_positions += self.particle_velocities * self.dt
            self.calculate_collisions()
            self.time += self.dt

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
                if m>n: #stops calculating collisions with self and double collisions
                    if math.sqrt(pow(p_x-p2_x,2) + pow(p_y-p2_y,2))<=2*self.particle_radius +1e-14:
                        print("collision!")
                        self.stop_clipping(n, m)
                        xdiff = self.particle_positions[n, 0] - self.particle_positions[m, 0]
                        ydiff = self.particle_positions[n, 1] - self.particle_positions[m, 1]
                        vxdiff = self.particle_velocities[n, 0] - self.particle_velocities[m, 0]
                        vydiff = self.particle_velocities[n, 1] - self.particle_velocities[m, 1]

                        self.particle_velocities[n, 0] = self.particle_velocities[n, 0] - xdiff * (xdiff*vxdiff+ydiff*vydiff)/(4*self.particle_radius*self.particle_radius)
                        self.particle_velocities[n, 1] = self.particle_velocities[n, 1] - ydiff * (xdiff*vxdiff+ydiff*vydiff)/(4*self.particle_radius*self.particle_radius)
                        self.particle_velocities[m, 0] = self.particle_velocities[m, 0] + xdiff * (xdiff*vxdiff+ydiff*vydiff)/(4*self.particle_radius*self.particle_radius)
                        self.particle_velocities[m, 1] = self.particle_velocities[m, 1] + ydiff * (xdiff*vxdiff+ydiff*vydiff)/(4*self.particle_radius*self.particle_radius)

                        # temp_nx = self.particle_velocities[n, 0]
                        # temp_ny = self.particle_velocities[n, 1]
                        # self.particle_velocities[n, 0] = self.particle_velocities[m, 0]
                        # self.particle_velocities[n, 1] = self.particle_velocities[m, 1]
                        # self.particle_velocities[m, 0] = temp_nx
                        # self.particle_velocities[m, 1] = temp_ny


    def stop_clipping(self,n,m):
        #this is suppost to stop particle clipping by teleporting them out of eachother
        xdiff = self.particle_positions[n, 0] - self.particle_positions[m, 0]
        ydiff = self.particle_positions[n, 1] - self.particle_positions[m, 1]
        theta = math.atan(abs(ydiff / (xdiff + 1e-15)))
        rdiff = math.sqrt(xdiff * xdiff + ydiff * ydiff)
        r_overlap = (self.particle_radius*2) - rdiff
        overlap_x = r_overlap * math.cos(theta)
        overlap_y = r_overlap * math.sin(theta)
        #the proportion each particle is moved back should be dependent on velocity not split equally
        self.particle_positions[n, 0] = self.particle_positions[n, 0] + overlap_x/2 * self.get_sign(xdiff) #essentialy passes a position vector from a to b or b to a
        self.particle_positions[n, 1] = self.particle_positions[n, 1] + overlap_y/2 * self.get_sign(ydiff)
        self.particle_positions[m, 0] = self.particle_positions[m, 0] + overlap_x/2 * self.get_sign(-xdiff)
        self.particle_positions[m, 1] = self.particle_positions[m, 1] + overlap_y/2 * self.get_sign(-ydiff)

    def get_sign(self, x):
        if x <0:
            return -1
        elif x>=0:
            return 1

    def save_arrays(self, step):
        self.all_position_data[step*self.particle_count:(step+1)*self.particle_count,:] = self.particle_positions

class Plotting():
    def __init__(self, sim, file_path):
        self.fig = plt.figure()
        self.ax = plt.subplot()
        self.ax.set_xlim(-sim.box_size_x, sim.box_size_x)
        self.ax.set_ylim(-sim.box_size_y, sim.box_size_y)
        self.box = patches.Rectangle((-sim.box_size_x/2,-sim.box_size_y/2),sim.box_size_x,sim.box_size_y,fill=False)
        self.ax.add_patch(self.box)
        self.timestamp = self.ax.text(.03, .94, 'Time: ', color='b', transform=self.ax.transAxes, fontsize='x-large')
        self.lines = []
        self.plots = []
        self.initiate_plot(sim)
        self.directory = file_path

    def initiate_plot(self, sim):
        for i in range(sim.particle_count):
            plot = self.ax.scatter([],[],color='black',s=60)
            line, = self.ax.plot([],[],color='blue',linewidth=1)
            self.lines.append(line)
            self.plots.append(plot)

    def start_animation(self, sim):
        self.ani = animation.FuncAnimation(self.fig, self.animate, repeat=True, frames=int(sim.time_steps/1),
                                       fargs=(sim,), blit=True, interval=10, )
        self.ani.save(self.directory, writer='imagemagick', fps=60)


    def animate(self, i, sim):
        for p in range(sim.particle_count):
            self.plots[p].set_offsets(sim.all_position_data[p+(i)*sim.particle_count, :])
        self.timestamp.set_text(f"Timestep: {i}")
        return self.plots + [self.timestamp]


# sim = Simulation()
# plot = Plotting(sim)
#plt.show()
print("done")