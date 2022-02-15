from flask import Flask,request,render_template
from create_animation import Simulation, Plotting
import os
import uuid

app = Flask(__name__, instance_path="/{project_folder_abs_path}/instance")

@app.route('/', methods=['GET','POST'])
def index():
    file_name = "/static/images/animation.gif"
    request_type_str = request.method
    sim_input = {"total_time":1e-11, "temperature":300, "particle_count":30}
    if request_type_str != 'GET':
        sim_input["total_time"] = request.form.get("run_time", type=float)
        sim_input["temperature"] = request.form.get("temperature", type=int)
        sim_input["particle_count"] = request.form.get("particle_count", type=int)
        file_name = "/static/images/" + uuid.uuid4().hex + ".gif"
        create_gif(os.getcwd()+file_name, sim_input)
    else:
        if os.path.exists(os.getcwd()+file_name) == False:
            """If the animation does not exist, then the simulation is setup and run and then animated"""
            create_gif(os.getcwd()+file_name, sim_input)
    return render_template('index.html', href=file_name)

def create_gif(file_name, sim_input):
    sim = Simulation(sim_input)
    sim.initialise_arrays()
    sim.run_simulation()
    plot = Plotting(sim, file_name)
    plot.start_animation(sim)

if __name__ == '__main__':
    app.run(port=7777)