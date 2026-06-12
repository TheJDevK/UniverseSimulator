#information
#Phase 2A:Orbital mechanics: successful. Newtonian mechanics for two bodies in space worked. Verlet integration works properly.
#Phase 2B:N-body simulation: Successful. Now can add multiple bodies to the same universe
#Phase 2C:Front-end development: Successful. Simulation window made better with subtle changes.
#Phase 3A:improving architecture and universe class. New JKM architecture has made code more organised and reduced number of lines by nearly 15% while doing the same thing.: stable version pending.


import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

#universe class
class Universe:
    def __init__(self, dt, max_time, G, verbose):

        self.dt = dt
        self.max_time = max_time
        self.G = G
        self.verbose = verbose
        self.time = 0.0
        self.objects = []
        self.rows = []

    def add_object(self, obj):
        self.objects.append(obj)

    def total_energy(self):
        PE = 0

        for a in range(len(self.objects)):
            for b in range(a + 1, len(self.objects)):
                r_vec = self.objects[b].pos - self.objects[a].pos
                r = np.linalg.norm(r_vec)
                PE += -(self.G * self.objects[a].mass * self.objects[b].mass) / r

        KE = sum(o.K_energy for o in self.objects)
        E = KE + PE
        print("Total Energy = ", E)

    def total_momentum(self):
        p = np.array([0.0, 0.0])
        for o in self.objects:
            p += o.momentum
        print("Momentum:", p)

    def apply_gravity(self):

        for i in self.objects:
            i.acc[:] = 0

        for i in self.objects:
            for j in self.objects:
                if i == j:
                    continue
                r_vec = j.pos - i.pos
                r = np.linalg.norm(r_vec)
                if r == 0:   #divison by 0 prevented
                    continue
                i.acc += (
                        self.G
                        * j.mass
                        * r_vec
                        / r ** 3
                )

    def integrate(self):

        old_acc = [o.acc.copy() for o in self.objects]

        for o, a in zip(self.objects, old_acc):
            o.pos += (o.vel * self.dt + 0.5 * a * self.dt ** 2)

        self.apply_gravity()

        for o, a in zip(self.objects, old_acc):
            o.vel += ( 0.5 * (a + o.acc) * self.dt)

    def logger(self):

        for o in self.objects:
            if self.verbose:
                print(
                    f" {o.name} t={self.time:.2f}s  "
                    f"pos = ({o.pos[0]:.2f}, {o.pos[1]:.2f})  "
                    f"vel = ({o.vel[0]:.2f}, {o.vel[1]:.2f})  "
                    f"p = {o.momentum}  "
                    f"KE = {o.K_energy:.2f}"
                    f"Acc = {o.acc}"
                )

        snapshot = {"t": self.time}

        for body in self.objects:
            snapshot[f"{body.name}_x"] = body.pos[0]
            snapshot[f"{body.name}_y"] = body.pos[1]
        self.rows.append(snapshot)

        # -----Old dataframe design. Use later-----
        # rows.append({
        #    "t": current_time,
        #    "x1": ball1.pos[0],
        #    "y1": ball1.pos[1],
        #    "v1": ball1.vel[0],
        #    "x2": ball2.pos[0],
        #    "y2": ball2.pos[1],
        #    "v2": ball2.vel[0],
        #    "p_total": (ball1.momentum[0] + ball2.momentum[0]),
        #    "KE_total": ball1.K_energy + ball2.K_energy,
        #    "collided": collision,  # 0 or 1
        # })
        # -----------------------------------------

    def step(self):

        self.integrate()
        self.logger()
        self.time += self.dt

    def dataframe(self):

        return pd.DataFrame(self.rows)

    def run(self):

        self.apply_gravity()
        while self.time < self.max_time:
            self.step()



#class to create objects in this universe
class Body:
    def __init__(self, name, mass, radius, x, y, vx, vy, ax, ay):
        self.name = name
        self.mass = mass
        self.radius = radius

        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)
        self.acc = np.array([ax, ay], dtype=float)

    @property
    def speed(self):
        return np.linalg.norm(self.vel)

    @property
    def volume(self):
        return 4/3 * np.pi * self.radius**3

    @property
    def density(self):
        return self.mass / self.volume

    @property
    def momentum(self):
        return self.mass * self.vel

    @property
    def K_energy(self):
        return 0.5 * self.mass * np.sum(self.vel**2)


#5)Visual engine
def simulator_visual(universe, df):
    # plotter engine
    #Giving colours and rendering sizes of objects. Not real radius - only rendered.
    #Rendering is for temporary visual testing.
    colors = {"Star1": "yellow", "Planet1": "deepskyblue", "Moon1": "red" }
    render_size = {"Star1": 500, "Planet1": 4.57, "Moon1": 1.25}

    #change background to black
    plt.style.use("dark_background")
    #size of canvas
    fig, ax = plt.subplots(figsize=(10, 10))
    for body in universe.objects:

        #Centre the star. change this to see how planet and moon's gravity affects the star.
        star_x = df["Star1_x"]
        star_y = df["Star1_y"]

        #relative coordinates
        x = df[f"{body.name}_x"] - star_x
        y = df[f"{body.name}_y"] - star_y

        #orbit plotting part.
        #two arguments x and y for plotting
        ax.plot(
            x,
            y,
            color=colors[body.name], #Colour given for plot
            linewidth=1, #width of plot line
            alpha=0.7, #transparency. 0 = invisible, 1 = solid
            label=body.name #labels of bodies
        )

        ax.scatter(
            #iloc stores positions in a list. iloc[-1] is last recorded position
            x.iloc[-1],
            y.iloc[-1],
            color=colors[body.name],
            s=render_size[body.name],
            #zorder decides what object has higher priority on a layer.
            #if planet had more priority than star and both were kept at same position planet would appear above star.
            zorder=10
        )
    #title
    ax.set_title(
        f"N-Body Simulation\nBodies: {len(universe.objects)}",
        fontsize=14
    )

    stats = (
        f"dt = {universe.dt}\n"
        f"Time = {universe.max_time}\n"
        f"Steps = {len(df)}"
    )
    #adding text to the canvas
    ax.text(
        0.02,
        0.02,
        stats,
        #this makes the coordinates screen coordinates, not physical coordinates.
        #(0,0) is bottom left corner and (1,1) is top right corner.
        transform=ax.transAxes
    )

    #removes numbering from both x and y axes.
    ax.set_xticks([])
    ax.set_yticks([])

    #this part removes the borders
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #set_aspect makes units equal. Scaling is done properly. Otherwise orbit would be circular and not elliptical.
    ax.set_aspect("equal")
    #Legend builds the box with star1, planet1 and moon1
    ax.legend()
    #Display the grand plot!
    plt.show()


#Universe creator
universe = Universe(dt = 0.01, max_time = 16.0, G = 1.0, verbose = False)

#Object creator
#Introducing the objects to reality in the universe
universe.add_object(Body("Star1", 10000.0, 2.0, 40.0, 40.0, 0, 0, 0, 0))
universe.add_object(Body("Planet1", 100.0,0.5 , 20.0, 10.0, 17.6, 0, 0, 0))
universe.add_object(Body("Moon1", 1.0, 0.1, 21.0, 10.0, 17.6, 10, 0, 0))
#x_wall = 2.0 #simple wall

universe.run()

df = universe.dataframe()
print(df.head())

simulator_visual(universe, df)

#print(df[df["collided"] == 1])
