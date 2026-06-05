#information
#Phase 2A:Orbital mechanics: successful. Newtonian mechanics for two bodies in space worked.
#Phase 2B:N-body simulation: IN OPERATION.


import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
rows = []

#essential constants
DT = 0.01
MAX_TIME = 1000 .0
current_time = 0.0
e = 1.0 #Elasticity
collision = 0
G = 1  #currently experimental, so G taken as 1. #unibersal gravitational constant = 6.6743e-11
objects = []

#class to create objects in this universe
class Uobjects:
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

#methods for objects

def total_energy(obj):
    PE = 0

    for a in range(len(obj)):
        for b in range(a + 1, len(obj)):
            r_vec = obj[b].pos - obj[a].pos
            r = np.linalg.norm(r_vec)

            PE += -(G * obj[a].mass * obj[b].mass) / r

    KE = sum(o.K_energy for o in obj)

    E = KE + PE
    print("Total Energy = ", E)

def total_momentum(obj):
    p = np.array([0.0, 0.0])
    for o in obj:
        p += o.momentum
    print("Momentum:", p)


#1) New updated N-body gravity engine
def apply_gravity(obj):

    for i in obj:
        i.acc[:] = 0

    for i in obj:
        for j in obj:
            if i == j:
               continue

            r_vec = j.pos - i.pos
            r = np.linalg.norm(r_vec)
            if r == 0:
                continue

            i.acc += (G * j.mass * r_vec / r ** 3)

    #-----Old 2-body simulation code.Saved for reference-----
    #direction = r_vec / r
    #G_force = (G * obj[0].mass * obj[1].mass) / r ** 2
    #a1 = G_force / obj[0].mass
    #a2 = G_force / obj[1].mass
    #obj[0].acc = a1 * direction
    #obj[1].acc = -a2 * direction
    #--------------------------------------------------------

#2)Velocity and position handler. Currently, handles only two objects.
def integrate(obj, dt):
    #Euler integration
    # 1. update velocities (if any accel)
    #obj[0].vel += obj[0].acc * dt
    #obj[1].vel += obj[1].acc * dt
    # problematic code here
    # 2. update positions
    #obj[0].pos += obj[0].vel * dt
    #obj[1].pos += obj[1].vel * dt

    #NEW UPDATE VERLET INTEGRATION
    #Creates a list of old accelerations of all objects present
    old_acc = [o.acc.copy() for o in obj]

    for o, a in zip(obj, old_acc):
        o.pos += o.vel * dt + 0.5 * a * dt * dt

    apply_gravity(obj)

    for o, a in zip(obj, old_acc):
        o.vel += 0.5 * (a + o.acc) * dt


#3)Collision handler. Currently handles only two objects.
def resolve_collision(obj):
    b1 = obj[0]
    b2 = obj[1]
    relative_pos = b1.pos - b2.pos
    distance = np.linalg.norm(relative_pos)
    if distance <= (b1.radius + b2.radius):
        normal = relative_pos/distance
        relative_vel = b1.vel - b2.vel
        #Velocity Along Normal
        van = np.dot(relative_vel, normal)
        if van > 0:
            return False

    if abs(b1.pos[0] - b2.pos[0]) <= (b1.radius + b2.radius):
        collision = 1
        print("Collision!")
        # (a) snap them so they are just touching, centered around midpoint
        mid = 0.5 * (b1.pos[0] + b2.pos[0])
        b1.pos[0] = mid - b1.radius
        b2.pos[0] = mid + b2.radius
        # (b) swap their x-velocities (equal masses, 1D, elastic)
        u1 = b1.vel[0]
        u2 = b2.vel[0]
        m1 = b1.mass
        m2 = b2.mass
        den = m1 + m2
        v1 = ((m1 - m2) / den) * u1 + ((2 * m2) / den) * u2
        v2 = ((2 * m1) / den) * u1 + ((m2 - m1) / den) * u2
        b1.vel[0] = v1
        b2.vel[0] = v2

        # handle collision here: snap positions, swap velocities, etc.

#4)Data logger
def logger(obj):

    for o in obj:
        print(
            f" {o.name} t={current_time:.2f}s  "
            f"pos = ({o.pos[0]:.2f}, {o.pos[1]:.2f})  "
            f"vel = ({o.vel[0]:.2f}, {o.vel[1]:.2f})  "
            f"p = {o.momentum}  "
            f"KE = {o.K_energy:.2f}"
            f"Acc = {o.acc}"
        )

    snapshot = {"t": current_time}

    for body in obj:
       snapshot[f"{body.name}_x"] = body.pos[0]
       snapshot[f"{body.name}_y"] = body.pos[1]

    rows.append(snapshot)

    #-----Old dataframe design. Use later-----
    #rows.append({
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
    #})
    #-----------------------------------------

#Universe creator
Star =   Uobjects("Star1", 10000.0, 2.0, 40.0, 40.0, 0, 0, 0, 0)
Planet = Uobjects("Planet1", 100.0,0.5 , 20.0, 10.0, 17.6, 0, 0, 0)
Moon =   Uobjects("Moon1", 1.0, 0.1, 21.0, 10.0, 17.6, 10, 0, 0)

objects.append(Star)
objects.append(Planet)
objects.append(Moon)
#x_wall = 2.0 #simple wall

apply_gravity(objects)
for o in objects:
    print(o.name, o.acc)
print("Acceleration magnitude:", np.linalg.norm(o.acc))
while current_time < MAX_TIME:
    integrate(objects, DT)
    #   resolve_collision(objects)
    logger(objects)
    total_energy(objects)
    total_momentum(objects)
    current_time = current_time + DT


df = pd.DataFrame(rows)
print(df.head())

#plotter engine
for body in objects:
    plt.plot(
        df[f"{body.name}_x"],
        df[f"{body.name}_y"],
        label=body.name
    )
plt.axis("equal")
plt.legend()
plt.show()

#print(df[df["collided"] == 1])

