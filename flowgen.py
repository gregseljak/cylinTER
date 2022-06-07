#%%
import logging
import numpy as np
class FlowGen():


    def __init__(self):
        self.v0 = 1
        self.xintval = np.array([-6,6])
        self.res = 1000
        self.a = 1
        self.radlen = int(self.res*np.sqrt(2))
        self.psi = self.psi_from_v(np.linspace(1.0001, 1.99, 20))
        #print(f" psi vals{self.psi}")
        self.phi = np.linspace(0,1,5)
        self.dt = 0.001
        self.streams = np.empty((len(self.psi), self.res), dtype=complex)
        self.equiphi = np.empty((len(self.psi), self.radlen), dtype=complex)
        self.populate_equilines()
        self.schema = "RK4" # or "FE" or "RE"
        self.schema_dict = {"RK4": "Runge-Kutta 4",
            "FE":"Forward-Euler", "RE":"Reverse-Euler"}
    

    def populate_equilines(self):
        """ Find points along streams and equipotentials for plotting"""
        phi = np.linspace(-20,20,len(self.psi))
        x = np.linspace(self.xintval[0], self.xintval[1], 1000)
        y = self.y_from_psix(self.psi, x)
        self.streams = x + 1j*y

        for i in range(len(self.psi)):
            radii = np.linspace(self.a+ 0.001, 15, self.radlen, dtype=np.double)
            plotpoints = np.empty(len(radii))
            theta = np.arccos(phi[i]/self.v0*radii/(radii**2 + 1))
            plotpoints = radii * np.cos(theta) + 1j* radii * np.sin(theta)
            plotpoints = plotpoints[np.argsort(np.real(plotpoints))] # sort by real part
            self.equiphi[i] = plotpoints
        print(f" streams.shape {self.streams.shape}")
        print(f" equiphi {self.equiphi.shape}")

    def psi_from_v(self, velocity): #checked: good
        """ Recover value of psi given a velocity;
        allows for equal spacing of streamlines to indicate
        velocity gradient """
        y0 = self.a*np.sqrt(self.v0/(velocity-self.v0))
        psi = self.v0 * (1-(self.v0/(velocity-self.v0)))*np.sqrt(self.v0/(velocity-self.v0))*self.a
        for xintcp in y0:
            if (xintcp < self.a):
                continue
        return self.antisymmetrize(psi)
    
    def antisymmetrize(self, array):    #checked: good
        antiarray = -1*np.flip(array)
        return np.append(antiarray, array)

    def _particle_velocityfield(self, u):
        """takes complex-valued singleton numpy array
            returns d(phi)/dz evaluated at that"""
        velocity = 0 + 0j
        r = np.abs(u)
        velocity += self.v0*(1 + self.a**2/r**2 - 2*(self.a**2)*(np.real(u)**2)/(r**4))
        velocity += -1j*self.v0/(r**4)*2*self.a*np.real(u)*np.imag(u)
        return velocity
    
    def psi(self, z):
        r = np.abs()
        return self.v0 - self.v0*self.a**2/(r**2)

    def xintval_integration(self, mindx = 0.001):
        """ create a matrix of complex-valued coordinates that advance iteratively by time step"""
        dt = self.dt
        positions = None
        y0 = np.linspace(-6,6,10)
        static_castx = np.array([self.xintval[0]])  # unfortunately necessary

        

        #print(f" y0 = {y0}")
        positions = [ [] for _ in range(len(y0)) ]
        maxlength = 0
        for i in range(len(y0)):
            itercounter = 0
            
            u = (self.xintval[0] + 1j*y0[i])
            (positions[i]).append(u)
            while (np.real(u) < self.xintval[1]):
                velocity = 0 + 0j
                itercounter += 1
                #if itercounter % 2500 == 0:
                    #print(f" particle {y0[i]} at {itercounter}: {np.real(u), np.imag(u)}")

                if self.schema == "FE":
                    velocity = self._particle_velocityfield(u)
                    
                
                elif self.schema == "RK4":
                    
                    k1 = self._particle_velocityfield(u)
                    k2 = self._particle_velocityfield(u + dt*k1/2)
                    k3 = self._particle_velocityfield(u + dt*k2/2)
                    k4 = self._particle_velocityfield(u + dt*k3)
                    velocity = (k1+2*k2+2*k3+k4)/6
                if (np.real(velocity) < 0):
                    logger.warning(f" BAD VELOCITY AT y0 = {y0}, step {itercounter}")
                    quit()
                u += velocity*dt
                """if np.abs(np.real((positions[i])[-1] - u)) > mindx:
                    # if this size of dt renders oversized dx
                    # reduce timestep size, and then reset the step
                    dt /= 2
                    u = (positions[i])[-1]"""
                    #else:
                (positions[i]).append(u)

        return positions

    def y_from_psix(self, inpsi = None, xvals_in = None, innerstream=None):
        # Given a populated np.array of psi and x, find y
        npsi = None
        xvals = None
        if inpsi is not None:
            npsi = inpsi
            if not isinstance(inpsi, np.ndarray):
                inpsi = np.array([inpsi])
        else:
            npsi = self.psi
        if xvals_in is not None:
            xvals = xvals_in
        else:
            xvals = np.linspace(self.xintval[0], self.xintval[1], 1000)

        solution = np.zeros((len(npsi), len(xvals)))

        a = 1
        
        for i in range(len(npsi)):
            b = npsi[i]/(-1*self.v0)

            for j in range(len(xvals)):
                x = xvals[j]
                c = (x**2-self.a**2)
                d = b*(x**2)
                roots = np.roots([a,b,c,d])
                """ Alors on aura 3 racines. S'il n'y a qu'une qui
                est réelle on a fini. Sinon il faut appliquer un peu
                de logique...
                """
                if np.sum(np.imag(roots) == 0) == 1:
                    # if only one real, choose by lowest imag value
                    if not innerstream:
                        solution[i,j] = roots[np.argmin(np.abs(np.imag(roots)))]
                    else:
                        solution[i,j] = np.nan
                elif (j > 0):
                    # if multiple candidates or no candidates, decide by magnitude
                    if not innerstream:
                        solution[i,j] = np.real(roots[np.argmax(np.abs(np.real(roots)))])

                else:
                    solution[i,j] = np.real(roots[np.argmin(np.abs(np.imag(roots)/np.real(roots)))])
                    logger.warning(f"""no obvious choice found for 
                    {npsi[i]}, {x}. Consider reordering or extending x_in
                    Resorting to best guess {solution[i,j]}; check fpt error""")
                
        return solution

    def evaluate_xerror(self):
        """
        1. calculer la trajectoires des particules avec le schema choisi

        Puis, pour chaque trajectoire:
        1. on calcule psi
        2. on isole les valeurs reelles (x)
        3. on retrouve la trajectoire analytique avec .y_from_psix(psi, x)
            -- exactement la solution voulue
        4. Pour trouver l'integrale on fait la partition de la trajectoire
            comme elle se présente directement
        """
        #self.schema = "RK4"

        schemasol = self.xintval_integration()
        yerror = np.zeros(len(schemasol))
        for i in range(len(schemasol)):
            SSvec = np.array(schemasol[i]).flatten()
            psi = self.v0*(1-self.a**2/ ((np.abs(SSvec[0]))**2) )*np.imag(SSvec[0])
            trueline = 1j*self.y_from_psix([psi], np.real(SSvec))[0]
            trueline += np.real(SSvec)
            xmaillage = np.real(SSvec)[1:] - np.real(SSvec)[:-1]
            line_differences = (np.imag(SSvec - trueline)**2)[1:]
            scaled = line_differences*xmaillage
            yerror[i] = np.sum(scaled)
        return yerror



flow = FlowGen()
#%%
import pandas as pd

DT = np.array([0.25, 0.1, 0.05, 0.025])
df = pd.DataFrame(data=np.zeros((4,2)), columns=["FE", "RK4"],index=DT)
for schema in ["FE", "RK4"]:
    for dt in DT:
        print(f" {schema}, {dt}")
        flow.dt = dt
        flow.schema = schema
        result = np.mean(np.abs(flow.evaluate_xerror()))
        print(result)
        df[schema].loc[dt] = result
        print("-"*30)
        print("\n")
#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
ax.loglog(DT, np.sqrt(df["FE"]),  label="FE",  marker="x")
ax.loglog(DT, np.sqrt(df["RK4"]), label="RK4", marker="x")
ax.loglog(DT, 0.1*DT, color="gray", label="$\delta t$")
ax.loglog(DT, 0.001*DT**4, color="gray", linestyle="dashed", label="$\delta t^2$")
ax.legend()
fig.suptitle("Erreur en RMS vs dt")
plt.show()
#%%
"""
def main():
    flow = FlowGen()
    flow.xintval = np.array([-6,6])
    
    flow.dt = 0.1
    flow.schema = "FE"
    print(flow.evaluate_xerror())
    return flow

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=("Lorentzian generator"))
    parser.add_argument("-v", type=int, default=3,
                        help=("set logging level: 0 critical, 1 error, "
                              "2 warning, 3 info, 4 debug, default info"))
    
    args = parser.parse_args()
    logging_translate = [logging.CRITICAL, logging.ERROR, logging.WARNING,
                         logging.INFO, logging.DEBUG]
    logging.basicConfig(level=logging_translate[args.v],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
    """