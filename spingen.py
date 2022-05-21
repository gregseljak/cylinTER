#%%
import logging
import time
from scipy.optimize import curve_fit
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

def binsearch(function, domain):
    ydomain = domain
    survey = function(ydomain)
    bigroots = np.where(survey[:-1]*survey[1:] <= 0)[0]
    if len(bigroots) == 0:
        #print(" No roots seen in survey ")
        return np.array([])
    roots = np.empty(len(bigroots))
    for i in range(len(roots)):
        ymin = ydomain[bigroots[i]]
        ymax = ydomain[bigroots[i]+1]
        yguess = (ymin+ymax)/2
        
        cnt,threshold = 0, 1000
        while (abs(ymax-ymin) > 0.001):
            yguess = (ymin+ymax)/2
            if function(yguess)*function(ymin) > 0:
                ymin = yguess
            if function(yguess)*function(ymax) > 0:
                ymax = yguess
            cnt += 1
            if cnt >= threshold:
                print(f"threshold {threshold} reached")
                break
        roots[i] = yguess
    return roots

def roots_to_parametrization(ragged_array):
    rgdarr = ragged_array
    nb_psipts = 0
    for j in range(len(rgdarr)):
        nb_psipts += len(rgdarr[j])
    stream = np.zeros(nb_psipts, dtype=complex)
    xidx = 0
    while len(rgdarr[xidx]) == 0:
        xidx += 1
        if xidx == len(rgdarr)-1:
            return np.array([], dtype=complex)
    arg = np.argmin(np.imag(rgdarr[xidx]))
    stream[0] = (rgdarr[xidx])[arg]
    rgdarr[xidx] = np.delete(rgdarr[xidx], arg)
    xidx += 1
    for j in range(nb_psipts-2):
        #print(f"x = {xidx} j = {j} nbp = {nb_psipts-2}")
        if len(rgdarr[xidx]) == 0:
            xidx = 0
            while len(rgdarr[xidx]) == 0:
                xidx += 1
        arg = np.argmin(np.abs(np.imag(stream[j]) - np.imag(rgdarr[xidx])))
        stream[j+1] = (rgdarr[xidx])[arg]
        rgdarr[xidx] = np.delete(rgdarr[xidx], arg)
        lenleft = 0
        lenright = 0
        if xidx > 0:
            lenleft = len(rgdarr[xidx-1])
        if xidx < len(rgdarr)-1:
            lenright = len(rgdarr[xidx+1])
        lenmiddle = len(rgdarr[xidx])
        dists = np.zeros(3) + 1000
        if lenleft > 0:
            dists[0] = np.min(np.abs(stream[j] - rgdarr[xidx-1]))
        if lenright > 0:
            dists[2] = np.min(np.abs(stream[j] - rgdarr[xidx+1]))
        if lenmiddle > 0:
            dists[1] = np.min(np.abs(stream[j] - rgdarr[xidx]))
        xidx += (np.argmin(dists)-1)
    return stream[:-1]

def sort_parametrization(usrinroots):
    seglist = []
    inroots = usrinroots
    if not isinstance(inroots[0], np.ndarray):
        inroots = np.array([usrinroots], dype = np.ndarray)
    for i in range(len(inroots)):
        sorted_roots = roots_to_parametrization(inroots[i])
        if len(sorted_roots) <= 1:
            continue
        dx = np.real(sorted_roots[1] - sorted_roots[0])
        segs = np.append(np.array([1]), np.where(np.abs(np.real(sorted_roots[1:]-sorted_roots[:-1]))>2*dx))
        segs = np.append(segs, np.array([-1]))
        for j in range(1, len(segs)):
            start, stop = segs[j-1]+1, segs[j]
            seglist.append( ((i,j), sorted_roots[start:stop]) )
    return seglist

class SpinGen():


    def __init__(self, gamma=0):
        self.equiphi = None
        self.streams=None
        self.v0 = 1
        self.gamma = gamma
        self.xintval = np.array([-6,6])
        self.res = 1000
        self.a = 1
        self.psi = np.linspace(-8,8,24)
        if 0 in self.psi:
            print(f" WARNING 0 in SpinGen.psi")
        basephi = np.linspace(0,self.gamma, 11)[:-1] - 3*self.gamma
        self.phi = basephi
        for vv in range(6):
            self.phi = np.append(self.phi, basephi+vv*self.gamma)
        self.dt = 0.001
        self.streams = np.empty((len(self.psi), self.res), dtype=complex)
        self.equiphi = np.empty((len(self.psi), self.res), dtype=complex)
        #self.populate_equilines()
        self.schema = "RK4" # or "FE" or "RE"
        self.schema_dict = {"RK4": "Runge-Kutta 4",
            "FE":"Forward-Euler", "RE":"Reverse-Euler"}
    
    ### utilities ###
    def _make_array(self, args):
        """ utility to make optional arguments more flexible """
        data = []
        for arg in args:
            if arg is None:
                data.append(None)
            elif (not isinstance(arg, np.ndarray)):
                data.append(np.ndarray(arg), dtype=complex)
            else:
                data.append(arg)
        return data

    def phifn(self, z):
        phi = self.v0*(1+self.a**2*np.real(z)/(np.abs(z)**2))
        phi -= self.gamma/(2*np.pi)*np.arctan2(np.imag(z), np.real(z))
    ### end of utilities ###

    def populate_equilines(self, plot=False):
        """ Find points along streams and equipotentials for plotting.
            Populates self.equiphi and self.equipsi with lists:
            self.equipsi = [(psi1,zsarray1), (psi2, zsarray2), ...]
            self.equiphi = [(phi1,zharray1), (phi2, zharray2), ...]"""
        streams = []
        equiphis = []
        goodequiphi = []
        x = np.linspace(self.xintval[0], self.xintval[1], 100)
        psisol = self.y_from_psix(self.psi, x)
        print("done psi")
        phisol = self.y_from_phix(self.phi, x)
        print("donephi")
        streams = sort_parametrization(psisol)
        equiphis = sort_parametrization(phisol)
        for i in range(len(streams)):
            psidx = ((streams[i])[0])[0]
            streams[i] = (self.psi[psidx], (streams[i])[1])
        for i in range(len(equiphis)):
            phidx, line = equiphis[i]
            #line = -1*np.conjugate(1j*line)
            phi = self.phi[phidx[0]]
            yidx = 0
            while len(line)>0:
                if yidx >= len(line)-1:
                    goodequiphi.append((phi, np.imag(line)+1j*np.real(line)))
                    line = []
                elif np.abs(np.imag(line[yidx]) - np.imag(line[yidx+1])) > 0.5:
                    #print(f" found a break at {phi}")
                    goodequiphi.append((phi, np.imag(line)[:yidx+1]+1j*np.real(line)[:yidx+1]))
                    line = line[yidx+1:]
                    yidx = 0
                else:
                    yidx += 1

        self.streams = streams
        self.equiphi = goodequiphi
        # graphing:
        if plot:
            print(f"psi: {len(self.psi)} unique seeds, {len(self.streams)} segments")
            print(f"phi:{len(self.phi)} unique seeds, {len(self.equiphi)} segments")
            fig, ax = plt.subplots(1)
            mymarkers=["o", "x", ".", "P", "+"]
            for stream in self.streams:
                label = str(stream[0])
                data = stream[1]
                ax.plot(np.real(data), np.imag(data), color="tab:blue")
            for i in range(len(self.equiphi)):
                stream = self.equiphi[i]
                label = str(np.round(stream[0], decimals = 3))
                data = stream[1]
                ax.plot(np.real(data), np.imag(data), label=label, marker=mymarkers[(i//10)//len(mymarkers)])
            ax.set_aspect(1)
            ax.set_xlim(-6,6)
            ax.set_ylim(-6,6)
            ax.legend(bbox_to_anchor=(1, 1))
            t = np.linspace(0,np.pi*2, 100)
            ax.plot(np.sin(t), np.cos(t), color="black")
            return fig,ax

    def _equipsi_fn(self,psi,x):

        def subordinate_equipsi(y):
            return psi - (self.v0*(y-self.a**2*y/(x**2+y**2) + self.gamma/(2*np.pi)*np.log(np.abs(x+1j*y))))   
        return subordinate_equipsi

    def y_from_psix(self, inpsi = None, xvals_in = None, innerstream=False):
        # Given a populated np.array of psi and x, find y
        npsi = self.psi
        xvals = np.linspace(self.xintval[0], self.xintval[1], 1000)
        uservals = self._make_array((inpsi, xvals_in))
        if uservals[0] is not None:
            npsi = uservals[0]
        if uservals[1] is not None:
            xvals = uservals[1]

        solution = np.empty((len(npsi), len(xvals)), dtype=np.ndarray)
        for i in range(len(npsi)):
            psi = npsi[i]
            for j in range(len(xvals)):
                x = xvals[j]
                equipsi = self._equipsi_fn(psi,x)
                roots = binsearch(equipsi, 1.5*np.linspace(-8, 8, 1000))
                if not innerstream:
                    roots = np.delete(roots, np.where(np.abs(x+1j*roots) <= self.a)[0])
                #if len(roots) == 0:
                    #print(f" y_from_psix problem at psi = {psi}, x = {x}")
                solution[i,j] = x + 1j*roots
        return solution
    
    def y_from_phix(self, inphi = None, xvals_in = None):
        """ subtle difference from y_from_psix: where the streamfunction is defined continuously,
            equiphilines have a jump discontinuity at theta=pi / theta = -pi.
            Left untreated, this gives a line of extraneous roots at y = 0 """
        nphi = self.phi
        xvals = np.linspace(self.xintval[0], self.xintval[1], 1000)
        uservals = self._make_array((inphi, xvals_in))
        if uservals[0] is not None:
            nphi = uservals[0]
        if uservals[1] is not None:
            xvals = uservals[1]

        xvals = xvals[np.where(xvals != 0)[0]]
        solution = np.empty((len(nphi), len(xvals)), dtype=np.ndarray)
        for i in range(len(nphi)):
            phi = nphi[i]
            for j in range(len(xvals)):
                x = xvals[j]
                equipsi = self._equiphi_fn(phi,x)
                #roots = binsearch(equipsi, np.linspace(0.001, 20, 500))
                #roots = np.append(roots, binsearch(equipsi, np.linspace(-20,-0.001, 500)))
                roots = binsearch(equipsi, np.linspace(self.xintval[0],self.xintval[-1], 1000))
                roots = np.delete(roots, np.where(np.abs(x+1j*roots) < self.a)[0])
                solution[i,j] = x + 1j*roots
        return solution

    def _equiphi_fn(self,phi,y):

        def subordinate_equiphi(x):
            return phi - (self.v0*(x+self.a**2*x/(x**2+y**2) - self.gamma/(2*np.pi)*np.arctan2(y, x)))   
        return subordinate_equiphi



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

################## Should be good from here #################################      
    def antisymmetrize(self, array):    #checked: good
        antiarray = -1*np.flip(array)
        return np.append(antiarray, array)

    def _particle_velocityfield(self, u):
        """takes complex-valued singleton numpy array
            returns d(phi)/dz evaluated at that"""
        velocity = 0 + 0j
        r = np.abs(u)
        velocity += self.v0*(1 + self.a**2/r**2 - 2*(self.a**2)*(np.real(u)**2)/(r**4))
        velocity += self.gamma/(2*np.pi)*(np.imag(u)/(r**2))
        velocity += -1j*self.v0/(r**4)*2*self.a*np.real(u)*np.imag(u)
        velocity += -1j*(self.gamma/2/np.pi)*np.real(u)/(r**2)
        return velocity

    def xintval_integration(self, mindx = 0.001):
        """ create a matrix of complex-valued coordinates that advance iteratively by time step"""
        dt = self.dt
        positions = None
        y0 = None
        static_castx = np.array([self.xintval[0]])  # unfortunately necessary
        y0 = self.y_from_psix(inpsi = self.psi, xvals_in=static_castx)
        #print(f" y0 = {y0}")
        positions = [ [] for _ in range(len(y0)) ]
        maxlength = 0
        for i in range(len(y0)):
            itercounter = 0
            
            u = (self.xintval[0] + 1j*y0[i])[0]
            (positions[i]).append(u)
            while (np.real(u) < self.xintval[1]):
                velocity = 0 + 0j
                itercounter += 1
                #if itercounter % 2500 == 0:
                #    print(f" particle {y0[i]} at {itercounter}: {np.real(u), np.imag(u)}")

                if self.schema == "FE":
                    velocity = self._particle_velocityfield(u)
                    
                
                elif self.schema == "RK4":
                    
                    k1 = self._particle_velocityfield(u)
                    k2 = self._particle_velocityfield(u + dt*k1/2)
                    k3 = self._particle_velocityfield(u + dt*k2/2)
                    k4 = self._particle_velocityfield(u + dt*k3)
                    velocity = (k1+2*k2+2*k3+k4)/6

                u += velocity*dt
                if np.abs(np.real((positions[i])[-1] - u)) > mindx:
                    # if this size of dt renders oversized dx
                    # reduce timestep size, and then reset the step
                    dt /= 2
                    u = (positions[i])[-1]
                else:
                    (positions[i]).append(u)

        maxlength = max(len(i) for i in positions)            
        pos_array = np.empty((len(y0), maxlength), dtype=complex)
        print(f" maxlen {maxlength}")
        for i in range(len(pos_array)):
            ulen = len(positions[i])
            pos_array[i,:ulen] = np.array(positions[i])
            pos_array[i,ulen:] = np.nan
        return pos_array


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
        self.schema = "RK4"

        rk4 = self.xintval_integration()
        validity_vec = np.argmin(np.isnan(rk4) == False, axis=1)
        psivals = self.v0*(1-self.a**2/ ((np.abs(rk4[:,0]))**2) )*np.imag(rk4[:,0])
        true_rk4 = np.zeros(rk4.shape, dtype=complex)
        yerrors = np.zeros(len(rk4))
        for i in range(len(rk4)):
            last_idx = 0
            if validity_vec[i] != 0:
                last_idx = validity_vec[i]
            else:
                last_idx = -1
            true_rk4[i,:last_idx] = 1j*self.y_from_psix(np.array([psivals[i]]), np.real(rk4[i,:last_idx]))
            true_rk4[i,:last_idx] += np.real(rk4[i,:last_idx])
            xmaillage = np.real(rk4[i,:last_idx]) - np.real(np.roll(rk4[i, :last_idx], shift=1))
            xmaillage[0] = (self.xintval[1] - self.xintval[0]) - np.sum(xmaillage[1:])
            yerr = xmaillage*np.abs(np.imag(true_rk4[i,:last_idx]) - np.imag(rk4[i,:last_idx]))
            yerrors[i] = np.sum(yerr)
        return yerrors


    def development_histograms(self, devtime, T1Density:int):
        # todo: particles leaving on diagonal escape the zbdry fix
        zbdry = 5+5j # keep it square
        nb_particles = 20000
        
        nb_steps = int(devtime/self.dt)
        fig, ax = plt.subplots(3)
        bins = np.arange(0,int(100),1, dtype=int)*2
        

        def uniform_distribution(arraylen:int):
            positions = np.zeros(arraylen, dtype=complex)
            badstarts = np.arange(0,len(positions))
            while len(badstarts) > 0:
                positions[badstarts] += np.random.uniform(-1*np.real(zbdry), np.real(zbdry), len(badstarts))
                positions[badstarts] += 1j*np.random.uniform(-1*np.imag(zbdry),np.imag(zbdry), len(badstarts))
                badstarts = np.where(np.abs(positions)<self.a)[0]
            return positions

        def identitybdry(z : np.ndarray):
            ybound = np.where(np.abs(np.imag(z)) > np.imag(zbdry))[0]
            xbound = np.where(np.abs(np.real(z)) > np.real(zbdry))[0]
            z[ybound] -= np.imag(z[ybound])/np.abs(np.imag(z[ybound]))*2*np.imag(zbdry)
            z[xbound] -= np.real(z[xbound])/np.abs(np.real(z[xbound]))*2*np.real(zbdry)
            return z
        
        linvals = np.linspace(-1*np.real(zbdry),1*np.real(zbdry), T1Density+1)[0:-1]
        linvals = linvals + (linvals[1]-linvals[0])/2
        def densitycnt(particles):
            linvals = np.linspace(-1*np.real(zbdry),1*np.real(zbdry), T1Density+1)[0:-1]
            rad = (linvals[1]-linvals[0])/2
            print(f"rad {rad}")
            linvals = linvals + rad
            tuilecnts = np.empty((len(linvals),len(linvals)),dtype=int)
            for i in range(len(tuilecnts)):
                for j in range(len(tuilecnts)):
                    BoolX = (np.abs(np.real(particles) - linvals[j]) < rad)
                    BoolY = (np.abs(np.imag(particles) - linvals[i]) < rad)
                    tuilecnts[i,j] = np.sum(BoolX*BoolY)
            return tuilecnts.flatten()
        
        positions = uniform_distribution(nb_particles)
        pbb0 = densitycnt(positions)
        pbb1 = None
        pbb2 = None
        #ax[0].scatter(np.real(positions[::10]), np.imag(positions[::10]))
        for step in range(nb_steps):
            positions += self.dt*self._particle_velocityfield(positions)
            positions = identitybdry(positions)
            if step == int(nb_steps/2):
                pbb1 = densitycnt(positions)
        #        ax[1].scatter(np.real(positions[::10]), np.imag(positions[::10]))
            if (step % (nb_steps//10) == 0):
                print(f"step {step} / {nb_steps}")
        pbb2 = densitycnt(positions)
        #ax[2].scatter(np.real(positions[::10]), np.imag(positions[::10]))
        scale = np.real(zbdry)
        #ax[0].scatter(linvals, linvals, color="gray")
        plt.show()

        return {"nbpts":nb_particles,"bins":bins ,"ppb":(pbb0,pbb1,pbb2), "devt" :devtime, "srad":T1Density}

    def show_histograms(self, histdic):
        # todo: least-squares poisson pmf not working anymore :(
        fig, axs = plt.subplots(len(histdic["ppb"]))
        def poi_pmf(x,mu):
            return poisson.pmf(x,mu)
        
        initialfit = None
        for i in range(len(axs)):
            nbpts = histdic["nbpts"]
            ppb = (histdic["ppb"])[i]
            vals, bins, patches = axs[i].hist(ppb, bins = histdic["bins"], density=True)
            print(f"np.sum(vals) {np.sum(vals)}")
            binmid = bins[:-1] + 1#(bins[1] -bins[0])/2
            #axs[i].plot(binmid, vals)
            vals *= 2
            retval = curve_fit(poi_pmf, binmid, vals)
            print(f"retval[0] {retval[0]}")
            retval = (12.5, None)
            axs[i].plot(binmid, poi_pmf(binmid,retval[0]), color="gray", linestyle="dashed", label=f"$\lambda = {retval[0]}$")
            if i == 0:
                initialfit = poi_pmf(histdic["bins"],retval[0])
            else:
                axs[i].plot(histdic["bins"], initialfit, color="black")

        title =f"""T=0, T={int(histdic["devt"]*50)/100}, T={int(histdic["devt"]*100)/100}"""
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()








def main():
    
    flow = SpinGen(gamma=10)
    histogram = flow.show_histograms(flow.development_histograms(10,40))

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description=("Flowgen"))
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


flow = None
def showpoints_bydensitycnt(histogram, specialvals):
    return None
    
    """
    ### miscreated function
    ### returns histogram of points by local density (ie. cluster)
    def densitycnt(positions):
            densitycnt = np.empty(len(positions), dtype=int)
            pointsbybin = [[] for _ in range(len(bins)) ]
            for i in range(len(positions)):
                p0 = positions[i]
                distances = identitybdry(positions - p0)
                densitycnt[i] = int(np.sum(np.abs(distances) < sradius) - 1)
                if np.abs(p0) < (self.a + 2*sradius):
                    pass #densitycnt[i] /= (1-intersectingarea/np.pi*sradius**2) # it would be nice to get a function here
                densitycnt[i] = int(np.round(densitycnt[i], decimals=0))
                if densitycnt[i] < len(pointsbybin):
                    pointsbybin[densitycnt[i]].append(p0)
                else:
                    pointsbybin[-1].append(p0)
            return pointsbybin"""
