from math import dist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import RK45

# Define the satellite class
class Satellite:
    g = 9.81 # m/s^2

    # Define the constructor
    def __init__(self, mass, inertia, thrusterDist,thrust = 100.0):
        """
        mass: mass of the satellite in kg
        inertia: moment of inertia of the satellite in kg*m^2
        com: distance from the center of mass to the center of rotation in m
        
        """
        self.mass = mass
        self.inertia = inertia
        self.dist = thrusterDist # Distance between center of mass and the thruster application point
        self.stVec = np.zeros((1, 6)) # State vector [x, y, theta, xdot, ydot, thetadot]
        self.t = np.zeros(1)
        self.eq = None # equations of motion

        # Thruster control
        
        self.cmd = np.zeros(4) # Thruster commands [Fa, Fb, Fc, Fd]
        self.cmdTmp = np.zeros(4)
        self.thrust = thrust
        self.thrustSeq = np.zeros((1, 5)) # Thrust on/off sequence of the 4 thrusters: time, Fa, Fb, Fc, Fd
        self.tfAll = self.thrustSeq[:,0] # Time sequence for the thruster commands
        self.nCurrent =  0 # Thruster sequence index
        self.nNext = 1 # Next thruster sequence index
        self.tfCurrent = self.tfAll[self.nCurrent] # Current thruster sequence time
        if len(self.tfAll) > 1:
            self.tfNext = self.tfAll[self.nNext] # Next thruster sequence time
        else:
            self.tfNext = self.tfCurrent + 1.0 # If only one command, set next time to current time + 1s
        
        # Control parameters
        self.wpt = np.zeros(2) # waypoint
        self.pos = np.zeros(3) # position
        self.vel = np.zeros(3) # velocity
        self.controlMode = 'static' # static or dynamic
        self.maxForce = 5.0 # N
        self.minForce = 0.1 # N
        self.maxAngle = np.pi*45/180 # rad

        # Gains for PID controller static position
        self.kpx = 1.0
        self.kdx = 1.0
        self.kix = 1.0
        self.kpy = 1.0
        self.kdy = 1.0
        self.kiy = 1.0
        self.kptheta = 1.0
        self.kdtheta = 1.0
        self.kitheta = 1.0

        # Gains for PID controller dynamic position
        self.kpxdot = 1.0
        self.kdxdot = 1.0
        self.kixdot = 1.0
        self.kpydot = 1.0
        self.kdydot = 1.0
        self.kiydot = 1.0
        self.kpthetadot = 1.0
        self.kdthetadot = 1.0
        self.kithetadot = 1.0

        # Plot parameters and variables
        self.generateSatGeometry()

    def __str__(self):
        """
        Returns a string representation of the satellite
        """
        return "satellite with mass {} kg, inertia {} kg*m^2, and thruster distance {} m from center of mass".format(self.mass, self.inertia, self.dist)

    def setPhysics(self, g):
        """
        Sets the physics of the satellite
        args:
            g: acceleration due to gravity in m/s^2
        """
        self.g = g

    def setConditions(self, x0, y0, theta0, xdot0, ydot0, thetadot0):
        """
        Sets the initial conditions of the satellite
        
        args:
            x0: initial x position in m
            y0: initial y position in m
            theta0: initial angle in radians
            xdot0: initial x velocity in m/s
            ydot0: initial y velocity in m/s
            thetadot0: initial angular velocity in rad/s
        """
        self.stVec[-1, :] = [x0, y0, theta0, xdot0, ydot0, thetadot0]

    def setWaypoint(self, x, y):
        """
        Sets the current waypoint of the satellite
        args:
            x: x position in m
            y: y position in m
        """
        self.wpt = np.array([x, y])

    def setPos(self, x, y, theta):
        """
        Sets the objective position of the satellite
        args:
            x: x position in m
            y: y position in m
        """
        self.pos = np.array([x, y, theta])
    
    def setVel(self, vx, vy, omega):
        """
        Sets the objective velocity of the satellite
        args:
            vx: x velocity in m/s
            vy: y velocity in m/s
        """
        self.vel = np.array([vx, vy, omega])

    def setControlMode(self, mode):
        """
        Sets the control mode of the satellite
        args:
            mode: control mode
        """
        if mode == 'static' or mode == 'dynamic':
            self.controlMode = mode
        else:
            print('Invalid control mode, mode set to {}'.format(self.controlMode))
    
    def setGains(self, gainDict):
        """
        Sets the gains for the PID controller
        args:
            gainDict: dictionary of gains
        """
        self.gainDict = gainDict
        self.kpx = gainDict['Kp_x']
        self.kdx = gainDict['Kd_x']
        self.kix = gainDict['Ki_x']
        self.kpy = gainDict['Kp_y']
        self.kdy = gainDict['Kd_y']
        self.kiy = gainDict['Ki_y']
        self.kptheta = gainDict['Kp_theta']
        self.kdtheta = gainDict['Kd_theta']
        self.kitheta = gainDict['Ki_theta']
        self.kpxdot = gainDict['Kp_xdot']
        self.kdxdot = gainDict['Kd_xdot']
        self.kixdot = gainDict['Ki_xdot']
        self.kpydot = gainDict['Kp_ydot']
        self.kdydot = gainDict['Kd_ydot']
        self.kiydot = gainDict['Ki_ydot']
        self.kpthetadot = gainDict['Kp_thetadot']
        self.kdthetadot = gainDict['Kd_thetadot']
        self.kithetadot = gainDict['Ki_thetadot']
    
    def setThrustSequence(self, thrustSeq):
        """
        Sets the thrust sequence for the thrusters
        args:
            thrustSeq: thrust sequence of the 4 thrusters: time, Fa, Fb, Fc, Fd
        """
        self.thrustSeq = thrustSeq
        self.tfAll = self.thrustSeq[:,0]
        print('Thrust sequence set. Time sequence: {}'.format(self.tfAll))

    def rotMatrix(self,y):
        """
        Returns the rotation matrix for the angle of the state vector
        args:
            None 
        returns:
            R: rotation matrix
        """
        self.R = np.array([[np.cos(y[2]), -np.sin(y[2])],
                      [np.sin(y[2]), np.cos(y[2])]])
        return self.R
    
    def udpateRotMatrix(self):
        """
        Updates the rotation matrix for the angle of the state vector
        args:
            None
        returns:
            None
        """        
        self.R = np.array([[np.cos(self.stVec[-1,2]), -np.sin(self.stVec[-1,2])],
                           [np.sin(self.stVec[-1,2]),  np.cos(self.stVec[-1,2])]])
        return self.R

    def control(self, t, y):
        """
        Sets the control inputs for the thruster of the satellite in case of a PID controller
        args:
            t: time in s
            y: state vector
        """
        
        if (t > self.tfNext) & (t <= self.tfAll[-1]):
            print(t < self.tfNext)
            print('t {}, nCurrent: {}, nNext: {}, tfCurrent: {}, tfNext: {}'.format(t,self.nCurrent, self.nNext, self.tfCurrent, self.tfNext))
            self.nCurrent += 1
            self.nNext += 1
            self.tfCurrent = self.tfAll[self.nCurrent]
            self.tfNext = self.tfAll[self.nNext]
        self.cmdTmp = self.thrustSeq[(self.thrustSeq[:,0] > self.tfCurrent), :][0,1:]

    def eqGenerator(self):
        """
        Generates the equations of motion for the satellite
        
        Args:
            None
        Returns:
            None
        """
        def eq(t, y):
            """
            Returns the equations of motion for the satellite
            args:
                t: time in s
                y: state vector
            returns:
                ydot: derivative of the state vector
            """

            # y = [x, y, theta, xdot, ydot, thetadot]
            # ydot = [xdot, ydot, thetadot, xddot, yddot, thetaddot]
            #self.pid(t, y)
            self.control(t, y)
            ydot = np.zeros(6)
            ydot[0] = y[3]
            ydot[1] = y[4]
            ydot[2] = y[5]
            ydot[3] = -self.thrust*np.sin(self.stVec[-1,2])*(self.cmdTmp[0]+self.cmdTmp[1]-(self.cmdTmp[2]+self.cmdTmp[3]))/self.mass
            ydot[4] = self.thrust*np.cos(self.stVec[-1,2])*(self.cmdTmp[0]+self.cmdTmp[1]-(self.cmdTmp[2]+self.cmdTmp[3]))/self.mass
            ydot[5] = self.thrust*self.dist*(self.cmdTmp[0]+self.cmdTmp[2]-(self.cmdTmp[1]+self.cmdTmp[3]))/self.mass
            return ydot
        self.eq = eq
    
    def updateState(self, t, y):
        """
        Updates the state vector and time vector
        args:
            t: time in s
            y: state vector
        """
        self.stVec = np.vstack((self.stVec, y))
        self.cmd = np.vstack((self.cmd, self.cmdTmp))
        self.t = np.append(self.t, t)
        self.udpateRotMatrix()
    
    def plot(self):
        """
        Plots the state vector
        args:
            None
        """
        fig, axs = plt.subplots(4, 2)
        # Make the figure large
        fig.set_size_inches(18.5, 10.5)

        # Separate the plots
        fig.tight_layout(pad=3.0)

        axs[0, 0].plot(self.t, self.stVec[:, 0],'r')
        axs[0, 0].set_title('x')
        axs[0, 0].set_ylabel('Position (m)')
        axs[0, 0].set_xlabel('Time (s)')

        axs[0, 1].plot(self.t, self.stVec[:, 3],'r')
        axs[0, 1].set_title('xdot (m/s)')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Velocity (m/s)')

        axs[1, 0].plot(self.t, self.stVec[:, 1],'g')
        axs[1, 0].set_title('y (m)')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Position (m)')

        axs[1, 1].plot(self.t, self.stVec[:, 4],'g')
        axs[1, 1].set_title('ydot (m/s)')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Velocity (m/s)')

        axs[2, 0].plot(self.t, self.stVec[:, 2]*180/np.pi,'b')
        axs[2, 0].set_title('theta (deg)')
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylabel('Angle (deg)')

        axs[2, 1].plot(self.t, self.stVec[:, 5]*180/np.pi,'b')
        axs[2, 1].set_title('thetadot (deg/s)')
        axs[2, 1].set_xlabel('Time (s)')
        axs[2, 1].set_ylabel('Angular Velocity (deg/s)')

        axs[3, 0].plot(self.t, self.cmd[:, 0], label='Fa')
        axs[3, 0].plot(self.t, self.cmd[:, 1], label='Fb')
        axs[3, 0].plot(self.t, self.cmd[:, 2], label='Fc')
        axs[3, 0].plot(self.t, self.cmd[:, 3], label='Fd')
        axs[3, 0].set_title('Thruster Commands')
        axs[3, 0].set_xlabel('Time (s)')
        axs[3, 0].set_ylabel('Thrust (N)')
        axs[3, 0].legend() 

        plt.show()

    def plotControl(self):
        """
        Plots the control  Fa, Fb, Fc and Fd on the same graph
        args:
            None
        """
        fig, ax = plt.subplots()
        ax.plot(self.t, self.cmd[:, 0], label='Fa')
        ax.plot(self.t, self.cmd[:, 1], label='Fb')
        ax.plot(self.t, self.cmd[:, 2], label='Fc')
        ax.plot(self.t, self.cmd[:, 3], label='Fd')
        ax.set_title('Thruster Commands')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Thrust (N)')
        ax.legend() 

        plt.show()

    def generateSatGeometry(self,xpos=0., ypos=0., thpos=0.,):
        """
        Generates the geometry of the satellite for plotting
        args:
            xpos: x position of the satellite in m
            ypos: y position of the satellite in m
            thpos: angle of the satellite in radians
        returns:
            satSquare: geometry of the satellite body
            tGeom: geometry of the thrusters
            plume: geometry of the thruster plumes
            ta, tb, tc, td: geometry of the thrusters in the body frame
            pa, pb, pc, pd: geometry of the thruster plumes in the body frame
        """
        def offsetPos(geom,xoff, yoff, thoff):
            c = np.cos(thoff)
            s = np.sin(thoff)
            geomTmp = np.dot(np.array([[c, -s],[s,  c]]), geom.T).T
            geomTmp  = geomTmp + xoff*np.array([np.ones(geom.shape[0]), np.zeros(geom.shape[0])]).T + yoff*np.array([np.zeros(geom.shape[0]), np.ones(geom.shape[0])]).T
            return geomTmp
        
        self.satSquare = np.array([[self.dist, self.dist],
                                   [self.dist, -self.dist],
                                   [-self.dist, -self.dist],
                                   [-self.dist, self.dist],
                                   [self.dist, self.dist]])
        self.satSquare = offsetPos(self.satSquare, xpos, ypos, thpos)
        
        self.tGeom = .2*np.array([[0, 0],
                                  [ .5*self.dist, self.dist],
                                  [-.5*self.dist, self.dist],
                                  [       0,    0]])

        self.plume = .2*np.array([[.5*self.dist, self.dist],
                                  [0, 3*self.dist],
                                  [-.5*self.dist, self.dist],
                                  [.5*self.dist, self.dist]])

        self.ta = -self.tGeom + np.array([[self.dist, -self.dist],
                                          [self.dist, -self.dist],
                                          [self.dist, -self.dist],
                                          [self.dist, -self.dist]])
        self.ta = offsetPos(self.ta, xpos, ypos, thpos)
        
        self.tb = -self.tGeom + np.array([[-self.dist, -self.dist],
                                          [-self.dist, -self.dist],
                                          [-self.dist, -self.dist],
                                          [-self.dist, -self.dist]])
        self.tb = offsetPos(self.tb, xpos, ypos, thpos)
        
        self.tc =  self.tGeom + np.array([[-self.dist, self.dist],
                                          [-self.dist, self.dist],
                                          [-self.dist, self.dist],
                                          [-self.dist, self.dist]])
        self.tc = offsetPos(self.tc, xpos, ypos, thpos)
        
        self.td =  self.tGeom + np.array([[self.dist, self.dist],
                                          [self.dist, self.dist],
                                          [self.dist, self.dist],
                                          [self.dist, self.dist]])
        self.td = offsetPos(self.td, xpos, ypos, thpos)
        
        self.pa =-self.plume + np.array([[self.dist, -self.dist],
                                         [self.dist, -self.dist],
                                         [self.dist, -self.dist],
                                         [self.dist, -self.dist]])
        self.pa = offsetPos(self.pa, xpos, ypos, thpos)
        
        self.pb =-self.plume + np.array([[-self.dist, -self.dist],
                                         [-self.dist, -self.dist],
                                         [-self.dist, -self.dist],
                                         [-self.dist, -self.dist]])
        self.pb = offsetPos(self.pb, xpos, ypos, thpos)
        
        self.pc = self.plume + np.array([[-self.dist, self.dist],
                                         [-self.dist, self.dist],
                                         [-self.dist, self.dist],
                                         [-self.dist, self.dist]])
        self.pc = offsetPos(self.pc, xpos, ypos, thpos)
        
        self.pd = self.plume + np.array([[self.dist, self.dist],
                                         [self.dist, self.dist],
                                         [self.dist, self.dist],
                                         [self.dist, self.dist]])
        self.pd = offsetPos(self.pd, xpos, ypos, thpos)
        return self.satSquare, self.tGeom, self.plume, self.ta, self.tb, self.tc, self.td, self.pa, self.pb, self.pc, self.pd

    def plot2D(self,skipFrames=1):
        """
        Plots the satellite in 2D
        args:
            None
        """
        fig = plt.figure()
        ax = plt.axes(xlim=(-0.1, 5), ylim=(-2*self.dist, 2*self.dist))
        ax.set_aspect('equal')
        ax.grid(True)
        for indFrame in range(len(self.t)):
            if indFrame % skipFrames != 0:
                self.generateSatGeometry(self.stVec[indFrame, 0], self.stVec[indFrame, 1], self.stVec[indFrame, 2])
                ax.plot(self.satSquare[:,0], self.satSquare[:,1], 'k-')
                ax.plot(self.ta[:,0], self.ta[:,1], 'r-')
                ax.plot(self.tb[:,0], self.tb[:,1], 'g-')
                ax.plot(self.tc[:,0], self.tc[:,1], 'b-')
                ax.plot(self.td[:,0], self.td[:,1], 'm-')
                if self.cmd[indFrame, 0] > 0:
                    ax.plot(self.pa[:,0], self.pa[:,1], 'r-')
                if self.cmd[indFrame, 1] > 0:
                    ax.plot(self.pb[:,0], self.pb[:,1], 'g-')
                if self.cmd[indFrame, 2] > 0:
                    ax.plot(self.pc[:,0], self.pc[:,1], 'b-')
                if self.cmd[indFrame, 3] > 0:
                    ax.plot(self.pd[:,0], self.pd[:,1], 'm-')
                ax.legend()
        plt.show()

    def animate(self):
        """
        Animates the satellite  in 2D
        args:
            None
        """

        fig, ax = plt.subplots()#figure()
        # ax = plt.axes(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
        ax.grid(True)
        ax.set_aspect('equal')
        
        # Generate empty lines for the satellite, thrusters and plumes
        satLine, = ax.plot([], [], lw=2, color='black')
        thaLine, = ax.plot([], [], lw=1, color='black')
        thbLine, = ax.plot([], [], lw=1, color='black')
        thcLine, = ax.plot([], [], lw=1, color='black')
        thdLine, = ax.plot([], [], lw=1, color='black')
        plaLine, = ax.plot([], [], lw=2, color='yellow')
        plbLine, = ax.plot([], [], lw=2, color='yellow')
        plcLine, = ax.plot([], [], lw=2, color='yellow')
        pldLine, = ax.plot([], [], lw=2, color='yellow')

        def init(): # initialization function: plot the background of each frame
            satLine.set_data([], [])
            thaLine.set_data([], [])
            thbLine.set_data([], [])
            thcLine.set_data([], [])
            thdLine.set_data([], [])
            plaLine.set_data([], [])
            plbLine.set_data([], [])
            plcLine.set_data([], [])
            pldLine.set_data([], [])
            return satLine, thaLine, thbLine, thcLine, thdLine, plaLine, plbLine, plcLine, pldLine
        
        def animate(i): # animation function. This is called sequentially
            # Set a frame centered on the satellite position
        
            ax.set_xlim(self.stVec[i, 0] - 2*self.dist, self.stVec[i, 0] + 2*self.dist)
            ax.set_ylim(self.stVec[i, 1] - 2*self.dist, self.stVec[i, 1] + 2*self.dist)
            self.generateSatGeometry(self.stVec[i, 0], self.stVec[i, 1], self.stVec[i, 2])
            satLine.set_data(self.satSquare[:,0], self.satSquare[:,1])
            thaLine.set_data(self.ta[:,0], self.ta[:,1])
            thbLine.set_data(self.tb[:,0], self.tb[:,1])
            thcLine.set_data(self.tc[:,0], self.tc[:,1])
            thdLine.set_data(self.td[:,0], self.td[:,1])
            
            if self.cmd[i, 0] > 0:
                plaLine.set_data(self.pa[:,0], self.pa[:,1])
            else:
                plaLine.set_data([], [])
            if self.cmd[i, 1] > 0:
                plbLine.set_data(self.pb[:,0], self.pb[:,1])
            else:
                plbLine.set_data([], [])
            if self.cmd[i, 2] > 0:
                plcLine.set_data(self.pc[:,0], self.pc[:,1])
            else:
                plcLine.set_data([], [])
            if self.cmd[i, 3] > 0:
                pldLine.set_data(self.pd[:,0], self.pd[:,1])
            else:
                pldLine.set_data([], [])
                
            return satLine, thaLine, thbLine, thcLine, thdLine, plaLine, plbLine, plcLine, pldLine
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.t), interval=1)#, blit=True)
        plt.show()

    def solve(self, t0, tf, dt):
        """
        Solves the equations of motion for the satellite
        args:
            t0: initial time in s
            tf: final time in s
            dt: time step in s
        """
        self.eqGenerator()
        r = RK45(self.eq, t0, self.stVec[-1, :], tf, max_step=dt)
        while r.status == 'running':
            r.step()
            self.updateState(r.t, r.y)
        self.plot()

def main():
    satellite = Satellite(100, 100, 1.0,thrust=1000.0)
    satellite.thrust = 20.0
    satellite.setPhysics(9.81)
    satellite.eqGenerator()
    satellite.setConditions(0., 0., 0.0, 0.0, 0.0, 0.0)
    satellite.setControlMode('static')
    satellite.setThrustSequence(np.array([[ 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [ 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [ 2.0, 0.0, 0.0, 0.0, 0.0],
                                          [ 3.0, 0.0, 0.0, 1.0, 1.0],
                                          [ 4.0, 1.0, 0.0, 1.0, 0.0],
                                          [ 5.0, 0.0, 0.0, 0.0, 0.0],
                                          [ 6.0, 0.0, 1.0, 0.0, 1.0],
                                          [ 8.0, 0.0, 0.0, 0.0, 0.0],
                                          [ 9.0, 1.0, 1.0, 0.0, 0.0],
                                          [10.0, 0.0, 0.0, 0.0, 0.0],
                                          [11.0, 0.0, 0.0, 1.0, 1.0],
                                          [12.0, 0.0, 0.0, 0.0, 0.0],
                                          [13.0, 0.0, 1.0, 0.0, 1.0],
                                          [14.0, 0.0, 0.0, 0.0, 0.0],
                                          [15.0, 1.0, 0.0, 1.0, 0.0],
                                          [16.0, 1.0, 1.0, 0.0, 0.0],]))
    #satellite.setPos(1.0, 1.0, 0.0)
    #satellite.setVel(0.0, 0.0, 0.0)

    gainDict = {'Kp_x': 5.0, 
                'Kd_x': 5.0, 
                'Ki_x': 0.0, 
                'Kp_y': 1.0, 
                'Kd_y': 10.0, 
                'Ki_y': 0.0, 
                'Kp_theta': .50, 
                'Kd_theta': 5.0, 
                'Ki_theta': 0.0,
                'Kp_xdot': 0.0,
                'Kd_xdot': 0.0,
                'Ki_xdot': 0.0,
                'Kp_ydot': 0.0,
                'Kd_ydot': 0.0,
                'Ki_ydot': 0.0,
                'Kp_thetadot': 0.0,
                'Kd_thetadot': 0.0,
                'Ki_thetadot': 0.0}
    
    satellite.setGains(gainDict)

    satellite.setWaypoint(1.0, 1.0)
    print('Starting simulation')
    satellite.solve(0, 20, 0.005)
    #satellite.plot2D()
    satellite.animate()
    
if __name__ == "__main__":
    main()
    

    
