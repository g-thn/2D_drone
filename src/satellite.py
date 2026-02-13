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
        self.tfNext = self.tfAll[self.nNext] # Next thruster sequence time
        
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

    def control(self, t, y):
        """
        Sets the control inputs for the thruster of the satellite in case of a PID controller
        args:
            t: time in s
            y: state vector
        """
        if t > self.tfNext:
            self.nCurrent += 1
            self.nNext += 1
            self.tfCurrent = self.tfAll[self.nCurrent]
            self.tfNext = self.tfAll[self.nNext]
        self.cmdTmp = self.cmdAll[(self.cmdAll[:,0] > self.tfCurrent), :][0,:]
        

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
            ydot[3] = self.cmdTmp[0]*np.sin(y[2]+self.cmdTmp[1])/self.mass
            ydot[4] = self.cmdTmp[0]*np.cos(y[2]+self.cmdTmp[1])/self.mass - self.g
            ydot[5] = self.cmdTmp[0]*np.sin(self.cmdTmp[1])*self.com/self.inertia
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

        axs[3, 0].plot(self.t, self.cmd[:, 0],'k')
        axs[3, 0].set_title('F (N)')
        axs[3, 0].set_xlabel('Time (s)')
        axs[3, 0].set_ylabel('Force (N)')

        axs[3, 1].plot(self.t, self.cmd[:, 1]*180/np.pi,'k')
        axs[3, 1].set_title('delta (deg)')  
        axs[3, 1].set_xlabel('Time (s)')
        axs[3, 1].set_ylabel('Angle (deg)')

        plt.show()

    def plotControl(self):
        """
        Plots the control inputs
        args:
            None
        """
        fig, axs = plt.subplots(2, 1)
        # Make the figure large
        fig.set_size_inches(18.5, 10.5)

        # Separate the plots
        fig.tight_layout(pad=3.0)

        axs[0].plot(self.t, self.cmd[:, 0])
        axs[0].set_title('F (N)')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Force (N)')

        axs[1].plot(self.t, self.cmd[:, 1]*180/np.pi)
        axs[1].set_title('delta (deg)')  
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Angle (deg)')

        plt.show()

    def plot2D(self):
        """
        Plots the satellite in 2D
        args:
            None
        """
        fig = plt.figure()
        ax = plt.axes(xlim=(-0.1, 5), ylim=(-0.5, 0.5))
        ax.set_aspect('equal')
        x_top = self.com*np.sin(self.stVec[:, 2])
        y_top = self.com*np.cos(self.stVec[:, 2])
        x_bot = -self.com*np.sin(self.stVec[:, 2])
        y_bot = -self.com*np.cos(self.stVec[:, 2])
        x_force = self.cmd[:,0]*np.sin(self.stVec[:, 2]+self.cmd[:,1])*0.05
        y_force = self.cmd[:,0]*np.cos(self.stVec[:, 2]+self.cmd[:,1])*0.05
        
        for i in range(len(self.t),10):
            plt.plot([self.stVec[i, 0]+x_top[i], self.stVec[i, 0]+x_bot[i]], 
                     [self.stVec[i, 1]+y_top[i], self.stVec[i, 1]+y_bot[i]], color='red')
            plt.plot([self.stVec[i, 0]+x_top[i], self.stVec[i, 0]+x_top[i]+x_force[i]], 
                     [self.stVec[i, 1]+y_top[i], self.stVec[i, 1]+y_top[i]+y_force[i]], color='blue')
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
        x_top = self.com*np.sin(self.stVec[:, 2])
        y_top = self.com*np.cos(self.stVec[:, 2])
        x_bot = -self.com*np.sin(self.stVec[:, 2])
        y_bot = -self.com*np.cos(self.stVec[:, 2])
        x_force = self.cmd[:,0]*np.sin(self.stVec[:, 2]+self.cmd[:,1])
        y_force = self.cmd[:,0]*np.cos(self.stVec[:, 2]+self.cmd[:,1])

        # Normalize the force
        norm = (x_force**2+y_force**2)**0.5
        x_force = x_force/(self.maxForce)
        y_force = y_force/(self.maxForce)

        line, = ax.plot([], [], lw=2, color='red')
        force, = ax.plot([], [], lw=1, color='blue')
        comPt, = ax.plot([], [], lw=2, color='black',marker='o')

        def init():
            line.set_data([], [])
            force.set_data([], [])
            return line, force
        def animate(i):
            
            ax.set_xlim(self.stVec[i, 0] - .5, self.stVec[i, 0] + .5)
            ax.set_ylim(self.stVec[i, 1] - .5, self.stVec[i, 1] + .5)
            line.set_data([self.stVec[i, 0]+x_top[i], self.stVec[i, 0]+x_bot[i]], 
                          [self.stVec[i, 1]+y_top[i], self.stVec[i, 1]+y_bot[i]])
            force.set_data([self.stVec[i, 0]+x_top[i], self.stVec[i, 0]+x_top[i]+x_force[i]], 
                           [self.stVec[i, 1]+y_top[i], self.stVec[i, 1]+y_top[i]+y_force[i]])
            comPt.set_data([self.stVec[i, 0]], 
                           [self.stVec[i, 1]])
            #force.set_color([0, 0, 1-norm[i]/self.maxForce])
            
            
            
            return line, force
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
    satellite = satellite(0.5, 0.1, 0.1)
    satellite.setPhysics(9.81)
    satellite.eqGenerator()
    satellite.setConditions(0., 0., 0*np.pi/180., 2.0, 0.0, 0.0)
    satellite.setControlMode('static')
    satellite.setLimitControl(100.0,0.0,45.0*np.pi/180)
    satellite.setThrustSequence(np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                          [1.0, 1.0, 0.0, 0.0, 0.0],
                                          [2.0, 1.0, 0.0, 1.0, 0.0],
                                          [3.0, 0.0, 1.0, 0.0, 0.0],
                                          [4.0, 0.0, 1.0, 1.0, 0.0],
                                          [5.0, 0.0, 0.0, 0.0, 1.0],
                                          [6.0, 1.0, 0.0, 0.0, 0.0]]))
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
    satellite.solve(0, 100, 0.01)
    #satellite.plot2D()
    satellite.animate()
    
if __name__ == "__main__":
    main()
    

    
