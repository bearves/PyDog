import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.linalg import expm
import qpsolvers


class QuadSingleBodyWBC(object):
    """
        Quadruped robot whole body controller (PID) based on single floating base dynamic.
    """

    # WBC settings
    leap: int = 25                  # Solve WBC every (leap) simulating step
    dt: float = 1/1000.             # time step of the simulator
    dt_wbc: float = 1/1000. * leap  # WBC solving time interval

    # System config
    n_leg: int = 4
    dim_s: int = 12                 # dimension of system states
    dim_u: int = n_leg * 3          # dimension of system inputs

    # Dynamic constants
    Ib : np.ndarray = np.diag([0.07, 0.26, 0.242])*0.8 # mass inertia of floating base
    invIb : np.ndarray = np.eye(3)
    mb : float = 10                                    # mass of the floating base
    ground_fric : float = 0.4                          # ground Coulomb friction constant
    fz_min : float = 0.1                               # minimum vertical force for supporting leg
    fz_max : float = 150                               # maximum vertical force for supporting leg

    # WBC weights
    Qk = np.array([1000,1000,0,0,0,100,1,1,1,1,1,1])
    kpf = Qk[3:6]
    kdf = Qk[9:12]
    kpt = Qk[0:3]
    kdt = Qk[6:9]

    # system ref states and actual states
    x0 : np.ndarray = np.zeros(dim_s) # current state
    x_ref : np.ndarray = np.zeros(dim_s) # reference states

    # WBC holders
    H : np.ndarray = np.zeros((dim_u, dim_u))
    G : np.ndarray = np.zeros(dim_u)
    C : np.ndarray = np.zeros((6, ))

    def __init__(self, sim_dt: float):
        """
            Create a whole body controller based on single floating base dynamic for quadruped robot.

            Parameters:
                sim_dt (float): time step of the simulator
        """
        self.dt = sim_dt
        self.dt_wbc = sim_dt * self.leap
        self.invIb = np.linalg.inv(self.Ib)


    def need_solve(self, count: int) -> bool:
        """
            Check whether the wbc need to be solved at this time count.

            Parameters:
                count (int): step counter of the simulator

            Returns:
                solve_flag (bool): a flag indicting whether wbc should be
                                   solved at this time count
        """
        solve_flag = count % self.leap == 0
        return solve_flag


    def update_wbc_matrices(self,
                            body_orn : np.ndarray,
                            r_leg : np.ndarray,
                            support_state : np.ndarray,
                            x0 : np.ndarray,
                            x_ref : np.ndarray):
        """
            Update matrices H, G, C, c, D, d for solve floating base PID control problem.

            For floating base system, its dynamics can be written as

                            m * vdot = m * g + sum(f_i)
                            J * wdot + w x Jw = sum (r_leg_i x f_i)
            
            where 
                            m is the body mass, 
                            J is the body inertia in WCS, 
                            r_leg is the leg tip position minus body position in WCS
                            f     is the leg tip force in WCS
                            g is the gravitational acceleration
                            w is the angular velocity of the body,
                            vdot is the linear acceleration of the body,
                            wdot is the angular acceleration of the body

            Simplify and neglect w x Jw, we have 

                            [I   I   I   I;           [m * (vdot - g);
                            r1^ r2^ r3^ r4^] * f    =  J * wdot]                    (Eq.1)

            Here, hat(^) is the matrix form of cross product. Eq.1 can be rewritten as
                                            A * f - b = 0,
            where
                                    A = [I   I   I   I; 
                                        r1^ r2^ r3^ r4^],
                                    b = [m * (vdot - g);
                                         J * wdot]

            First a PD control law is formulated to get expect vdot and wdot

                            vdot_ext = kpf * (pos_ref - pos_act) + kdf * (vel_ref - vel_act)
                            wdot_ext = kpt * (rpy_ref - rpy_act) + kdt * (angvel_ref -angvel_act)

            Then, a QP problem can be formulated as

                            min (A*f - b).T Q (A*f - b) + f.T * (R.T * R) * f
                            s.t.
                                    for swinging legs,
                                            f_i = 0 
                                    for supporting legs,
                                        f_i_x, f_i_y < mu f_i_z;
                                        fz_min < f_i_z < fz_max
            
            And it can be rewritten in the standard QP form as
                            min 1/2 * f.T * H * f + G.T * f
                            s.t.
                                    D f = d
                                    C f < c
            
            Parameters:
                body_orn (array(4)): current actual body orientation in quaternion.
                r_leg (array(n_leg * 3)): current leg tip position in WCS.
                support_state (array(n_leg)): current leg support state.
                x0 (array(dim_s)): current robot state.
                x_ref (array(dim_s)): reference robot state.
        """

        self.x0 = x0.copy()
        Rb = Rot.from_quat(body_orn).as_matrix()
        Iw = Rb @ self.Ib @ Rb.T

        self.x_ref = x_ref.copy()
        self.x_err = x_ref - x0
        
        vdot_ext = self.kpf * self.x_err[3:6] + self.kdf * self.x_err[9:12]
        wdot_ext = self.kpt * self.x_err[0:3] + self.kdt * self.x_err[6:9]
        
        b = np.zeros(6)
        b[0:3] = self.mb * (vdot_ext + np.array([0.38, 0, 9.81])) 
        b[3:6] = Iw @ wdot_ext

        A = np.zeros((6, self.dim_u))
        p_b = self.x0[3:6]
        for i in range(self.n_leg):
            A[0:3, 0+i*3:3+i*3] = np.eye(3)

            ri = r_leg[0+i*3:3+i*3] - p_b
            rx, ry, rz = ri[0], ri[1], ri[2]
            rHat = np.array([[  0, -rz,  ry],
                             [ rz,   0, -rx],
                             [-ry,  rx,   0]])
            A[3:6, 0+i*3:3+i*3] = rHat
        
        self.R = 1e-6 * np.eye(self.dim_u)
        self.H = A.T @ A + self.R.T @ self.R
        self.G = -A.T @ b

        self.C, self.c, _ = self.cal_friction_constraint(support_state)
        self.D, self.d, _ = self.cal_swing_force_constraints(support_state)


    def cal_friction_constraint(self, support_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """
            Generate inequality constraints that obey the friction physics
            for this moment,
                                C f < rhs
            where f in R(3*n_leg x 1) is the leg tip force in WCS of this moment,
                  C in R(6*n_support x 3*n_leg) is the constraint matrix,
                  rhs in R(6*n_support x 1) is the RHS of the constraint.
            
            Parameters:
                support_state (array(n_leg)) :
                    Leg support state, 1=support, 0=swing.
            
            Returns:
                C (array(n_support*6, n_leg*3)) : constraint matrix.
                rhs (array(n_support*6) : RHS of the constraint.
                n_support (int) : number of supporting legs.
        """

        assert(isinstance(support_state, np.ndarray))
        assert(support_state.size == self.n_leg)

        # friction pyramid of the support leg's force f_i_k
        # i.e.
        #             fz < fz_max
        #            -fz < -fz_min
        # -fx -  mu * fz < 0
        #  fx -  mu * fz < 0
        # -fy -  mu * fz < 0
        #  fy -  mu * fz < 0
        mu = self.ground_fric
        C_support = np.array([[ 0,  0,   1],
                              [ 0,  0,  -1],
                              [ 1,  0, -mu],
                              [-1,  0, -mu],
                              [ 0,  1, -mu],
                              [ 0, -1, -mu]])
        rhs_support = np.array([self.fz_max, -self.fz_min, 0, 0, 0, 0])

        n_support : int = int(np.sum(support_state >= 0.7))
        C = np.zeros((n_support*6, self.dim_u))
        rhs = np.zeros(n_support*6)

        cnt = 0
        for i in range(self.n_leg):
            if support_state[i] >= 0.7:
                C[0+cnt*6:6+cnt*6, 0+i*3:3+i*3] = C_support
                rhs[0+cnt*6:6+cnt*6] = rhs_support
                cnt = cnt+1
        return C, rhs, int(n_support)


    def cal_swing_force_constraints(self, support_state : np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """
            Generate equality constraints for this moment to restrict the 
            forces of swing legs to be zero,
                                D f = rhs
            where f in R(3*n_leg x 1) is the leg tip force in WCS of this moment,
                  D in R(3*n_swing x 3*n_leg) is the constraint matrix,
                  rhs = 0 in R(3*n_swing x 1) is the RHS of the constraint.

            Parameters:
                support_state (array(n_leg)) :
                    Leg support state, 1=support, 0=swing.
            
            Returns:
                D (array(n_swing*3, n_leg*3)) : constraint matrix.
                rhs (array(n_swing*3) : RHS of the constraint.
                n_swing (int) : number of swinging legs.

        """
        assert(isinstance(support_state, np.ndarray))
        assert(support_state.size == self.n_leg)

        # restricts of the leg forces according to the swing/support state
        # of this moment
        # i.e.
        #   fx_i_k = 0
        #   fy_i_k = 0     if  the ith leg is swinging
        #   fz_i_k = 0 
        D_swing = np.eye(3)

        n_support : int = int(np.sum(support_state >= 0.7))
        n_swing : int = self.n_leg - n_support
        D = np.zeros((n_swing*3, self.dim_u))
        rhs = np.zeros(n_swing*3)

        cnt = 0
        for i in range(self.n_leg):
            if support_state[i] >= 0.7:
                continue
            D[0+cnt*3:3+cnt*3, 0+i*3:3+i*3] = D_swing
            cnt = cnt+1
        return D, rhs, n_swing

    def solve(self) -> np.ndarray:
        """
            Solve WBC problem for quadruped robot, must be called after all
            matrices have been updated. 

            Returns:
                u_mpc(array(n_leg*3)) :
                    predicted optimal input of the system, i.e. the leg tip force in WCS.
        """
        #print(self.H)
        #print(np.linalg.det(self.H))
        res = qpsolvers.solve_qp(P=self.H, q=self.G.flatten(),
                                 A=self.D, b=self.d,
                                 G=self.C, h=self.c,
                                 solver="quadprog")
        u_wbc = res.flatten()
        return u_wbc
