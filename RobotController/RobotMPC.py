import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.linalg import expm
import qpsolvers


class QuadConvexMPC(object):
    """
        Quadruped robot model predictive controller
    """
    # MPC settings
    leap: int = 25                 # Solve MPC every (leap) simulating step
    dt: float = 1/1000.            # time step of the simulator
    dt_mpc: float = dt * leap # MPC solving time interval
    horizon_length: int = 20       # prediction horizon length of MPC

    # System config
    n_leg: int = 4
    dim_s: int = 13                # dimension of system states
    dim_u: int = n_leg * 3         # dimension of system inputs

    # Dynamic constants
    Ib : np.ndarray = np.diag([0.07, 0.26, 0.242]) # mass inertia of floating base
    invIb : np.ndarray = np.eye(3)
    mb : float = 10                                # mass of the floating base
    ground_fric : float = 0.4                      # ground Coulomb friction constant
    fz_min : float = 0.1                           # minimum vertical force for supporting leg
    fz_max : float = 150                           # maximum vertical force for supporting leg

    # MPC weights
    Qk = np.diag([100,100,100,1,1,100,1,1,1,1,1,1,0]) 
    Rk = 1e-6 * np.eye(dim_u)

    # MPC matrix lists
    Ak_list : np.ndarray = np.zeros((horizon_length, dim_s, dim_s))
    Bk_list : np.ndarray = np.zeros((horizon_length, dim_s, dim_u))
    Ck_list : np.ndarray = np.zeros((horizon_length, 6*n_leg, dim_u))
    ck_list : np.ndarray = np.zeros((horizon_length, 6*n_leg, 1))
    Dk_list : np.ndarray = np.zeros((horizon_length, 3*n_leg, dim_u))
    dk_list : np.ndarray = np.zeros((horizon_length, 3*n_leg, 1))
    n_support_list : np.ndarray = np.zeros(horizon_length, dtype=int)
    n_swing_list : np.ndarray = np.zeros(horizon_length, dtype=int)

    # system ref states and actual states
    x0 : np.ndarray = np.zeros(dim_s)  # current state
    x_ref_seq : np.ndarray = np.zeros((horizon_length, dim_s)) # future reference states
    X_ref : np.ndarray = np.zeros(horizon_length * dim_s)      # flattened future reference states

    # MPC super matrices
    Abar : np.ndarray = np.zeros((horizon_length*dim_s, dim_s))
    Bbar : np.ndarray = np.zeros((horizon_length*dim_s, horizon_length*dim_u))
    Cbar : np.ndarray = np.array([])
    cbar : np.ndarray = np.array([])
    Dbar : np.ndarray = np.array([])
    dbar : np.ndarray = np.array([])
    Qbar : np.ndarray = np.zeros((horizon_length*dim_s, horizon_length*dim_s))
    Rbar : np.ndarray = np.zeros((horizon_length*dim_u, horizon_length*dim_u))
    H    : np.ndarray = np.array([])
    G    : np.ndarray = np.array([])

    def __init__(self, sim_dt: float):
        """
            Create a model predictive controller for quadruped robot.

            Parameters:
                sim_dt (float): time step of the simulator
        """
        self.dt = sim_dt
        self.dt_mpc = sim_dt * self.leap
        self.invIb = np.linalg.inv(self.Ib)

    def cal_weight_matrices(self):
        """
            Compute time-invariant weight matrix Qbar and Rbar.
        """
        hz = self.horizon_length
        dim_s = self.dim_s
        dim_u = self.dim_u

        self.Qbar = np.zeros((dim_s * hz, dim_s * hz))
        self.Rbar = np.zeros((dim_u * hz, dim_u * hz))

        for cr in range(0, hz):
            self.Qbar[0 + cr * dim_s:dim_s + cr * dim_s, 0 + cr * dim_s:dim_s + cr * dim_s] = self.Qk
            self.Rbar[0 + cr * dim_u:dim_u + cr * dim_u, 0 + cr * dim_u:dim_u + cr * dim_u] = self.Rk


    def need_solve(self, count: int) -> bool:
        """
            Check whether the mpc need to be solved at this time count.

            Parameters:
                count (int): step counter of the simulator

            Returns:
                solve_flag (bool): a flag indicting whether MPC should be
                                   solved at this time count
        """
        solve_flag = count % self.leap == 0
        return solve_flag


    def update_mpc_matrices(self,
                            body_euler_seq : np.ndarray,
                            r_leg_seq : np.ndarray,
                            support_state_seq : np.ndarray,
                            x0 : np.ndarray,
                            x_ref_seq : np.ndarray):
        """
            Update time variant matrices Ak, Bk, Ck, ck, Dk, dk for each time
            moment in the prediction horizon.
            At each time moment, the system dynamics can be written as

                x_{k+1} = Ak xk + Bk uk
                s.t.
                    Ck uk < ck
                    Dk uk < dk
            
            Parameters:
                body_euler_seq (array(horizon_length, 3)) : 
                    sequence of future body euler angles in the predictive horizon. Euler
                    angle is in (roll, pitch, yaw) order.
                r_leg_seq (array(horizon_length, n_leg * 3)) :
                    sequence of future leg tip position in WCS in the predictive horizon.
                support_state_seq (array(horizon_length, n_leg)) :
                    sequence of future leg support state in the predictive horizon.
                x0 (array(dim_s)) :
                    current robot state.
                x_ref_seq (array(horizon_length, dim_s)) :
                    sequence of the reference robot state in the predictive horizon.
        """

        self.x0 = x0
        self.x_ref_seq = x_ref_seq

        body_pos = self.x0[3:6]

        hz = self.horizon_length
        for i in range(hz):
            A, B = self.cal_state_equation(body_pos, body_euler_seq[i, :], r_leg_seq[i, :])
            Ak, Bk = self.discretized_state_equation(A, B)
            Ck, ck, n_support = self.cal_friction_constraint(support_state_seq[i, :])
            Dk, dk, n_swing = self.cal_swing_force_constraints(support_state_seq[i, :])
            self.Ak_list[i, :, :] = Ak
            self.Bk_list[i, :, :] = Bk
            self.Ck_list[i, 0:6*n_support, :] = Ck
            self.ck_list[i, 0:6*n_support, 0] = ck
            self.Dk_list[i, 0:3*n_swing, :] = Dk
            self.dk_list[i, 0:3*n_swing, 0] = dk
            self.n_support_list[i] = n_support
            self.n_swing_list[i] = n_swing


    def update_super_matrices(self):
        """
            Collect all data to build Abar, Bbar, Cbar, cbar, Dbar, dbar for generic mpc problem:
                min f(X, U) = {X^T QBar X + U^T RBar U}
                s.t.
                    X_k+1 = Abar Xk + Bbar U
                    Cbar U < cbar
                    Dbar U = dbar
        """
        hz = self.horizon_length
        dim_s = self.dim_s
        dim_u = self.dim_u

        # build Abar and Bbar
        self.Abar = np.zeros((dim_s * hz, dim_s))
        self.Bbar = np.zeros((dim_s * hz, dim_u * hz))

        self.Abar[0:dim_s, 0:dim_s] = self.Ak_list[0, :, :]

        for i in range(1, hz):
            self.Abar[0 + i * dim_s:dim_s + i * dim_s, 0:dim_s] = \
                self.Ak_list[i, :, :] @ self.Abar[0 + (i - 1) * dim_s:dim_s + (i - 1) * dim_s, 0:dim_s]

        for col in range(0, hz):
            for row in range(0, hz):
                if row < col:
                    continue
                elif row == col:
                    self.Bbar[0 + row * dim_s:dim_s + row * dim_s, 0 + col * dim_u:dim_u + col * dim_u] = self.Bk_list[row, :, :]
                else:
                    self.Bbar[0 + row * dim_s:dim_s + row * dim_s, 0 + col * dim_u:dim_u + col * dim_u] \
                        = self.Ak_list[row, :, :] @ self.Bbar[0 + (row - 1) * dim_s:dim_s + (row - 1) * dim_s, 0 + col * dim_u:dim_u + col * dim_u]

        # build Cbar and cbar
        self.Cbar = np.zeros((6 * np.sum(self.n_support_list), dim_u * hz))
        self.cbar = np.zeros((6 * np.sum(self.n_support_list), 1))
        row_cnt = 0
        for i in range(hz):
            self.Cbar[row_cnt:row_cnt+6*self.n_support_list[i], 0+i*dim_u:dim_u+i*dim_u] = \
                self.Ck_list[i, 0:6*self.n_support_list[i], :]
            self.cbar[row_cnt:row_cnt+6*self.n_support_list[i], :] = \
                self.ck_list[i, 0:6*self.n_support_list[i], :]
            row_cnt = row_cnt + 6 * self.n_support_list[i]

        # build Dbar and dbar
        self.Dbar = np.zeros((3 * np.sum(self.n_swing_list), dim_u * hz))
        self.dbar = np.zeros((3 * np.sum(self.n_swing_list), 1))
        row_cnt = 0
        for i in range(hz):
            self.Dbar[row_cnt:row_cnt+3*self.n_swing_list[i], 0+i*dim_u:dim_u+i*dim_u] = \
                self.Dk_list[i, 0:3*self.n_swing_list[i], :]
            self.dbar[row_cnt:row_cnt+3*self.n_swing_list[i], :] = \
                self.dk_list[i, 0:3*self.n_swing_list[i], :]
            row_cnt = row_cnt + 3 * self.n_swing_list[i]

        # build X_ref
        for i in range(hz):
            self.X_ref[0+i*dim_s:dim_s+i*dim_s] = self.x_ref_seq[i, :]

        # build H and G
        self.H = self.Rbar + self.Bbar.T @ self.Qbar @ self.Bbar
        self.G = (self.Abar @ self.x0 - self.X_ref).T @ self.Qbar @ self.Bbar


    def cal_state_equation(self,    
                           body_pos: np.ndarray, 
                           body_euler: np.ndarray, 
                           r_leg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
            Calculate the state matrix A and input matrix B of
            the continuous system dynamics, i.e.
                qdot = Aq + Bu
            where
                q = [theta p omega v g] in R(13)
                theta = [roll pitch yaw] in R(3) is the euler angle of the body
                p = [x y z] in R(3) is the position of the body
                omega = [wx wy wz] in R(3) is the angular velocity of the body
                v = [xdot ydot zdot] in R(3) is the linear velocity of the body
                g in R is the gravity constant

                u = [f_1 f_2 ... f_n_leg] in R(3*n_leg) is the tip force of legs

                A in R(13 x 13) is the state matrix
                B in R(13 x 3*n_leg) is the input matrix

            Parameters:
                body_euler (array(3)) : body's euler angle in (roll, pitch, yaw) order.
                r_leg      (array(n_leg * 3)) : leg tip position in WCS.
            
            Returns:
                A (array(13, 13)) : A matrix of the continuous dynamics equation.
                B (array(13, n_leg * 3) : B matrix of the continuous dynamics equation.
        """
        # approximated continuous system dynamics
        yaw = body_euler[2]
        Rz = Rot.from_rotvec([0, 0, yaw]).as_matrix()
        A = np.zeros((self.dim_s, self.dim_s))
        A[0:3, 6:9] = Rz.T
        A[3:6, 9:12] = np.eye(3)
        A[11, 9] = 0.0381
        A[11, 12] = 1

        B = np.zeros((self.dim_s, self.dim_u))
        invIw = Rz @ self.invIb @ Rz.T

        for i in range(self.n_leg):
            ri = r_leg[0+i*3:3+i*3] - body_pos
            rx, ry, rz = ri[0], ri[1], ri[2]
            rHat = np.array([[  0, -rz,  ry],
                             [ rz,   0, -rx],
                             [-ry,  rx,   0]])
            B[6:9 ,0+i*3:3+i*3] = invIw @ rHat
            B[9:12,0+i*3:3+i*3] = (1./self.mb) * np.eye(3)

        return A, B


    def discretized_state_equation(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
            Get the discrete state matrix Ak and input matrix Bk
            using forward euler transformation, i.e.
                Ak = I + AT + 1/2 (AT)^2 + 1/6 (AT)^3 + ...,
                Bk =     BT + 1/2 AB T^2 + 1/6 (A*A*B)T^3 + ...
            
            Parameters:
                A (array(13, 13)) : A matrix of the continuous dynamics equation.
                B (array(13, n_leg * 3) : B matrix of the continuous dynamics equation.
            
            Returns:
                Ak (array(13, 13)) : Ak matrix of the discrete dynamics equation.
                Bk (array(13, n_leg * 3) : Bk matrix of the discrete dynamics equation.
        """
        ABc = np.zeros((self.dim_s+self.dim_u, self.dim_s+self.dim_u))
        ABc[0:self.dim_s,0:self.dim_s] = A
        ABc[0:self.dim_s,self.dim_s:self.dim_s+self.dim_u] = B
        ABk = self.dt_mpc * ABc
        expABk = expm(ABk)
        #Ak = np.eye(self.dim_s) + self.dt_mpc * A
        #Bk = self.dt_mpc * B
        Ak = expABk[0:self.dim_s,0:self.dim_s]
        Bk = expABk[0:self.dim_s,self.dim_s:self.dim_s+self.dim_u]

        return Ak, Bk

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
            Solve MPC problem for quadruped robot, must be called after all
            matrices have been updated. 

            Returns:
                u_mpc(array(n_leg*3, horizon_length)) :
                    predicted optimal input of the system, i.e. the leg tip force in WCS.
        """
        res = qpsolvers.solve_qp(P=self.H, q=self.G.flatten(),
                                 A=self.Dbar, b=self.dbar.flatten(),
                                 G=self.Cbar, h=self.cbar.flatten(),
                                 solver="quadprog")
        u_mpc = np.reshape(res, (self.dim_u, self.horizon_length), order='F')
        return u_mpc
