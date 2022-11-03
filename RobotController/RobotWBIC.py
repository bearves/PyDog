from turtle import rt
import numpy as np
import qpsolvers
import RobotController.RobotDynamics as rdyn
import RobotController.RobotTask as rtask


class QuadWBIC(object):
    """
        Quadruped robot whole body impulse controller
    """

    # System constants
    n_leg: int = 4
    nv: int = 6 + n_leg * 3   # dynamic model generalized velocity dimension
    ground_fric: float = 0.4  # round Coulomb friction constant
    fz_min : float = 0.15     # minimum vertical force for supporting leg
    fz_max : float = 150      # maximum vertical force for supporting leg

    # Selecting matrix of the floating base dynamics
    # Sf selects first 6 rows, which is the floating base dynamics
    Sf : np.ndarray = np.zeros((6, nv))

    # Weight matrix
    Q : np.ndarray = None


    def __init__(self) -> None:
        """
            Create a whole body impulse controller
        """
        self.Sf = np.zeros((6, self.nv))
        self.Sf[0:6, 0:6] = np.eye(6)

    
    def update(self, 
               dyn_model: rdyn.RobotDynamicModel,
               ref_x_wcs: np.ndarray,
               ref_xdot_wcs: np.ndarray,
               ref_xddot_wcs: np.ndarray,
               fr_mpc: np.ndarray ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Update and run quadruped WBIC algorithm.
            The WBIC algorithm first calculate kinematic WBC to get command dq, qdot, and qddot.
            Then it calls the optimization algorithm according to the MPC outputs to get the final joint torque.
            The outputs of this algorithm are used for target setting and feed-forward setting for lower PD controller.

            Parameters:
                dyn_model (RobotDynamicModel reference): Robot dynamic model. Updated already.
                ref_x_wcs     (array(7 + n_leg * 3)): Position reference of the model in task space, described in WCS.
                ref_xdot_wcs  (array(6 + n_leg * 3)): Velocity reference of the model in task space, described in WCS.
                ref_xddot_wcs (array(6 + n_leg * 3)): Acceleration reference of the model in task space, described in WCS.
                fr_mpc    (array(n_leg * 3)) : tip force in WCS obtained by MPC, in pinocchio's definition.
            
            Returns:
                dq (array(n_leg * 3)): Joint position adjustment.
                qdot (array(n_leg * 3)): Joint target velocity.
                tau (array(n_leg * 3)): Joint target torque.
        """
        
        nj = self.n_leg * 3
        # update tasks
        task_list = self.update_task(dyn_model, ref_x_wcs, ref_xdot_wcs, ref_xddot_wcs)

        # run kinWBC
        dq, qdot, qddot = self.run_kinWBC(dyn_model, task_list)

        # run WBIC
        tau = self.run_WBIC(dyn_model, qddot, fr_mpc)

        # get results: 
        # dq, qdot for joint PD control target
        # tau_j for joint PD feedforward
        return dq[6:6+nj], qdot[6:6+nj], tau
    

    def update_task(self,                
                dyn_model: rdyn.RobotDynamicModel,
                ref_x_wcs: np.ndarray,
                ref_xdot_wcs: np.ndarray,
                ref_xddot_wcs: np.ndarray) -> list[rtask.WBCTask]:
        """
            Update task list according to current status. The task instance calculates 
            target position, velocity, acceleration and jacobian of each task.
            All inputs are following pinocchio's definition.

            Parameters:
                dyn_model (RobotDynamicModel reference): Robot dynamic model. Updated already.
                ref_x_wcs     (array(7 + n_leg * 3)): Position reference of the model in task space, described in WCS.
                ref_xdot_wcs  (array(6 + n_leg * 3)): Velocity reference of the model in task space, described in WCS.
                ref_xddot_wcs (array(6 + n_leg * 3)): Acceleration reference of the model in task space, described in WCS.
            
            Returns:
                task_list (list[WBCTask]): task of task at current time. Updated already.
        """
        
        task_list = []
        
        # add body ori task
        task_list.append(rtask.BodyOriTask())
        # add body pos task
        task_list.append(rtask.BodyPosTask())

        # add swing leg tip task
        for leg in range(self.n_leg):
            if not dyn_model.is_leg_supporting(leg):
                task_list.append(rtask.TipPosTask(leg))
        
        for task in task_list:
            assert(isinstance(task, rtask.WBCTask))
            task.update(dyn_model, ref_x_wcs, ref_xdot_wcs, ref_xddot_wcs)

        return task_list
        

    def run_kinWBC(self,
               dyn_model: rdyn.RobotDynamicModel,
               task_list: list[rtask.WBCTask]) -> \
            tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Kinematic WBC task execution.
            Calculate dq, qdot, qddot using Task priority algorithm.
            Tasks are commonly ordered as 
                1. Support leg tip constraints
                2. Body orientation
                3. Body position
                4. Swing leg tip position
            
            for task_i, we have

                dq_i    = dq_{i-1}    + JtpreInv(pos_err - Jt_i * dq_{i-1})
                qdot_i  = qdot_{i-1}  + JtpreInv(vel_des - Jt_i * qdot{i-1})
                qddot_i = qddot_{i-1} + JtpreBar(acc_cmd - Jtdot_i*qdot - Jt_i * qddot{i-1})

            where

                pos_err = x_des_i - x_i
                acc_cmd = acc_des_i + kp(x_des_i - x_i) + kd(xdot_des_i - xdot_i)
                Jtpre = Jt_i N_{i-1}
                N_{i-1} = N_last * (I - pinv(Jt_{i-1})*Jt_{i-1})
                N_last_0 = I
                JtpreInv use SVD pseudo inverse.
                JtpreBar use dynamic consistent pseudo inverse.

            Parameters:
                dyn_model (RobotDynamicModel reference): Robot dynamic model. Updated already.
                task_list (list[WBCTask]): task of task at current time. Updated already.
            
            Returns:
                dq    (array(6 + n_leg * 3)): joint position adjustment.
                qdot  (array(6 + n_leg * 3)): joint desired velocity.
                qddot (array(6 + n_leg * 3)): joint commanded acceleration.
        """

        nv = self.nv

        # contact constraints setup
        Nlast = np.eye(nv)
        Jc = dyn_model.get_contact_jacobian_or_none()
        if Jc is None:
            Nlast = np.eye(nv)
        else:
            Nlast = np.eye(nv) - np.linalg.pinv(Jc) @ Jc

        dq = np.zeros(nv)
        qdot = np.zeros(nv)

        # task execution to obtain dq and qdot
        for task in task_list:
            Jt = task.get_jacobian()
            Jtpre = Jt @ Nlast
            JtpreInv = np.linalg.pinv(Jtpre)
            dq += JtpreInv @ (task.get_pos_err() - Jt @ dq)
            qdot += JtpreInv @ (task.get_vel_des() - Jt @ qdot)
            Nlast = Nlast @ (np.eye(nv) - JtpreInv @ Jtpre)

        # task execution to obtain qddot, different pseudo-inverse is used
        H = dyn_model.get_mass_mat()
        Nlast = np.eye(nv)
        qddot = np.zeros(nv)

        if Jc is None:
            Nlast = np.eye(nv)
            qddot = np.zeros(nv)
        else:
            JcBar = self.dynConInv(Jc, H)
            qddot = JcBar @ -dyn_model.get_contact_jcdqd_or_none()
            Nlast = np.eye(nv) - JcBar @ Jc

        for task in task_list:
            Jt = task.get_jacobian()
            Jtdqd = task.get_jdqd()
            Jtpre = Jt @ Nlast
            JtpreBar = self.dynConInv(Jtpre, H)
            qddot += JtpreBar @ (task.get_acc_des() - Jtdqd - Jt @ qddot)
            Nlast = Nlast @ (np.eye(nv) - JtpreBar @ Jtpre)

        return dq, qdot, qddot


    def run_WBIC(self,
             dyn_model: rdyn.RobotDynamicModel,
             qddot_cmd: np.ndarray,
             fr_mpc: np.ndarray) -> np.ndarray:
        """
            Solve WBIC problem using qddot_cmd from kinWBC and fr_mpc from MPC solver.
            The WBIC problem is a QP optimization problem as below.

            min (dfr.T Q_f dfr + da.T Q_a da)
            s.t.
                fr = fr_mpc + dfr,
                qddot = qddot_cmd + [da; 0].T,
                Sfb*(M*qddot + C*qdot + G) = Sfb*Jc.T*fr
                fr_x, fr_y < mu fr_z, fr_z < fr_max, fr_z > fr_min, for supporting legs.
            where
                da in R6x1 is the trunk accel adjustment.
                dfr in R12x1 is the tip reaction force adjustment.
            
            Parameters:
                dyn_model (RobotDynamicModel reference): Robot dynamic model. Updated already.
                qddot_cmd (array(6 + n_leg * 3)) : robot's acceleration commands obtained by kinWBC, in pinocchio's definition.
                fr_mpc    (array(n_leg * 3)) : tip force in WCS obtained by MPC, in pinocchio's definition.
            
            Returns:
                tau (array(n_leg * 3)): joint torque after optimization. 
        """

        # for optimization problem, leg x = [da dfr]
        # where da in R(6x1), dfr in R((3*nsp)x1)
        nsp = dyn_model.get_n_support_legs()
        nx = 6 + nsp * 3

        # setup cost Q = diag([Qa Qf])
        self.Q = np.zeros((nx, nx))
        # setup cost matrix Q = diag[Qa Qfr]
        Qa = 0.1 * np.eye(6)
        Qf = 1 * np.eye(nsp * 3)
        self.Q[0:6, 0:6] = Qa
        self.Q[6:nx, 6:nx] = Qf
        self.q = np.zeros(nx)
        
        # setup equality constraints D x = d, stored in self.D, self.d
        self.cal_dynamic_equation_constraint(dyn_model, qddot_cmd, fr_mpc)

        # setup inquality constraints C x < c, stored in self.C, self.c
        self.cal_support_constraint(dyn_model, fr_mpc)

        # solve qp problem
        self.solve_qp(dyn_model, qddot_cmd, fr_mpc)

        # solve joint torque
        self.solve_joint_tau(dyn_model)
        
        return self.tau_joint_result
        

    def cal_dynamic_equation_constraint(self,
                                        dyn_model: rdyn.RobotDynamicModel,
                                        qddot_cmd: np.ndarray,
                                        fr_mpc: np.ndarray):
        """
            Calculate equality constraint matrices.
            The equality constraint is actually the dynamic equation of the floating base, i.e.

                Sf ( M (qddot_cmd + [da; 0].T) + Coriolis + Gravity ) = Sf Jc.T (fr_mpc + dfr)

            Re-arrange the above equation, we obtain

                [(Sf M).top_left(6, 6),  -Sf Jc.T] [da; dfr] = -Sf ( M qddot_cmd + Coriolis + Gravity - Jc.T fr_mpc)

            Therefore, 

                D = [(Sf M).top_left(6, 6),  -Sf Jc.T],
                d = -Sf ( M qddot_cmd + Coriolis + Gravity - Jc.T fr_mpc).

            Parameters:
                dyn_model (RobotDynamicModel reference): Robot dynamic model. Updated already.
                qddot_cmd (array(6 + n_leg * 3)) : robot's acceleration commands obtained by kinWBC, in pinocchio's definition.
                fr_mpc    (array(n_leg * 3)) : tip force in WCS obtained by MPC, in pinocchio's definition.
        """
        nsp = dyn_model.get_n_support_legs()
        M = dyn_model.get_mass_mat()
        cori = dyn_model.get_coriolis_mat() @ dyn_model.v
        grav = dyn_model.get_gravity_term()
        Jc = dyn_model.get_contact_jacobian_or_none()

        nx = 6 + nsp * 3
        self.D = np.zeros((6, nx))
        self.D[0:6,0:6] = M[0:6,0:6]  # (Sf M).top_left(6, 6) == M.top_left(6, 6)
        self.d = -self.Sf @ (M @ qddot_cmd + cori + grav)
        if nsp > 0:
            self.D[0:6,6:nx] = -self.Sf @ Jc.T
            # select supporting leg's fr_mpc for optimization
            fr_mpc_support = np.zeros(nsp * 3)
            cnt = 0
            for leg in range(self.n_leg):
                if dyn_model.is_leg_supporting(leg):
                    fr_mpc_support[cnt*3:3+cnt*3] = fr_mpc[leg*3:3+leg*3]
                    cnt += 1
            self.d += self.Sf @ Jc.T @ fr_mpc_support


    def cal_support_constraint(self,
                                dyn_model: rdyn.RobotDynamicModel,
                                fr_mpc: np.ndarray):
        """
            Calculate inequality constraint matrices.
            The inequality constraint is actually the friction pyramid of the supporting legs, i.e.

                                        fz < fz_max
                                       -fz < -fz_min
                            -fx -  mu * fz < 0
                             fx -  mu * fz < 0
                            -fy -  mu * fz < 0
                             fy -  mu * fz < 0

            Rewrite using the matrix form:

                                       U (fr_mpc + dfr) < u

            Re-arrange the above equation, we obtain
                                        
                                        U dfr < u - U fr_mpc 

            Therefore, 

                C = U,
                c = u - U fr_mpc.

            Parameters:
                dyn_model (RobotDynamicModel reference): Robot dynamic model. Updated already.
                fr_mpc    (array(n_leg * 3)) : tip force in WCS obtained by MPC, in pinocchio's definition.
        """
        mu = self.ground_fric
        U_support = np.array([[ 0,  0,   1],
                              [ 0,  0,  -1],
                              [ 1,  0, -mu],
                              [-1,  0, -mu],
                              [ 0,  1, -mu],
                              [ 0, -1, -mu]])
        u_support = np.array([self.fz_max, -self.fz_min, 0, 0, 0, 0])

        nsp = dyn_model.get_n_support_legs()
        self.C = np.zeros((nsp*6, 6+nsp*3))
        self.c = np.zeros(nsp*6)

        if nsp > 0:
            cnt = 0
            for leg in range(self.n_leg):
                if dyn_model.is_leg_supporting(leg):
                    self.C[0+cnt*6:6+cnt*6, 6+cnt*3:6+3+cnt*3] = U_support
                    self.c[0+cnt*6:6+cnt*6] = u_support - U_support @ fr_mpc[0+leg*3:3+leg*3]
                    cnt += 1


    def solve_qp(self,
                 dyn_model: rdyn.RobotDynamicModel,
                 qddot_cmd: np.ndarray,
                 fr_mpc: np.ndarray):
        """
            Solve the WBIC qp problem, and add results [da,dfr] back to qddot and fr_mpc.
            Results are stored in self.fr_result and self.qddot_result.

            Parameters:
                dyn_model (RobotDynamicModel reference): Robot dynamic model. Updated already.
                qddot_cmd (array(6 + n_leg * 3)) : robot's acceleration commands obtained by kinWBC, in pinocchio's definition.
                fr_mpc    (array(n_leg * 3)) : tip force in WCS obtained by MPC, in pinocchio's definition.
        """

        x = qpsolvers.solve_qp(P=self.Q, q=self.q,
                               A=self.D, b=self.d.flatten(),
                               G=self.C, h=self.c.flatten(),
                               solver="quadprog")
        
        self.da_result = x[0:6]
        self.dfr_result = x[6:]

        # add dfr back to fr
        self.fr_result = fr_mpc.copy()
        cnt = 0
        for leg in range(self.n_leg):
            if dyn_model.is_leg_supporting(leg):
                self.fr_result[0+leg*3:3+leg*3] += self.dfr_result[0+cnt*3:3+cnt*3]
                cnt += 1
        # add da back to qddot
        self.qddot_result = qddot_cmd.copy()
        self.qddot_result[0:6] += self.da_result


    def solve_joint_tau(self,
                  dyn_model: rdyn.RobotDynamicModel):
        """
            Solve joint tau from dynamic equation:

                tau = [tau_f tau_j] = M qddot_result + C + G - Jc.T Fr_result

            Results are stored in self.tau_result and self.tau_joint_result.

            Parameters:
                dyn_model (RobotDynamicModel reference): Robot dynamic model. Updated already.
        """
        self.tau_result = dyn_model.get_tau(self.qddot_result, self.fr_result)
        self.tau_joint_result = self.tau_result[6:6+self.n_leg*3]
        return self.tau_joint_result

    
    def dynConInv(self, J: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
            Calculate the dynamic consistent pseudo-inverse of J, with the given mass matrix H, 

                JBar = H^-1 * J.T * (J * H^-1 * J.T)^-1

            Parameters:
                J : Jacobian matrix.
                H : Mass matrix of the dynamic model.
            Returns:
                JBar: The dynamic consistent pseudo-inverse of J.
        """
        HInv = np.linalg.inv(H)
        JBar = HInv @ J.T @ np.linalg.pinv(J @ HInv @ J.T)
        return JBar

