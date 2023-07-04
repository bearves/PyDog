#include "RobotMPC.h"
#include <vector>

QuadConvexMPC::QuadConvexMPC(double sim_dt, int leap) :
    qp_solver()
{
    /*
        Create a model predictive controller for quadruped robot.

        Parameters:
            sim_dt (float): time step of the simulator.
            leap (int): leap time of the MPC, so that the controller 
                solves MPC every (leap) simulating step.
    */
    back_solver = LocoMpc::QPSolver::SOLVER::QPOASES; // default solver

    this->dt = sim_dt;
    this->leap = leap;
    dt_mpc = sim_dt * leap;
    Ib.setZero();
    Ib.diagonal() << 0.07, 0.26, 0.242;
    invIb = Ib.inverse();

    mb = 10;
    ground_fric = 0.4; 
    fz_min = 0.1;
    fz_max = 150;

    Qk.resize(dim_s, dim_s);
    Qk.setZero();
    Qk.diagonal() << 100,100,0,0,0,100,1,1,1,1,1,1,0;

    Rk.resize(dim_u, dim_u);
    Rk.setIdentity();
    Rk *= 1e-6;

    for (int i = 0; i < HZ; i++)
    {
        Ak_list[i].resize(dim_s, dim_s);
        Bk_list[i].resize(dim_s, dim_u);
        Ck_list[i].resize(6*n_leg, dim_u);
        ck_list[i].resize(6*n_leg, 1);     
        Dk_list[i].resize(3*n_leg, dim_u); 
        dk_list[i].resize(3*n_leg, 1);     
        x_ref_seq[i].resize(dim_s, 1);
        support_state_seq[i].resize(n_leg, 1);
    }

    x0.resize(dim_s, 1);
    X_ref.resize(dim_s * HZ, 1);

    Abar.resize(HZ * dim_s, dim_s); 
    Bbar.resize(HZ * dim_s, HZ * dim_u); 
    Qbar.resize(HZ * dim_s, HZ * dim_s); 
    Rbar.resize(HZ * dim_u, HZ * dim_u); 

    H.resize(HZ * dim_u, HZ * dim_u);
    G.resize(1, HZ * dim_u);

    /*
    # friction pyramid of the support leg's force f_i_k
    # i.e.
    #             fz < fz_max
    #            -fz < -fz_min
    # -fx -  mu * fz < 0
    #  fx -  mu * fz < 0
    # -fy -  mu * fz < 0
    #  fy -  mu * fz < 0
    */
    Cf_support.resize(6, 3);
    cf_support.resize(6, 1);
    Cf_support << 0,  0,   1,
                  0,  0,  -1,
                  1,  0, -ground_fric,
                 -1,  0, -ground_fric,
                  0,  1, -ground_fric,
                  0, -1, -ground_fric;
    cf_support << fz_max, -fz_min, 0, 0, 0, 0;

    /*
    # restricts of the leg forces according to the swing/support state
    # of this moment
    # i.e.
    #   fx_i_k = 0
    #   fy_i_k = 0     if  the ith leg is swinging
    #   fz_i_k = 0
    */ 
    D_swing = Eigen::Matrix3d::Identity();
    d_swing = Eigen::Vector3d::Zero();

    empty.resize(0, 0);
}

void QuadConvexMPC::set_back_solver(LocoMpc::QPSolver::SOLVER solver)
{
    back_solver = solver;
}

void QuadConvexMPC::cal_weight_matrices()
{
    /*
        Compute time-invariant weight matrix Qbar and Rbar.
    */
    Qbar.setZero();
    Rbar.setZero();

    for(int cr = 0; cr < HZ; cr++)
    {
        Qbar.block<dim_s, dim_s>(cr*dim_s, cr*dim_s) = Qk;
        Rbar.block<dim_u, dim_u>(cr*dim_u, cr*dim_u) = Rk;
    }
}

bool QuadConvexMPC::need_solve(int count)
{
    /*
        Check whether the mpc need to be solved at this time count.

        Parameters:
            count (int): step counter of the simulator

        Returns:
            solve_flag (bool): a flag indicting whether MPC should be
                                solved at this time count
    */
    bool solve_flag = (count % leap == 0);
    return solve_flag;
}

void QuadConvexMPC::update_mpc_matrices(
    const Eigen::Matrix<double, HZ, 3>& body_euler_seq,
    const Eigen::Matrix<double, HZ, n_leg * 3>& r_leg_seq,
    const Eigen::Matrix<double, HZ, n_leg>& support_state_seq,
    const Eigen::Matrix<double, dim_s, 1>& x_act,
    const Eigen::Matrix<double, HZ, dim_s>& x_ref_seq)
{
    /*
        Update time variant matrices Ak, Bk, Ck, ck, Dk, dk for each time
        moment in the prediction horizon.
        At each time moment, the system dynamics can be written as

            x_{k+1} = Ak xk + Bk uk
            s.t.
                Ck uk < ck
                Dk uk = dk
        
        Parameters:
            body_euler_seq (array(horizon_length, 3)) : 
                sequence of future body euler angles in the predictive horizon. Euler
                angle is in (roll, pitch, yaw) order.
            r_leg_seq (array(horizon_length, n_leg * 3)) :
                sequence of future leg tip position in WCS in the predictive horizon.
            support_state_seq (array(horizon_length, n_leg)) :
                sequence of future leg support state in the predictive horizon.
            x_act (array(dim_s)) :
                current robot state.
            x_ref_seq (array(horizon_length, dim_s)) :
                sequence of the reference robot state in the predictive horizon.
    */

    x0 = x_act;
    for (int i = 0; i < HZ; i++)
    {
        Eigen::Matrix<double, 1, dim_s> x_ref_i = x_ref_seq.row(i);
        Eigen::Matrix<double, 1, n_leg> sp_i = support_state_seq.row(i);
        this->x_ref_seq[i] = x_ref_i.transpose();
        this->support_state_seq[i] = sp_i.transpose();
    }


    Eigen::Vector3d body_pos = x0.block<3,1>(3, 0);
    int n_support = 0;
    int n_swing = 0;
    total_support_legs = 0;
    total_swing_legs = 0;

    for(int i = 0; i < HZ; i++)
    {
        Eigen::Vector3d body_euler(body_euler_seq.row(i).transpose());
        Eigen::Matrix<double, n_leg*3, 1> r_leg(r_leg_seq.row(i).transpose());
        Eigen::Matrix<double, n_leg, 1> support_state(support_state_seq.row(i).transpose());

        cal_state_equation(body_pos, body_euler, r_leg, A, B);
        discretized_state_equation(A, B, Ak, Bk);
        cal_friction_constraint(support_state, Ck, ck, n_support);
        cal_swing_force_constraints(support_state, Dk, dk, n_swing);
        
        Ak_list[i] = Ak;
        Bk_list[i] = Bk;
        Ck_list[i].block(0, 0, 6*n_support, dim_u) = Ck.block(0, 0, 6*n_support, dim_u);
        ck_list[i].block(0, 0, 6*n_support, 1)     = ck.block(0, 0, 6*n_support, 1);
        Dk_list[i].block(0, 0, 3*n_swing, dim_u)   = Dk.block(0, 0, 3*n_swing, dim_u);
        dk_list[i].block(0, 0, 3*n_swing, 1)       = dk.block(0, 0, 3*n_swing, 1);
        n_support_list[i] = n_support;
        n_swing_list[i] = n_swing;
        total_support_legs += n_support;
        total_swing_legs += n_swing;
    }
}

void QuadConvexMPC::update_super_matrices()
{
    /*
        Collect all data to build Abar, Bbar, Cbar, cbar, Dbar, dbar for generic mpc problem:
            min f(X, U) = {X^T QBar X + U^T RBar U}
            s.t.
                X_k+1 = Abar Xk + Bbar U
                Cbar U < cbar
                Dbar U = dbar
    */

    // build Abar and Bbar
    Abar.setZero();
    Bbar.setZero();

    Abar.block<dim_s, dim_s>(0, 0) = Ak_list[0];
    for (int i = 1; i < HZ; i++)
    {
        Abar.block<dim_s, dim_s>(i*dim_s, 0) = Ak_list[i] * Abar.block<dim_s, dim_s>((i-1)*dim_s,0);
    }

    for (int col = 0; col < HZ; col++)
    {
        for (int row = 0; row < HZ; row++)
        {
            if (row < col) 
                continue;
            else if (row == col)
                Bbar.block<dim_s, dim_u>(row*dim_s, col*dim_u) = Bk_list[row];
            else
                Bbar.block<dim_s, dim_u>(row*dim_s, col*dim_u) = 
                    Ak_list[row] * Bbar.block<dim_s, dim_u>((row-1)*dim_s, col*dim_u);
        }
    }

    // build Cbar and cbar
    Cbar.resize(6 * total_support_legs, dim_u * HZ);
    cbar.resize(6 * total_support_legs, 1);
    Cbar.setZero();
    cbar.setZero();
    int row_cnt = 0;

    for (int i = 0; i < HZ; i++)
    {
        Cbar.block(row_cnt, i*dim_u, 6*n_support_list[i], dim_u) = Ck_list[i].block(0, 0, 6*n_support_list[i], dim_u);
        cbar.block(row_cnt, 0, 6*n_support_list[i], 1) = ck_list[i].block(0, 0, 6*n_support_list[i], 1);
        row_cnt += 6*n_support_list[i];
    }

    // build Dbar and dbar
    Dbar.resize(3 * total_swing_legs, dim_u * HZ);
    dbar.resize(3 * total_swing_legs, 1);
    Dbar.setZero();
    dbar.setZero();

    row_cnt = 0;

    for (int i = 0; i < HZ; i++)
    {
        Dbar.block(row_cnt, i*dim_u, 3*n_swing_list[i], dim_u) = Dk_list[i].block(0, 0, 3*n_swing_list[i], dim_u);
        dbar.block(row_cnt, 0, 3*n_swing_list[i], 1) = dk_list[i].block(0, 0, 3*n_swing_list[i], 1);
        row_cnt += 3*n_swing_list[i];
    }
    // build X_ref
    for (int i = 0; i < HZ; i++)
    {
        X_ref.block<dim_s, 1>(i*dim_s, 0) = x_ref_seq[i];
    }
}

void QuadConvexMPC::cal_state_equation(    
    const Eigen::Vector3d& body_pos, 
    const Eigen::Vector3d& body_euler, 
    const Eigen::Matrix<double, dim_u, 1>& r_leg, 
    Eigen::Matrix<double, dim_s, dim_s>& A,
    Eigen::Matrix<double, dim_s, dim_u>& B)
{
    /*
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
    */
    // approximated continuous system dynamics
    double yaw = body_euler[2];
    Eigen::AngleAxisd RotZ = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d Rz = RotZ.toRotationMatrix();
    A.setZero();
    A.block<3,3>(0, 6) = Rz.transpose();
    A.block<3,3>(3, 9) = Eigen::Matrix3d::Identity();
    A(11, 9) = 0.0381; // TODO: touch deepth, need to adjust
    A(11, 12)= 1;

    B.setZero();
    Eigen::Matrix3d invIw = Rz * invIb * Rz.transpose();

    for (int i = 0; i < n_leg; i++)
    {
        Eigen::Vector3d ri = r_leg.block<3,1>(i*3, 0) - body_pos;
        Eigen::Matrix3d rHat;
        rHat <<      0, -ri(2),  ri(1),
                    ri(2),      0, -ri(0),
                -ri(1),  ri(0),      0;
        B.block<3,3>(6, i*3) = invIw * rHat;
        B.block<3,3>(9, i*3) = (1./mb) * Eigen::Matrix3d::Identity();
    }
}

void QuadConvexMPC::discretized_state_equation(
    const Eigen::Matrix<double, dim_s, dim_s>& A, 
    const Eigen::Matrix<double, dim_s, dim_u>& B,
    Eigen::Matrix<double, dim_s, dim_s>& Ak,
    Eigen::Matrix<double, dim_s, dim_u>& Bk)
{
    /*
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
    */
    AT = dt_mpc * A;
    BT = dt_mpc * B;
    AT2 = AT * AT;
    AT3 = AT2 * AT;
    AT4 = AT2 * AT2;
    Ak.setIdentity();
    Ak += AT + 1/2.0*AT2   + 1/6.0*AT3    + 1/24.0*AT4;
    Bk  = BT + 1/2.0*AT*BT + 1/6.0*AT2*BT + 1/24.0*AT3*BT;
}

void QuadConvexMPC::cal_friction_constraint(
    const Eigen::Matrix<double, n_leg, 1>& support_state,
    Eigen::Matrix<double, n_leg*6, n_leg*3>& C,
    Eigen::Matrix<double, n_leg*6, 1>& c, 
    int& n_support)
{
    /*
        Generate inequality constraints that obey the friction physics
        for this moment,
                            C f < c
        where f in R(3*n_leg x 1) is the leg tip force in WCS of this moment,
                C in R(6*n_support x 3*n_leg) is the constraint matrix,
                c in R(6*n_support x 1) is the RHS of the constraint.
        
        Parameters:
            support_state (array(n_leg)) :
                Leg support state, 1=support, 0=swing.
        
        Returns:
            C (array(n_support*6, n_leg*3)) : constraint matrix.
            c (array(n_support*6) : RHS of the constraint.
            n_support (int) : number of supporting legs.
    */
    C.setZero();
    c.setZero();

    int cnt = 0;
    for(int i = 0; i < n_leg; i++)
    {
        if (support_state[i] >= 0.7)
        {
            C.block<6, 3>(cnt*6, i*3) = Cf_support;
            c.block<6, 1>(cnt*6, 0)   = cf_support;
            cnt++;
        }
    }
    n_support = cnt;
}

void QuadConvexMPC::cal_swing_force_constraints(
    const Eigen::Matrix<double, n_leg, 1>& support_state,
    Eigen::Matrix<double, n_leg*3, n_leg*3>& D,
    Eigen::Matrix<double, n_leg*3, 1>& d,
    int& n_swing)
{
    /*
        Generate equality constraints for this moment to restrict the 
        forces of swing legs to be zero,
                            D f = d
        where f in R(3*n_leg x 1) is the leg tip force in WCS of this moment,
                D in R(3*n_swing x 3*n_leg) is the constraint matrix,
                d = 0 in R(3*n_swing x 1) is the RHS of the constraint.

        Parameters:
            support_state (array(n_leg)) :
                Leg support state, 1=support, 0=swing.
        
        Returns:
            D (array(n_swing*3, n_leg*3)) : constraint matrix.
            d (array(n_swing*3) : RHS of the constraint.
            n_swing (int) : number of swinging legs.

    */
    D.setZero();
    d.setZero();

    int cnt = 0;
    for(int i = 0; i < n_leg; i++)
    {
        if (support_state[i] >= 0.7) continue;
        
        D.block<3, 3>(cnt*3, i*3) = D_swing;
        d.block<3, 1>(cnt*3, 0)   = d_swing;
        cnt++;
    }
    n_swing = cnt;
}

void QuadConvexMPC::solve(Eigen::Matrix<double, dim_u, 1>& u_mpc)
{
    /*
        Solve MPC problem for quadruped robot, must be called after all
        matrices have been updated. 

        Returns:
            u_mpc(array(n_leg*3)) :
                predicted optimal input of the system, i.e. the leg tip force in WCS.
    */
    // build H and G
    H = Rbar + Bbar.transpose() * Qbar * Bbar;
    G = (Abar * x0 - X_ref).transpose() * Qbar * Bbar;

    Eigen::MatrixXd solution;
    solution.resize(HZ * dim_u, 1);
    qp_solver.solve_qp(H, G, Dbar, dbar, Cbar, cbar, solution, back_solver);
    u_mpc = solution.topRows(dim_u);
}

void QuadConvexMPC::reduce_solve(Eigen::Matrix<double, dim_u, 1>& u_mpc)
{
    /*
        Reduce the scale of MPC problem and solve, must be called after all matrices have
            been updated. 
        Since for the swinging legs, the leg tip forces are always zero, we can remove these
        optimizing variables from the MPC problem and reduce the sizes of Bbar, Rbar and Cbar.
        Moreover, the equality constraints for the forces of swinging legs are also removed. In 
        this way, the solving performance can be largely improved. 
        
        Returns:
            u_mpc(array(n_leg*3, horizon_length)) :
                predicted optimal input of the system, i.e. the leg tip force in WCS.
    */
    // select all non-zero inputs, which are the tip forces of supporting legs
    u_idx.setConstant(-1);
    int n_support = 0;

    for (int i = 0; i < HZ; i++)
    {
        for (int leg = 0; leg < n_leg; leg++)
        {
            if (support_state_seq[i](leg, 0) > 0.7)
            {
                u_idx[n_support * 3 + 0] = i * dim_u + leg * 3 + 0;
                u_idx[n_support * 3 + 1] = i * dim_u + leg * 3 + 1;
                u_idx[n_support * 3 + 2] = i * dim_u + leg * 3 + 2;
                n_support++;
            }
        }
    }

    if (total_support_legs == 0)
    {
        u_mpc.setZero();
        return;
    }

    Breduce.resize(HZ*dim_s, 3*total_support_legs);
    Rreduce.resize(3*total_support_legs, 3*total_support_legs);
    Rreduce.setZero();
    Creduce.resize(6*total_support_legs, 3*total_support_legs);

    for (int i = 0; i < total_support_legs; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            Breduce.col(i*3+j)    = Bbar.col(u_idx[i*3+j]);
            Rreduce(i*3+j, i*3+j) = Rbar(u_idx[i*3+j], u_idx[i*3+j]);
            Creduce.col(i*3+j)    = Cbar.col(u_idx[i*3+j]);
        }
    }

    // build H and G
    // Hreduce.resize(3*total_support_legs, 3*total_support_legs);
    // Greduce.resize(1, 3*total_support_legs);
    Hreduce = Rreduce + Breduce.transpose() * Qbar * Breduce;
    Greduce = (Abar * x0 - X_ref).transpose() * Qbar * Breduce;

    // solve
    Eigen::MatrixXd solution;
    solution.resize(3*total_support_legs, 1);
    qp_solver.solve_qp(Hreduce, Greduce, empty, empty, Creduce, cbar, solution, back_solver);
    
    // map back results
    u_mpc.setZero();
    for(int i = 0; i < dim_u && u_idx[i] < dim_u; i++)
    {
        u_mpc(u_idx[i], 0) = solution(i, 0);
    }
}
