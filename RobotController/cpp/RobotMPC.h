#ifndef ROBOT_MPC_H
#define ROBOT_MPC_H

#include <Eigen/Dense>
#include "QPSolver.h"

class QuadConvexMPC
{
public:
    //  Quadruped robot model predictive controller
    //  MPC settings
    static const int HZ = 20;                  // prediction horizon length of MPC
    int leap;                                  // Solve MPC every (leap) simulating step
    double dt;                                 // time step of the simulator
    double dt_mpc;                             // MPC solving time interval

    // qp solver
    LocoMpc::QPSolver qp_solver;
    LocoMpc::QPSolver::SOLVER back_solver;

    // System config
    static const int n_leg{4};
    static const int dim_s{13};                // dimension of system states
    static const int dim_u{n_leg * 3};         // dimension of system inputs

    // Dynamic constants
    Eigen::Matrix3d Ib; // = np.diag([0.07, 0.26, 0.242]) // mass inertia of floating base
    Eigen::Matrix3d invIb;
    double mb; // = 10                                // mass of the floating base
    double ground_fric; // = 0.4                      // ground Coulomb friction constant
    double fz_min; // = 0.1                           // minimum vertical force for supporting leg
    double fz_max; // = 150                           // maximum vertical force for supporting leg

    // Friction constraints
    Eigen::MatrixXd Cf_support;
    Eigen::MatrixXd cf_support;
    // Swing constraints
    Eigen::Matrix3d D_swing;
    Eigen::Vector3d d_swing;

    // MPC weights
    Eigen::MatrixXd Qk; // = np.diag([100,100,0,0,0,100,1,1,1,1,1,1,0]) 
    Eigen::MatrixXd Rk; // = 1e-6 * np.eye(dim_u)

    // MPC matrix lists
    Eigen::MatrixXd   Ak_list[HZ];    // = np.zeros((HZ, dim_s, dim_s))
    Eigen::MatrixXd   Bk_list[HZ];    // = np.zeros((HZ, dim_s, dim_u))
    Eigen::MatrixXd   Ck_list[HZ];    // = np.zeros((HZ, 6*n_leg, dim_u))
    Eigen::MatrixXd   ck_list[HZ];    // = np.zeros((HZ, 6*n_leg, 1))
    Eigen::MatrixXd   Dk_list[HZ];    // = np.zeros((HZ, 3*n_leg, dim_u))
    Eigen::MatrixXd   dk_list[HZ];    // = np.zeros((HZ, 3*n_leg, 1))
    int n_support_list[HZ]; // = np.zeros(HZ, dtype=int) // number of legs in supporting in the horizon
    int n_swing_list[HZ]; // = np.zeros(HZ, dtype=int)   // number of legs in swinging in the horizon

    // system ref states and actual states
    Eigen::MatrixXd x0; // = np.zeros(dim_s)  // current state
    Eigen::MatrixXd x_ref_seq[HZ]; // = np.zeros((HZ, dim_s)) // future reference states
    Eigen::MatrixXd X_ref; // = np.zeros(HZ * dim_s)      // flattened future reference states
    Eigen::MatrixXd support_state_seq[HZ]; // = np.zeros((HZ, n_leg))  // future leg support states

    // MPC super matrices
    Eigen::MatrixXd Abar; // = np.zeros((HZ*dim_s, dim_s))
    Eigen::MatrixXd Bbar; // = np.zeros((HZ*dim_s, HZ*dim_u))
    Eigen::MatrixXd Cbar; // = np.array([])
    Eigen::MatrixXd cbar; // = np.array([])
    Eigen::MatrixXd Dbar; // = np.array([])
    Eigen::MatrixXd dbar; // = np.array([])
    Eigen::MatrixXd Qbar; // = np.zeros((HZ*dim_s, HZ*dim_s))
    Eigen::MatrixXd Rbar; // = np.zeros((HZ*dim_u, HZ*dim_u))
    Eigen::MatrixXd H   ; // = np.array([])
    Eigen::MatrixXd G   ; // = np.array([])

    // reduced super matrices
    Eigen::Array<int, dim_u * HZ, 1> u_idx;
    Eigen::MatrixXd Breduce;
    Eigen::MatrixXd Creduce;
    Eigen::MatrixXd Rreduce;
    Eigen::MatrixXd Hreduce; // = np.array([])
    Eigen::MatrixXd Greduce; // = np.array([])

    // temp matrices
    Eigen::Matrix<double, dim_s, dim_s> A, Ak;
    Eigen::Matrix<double, dim_s, dim_u> B, Bk;
    Eigen::Matrix<double, 6*n_leg, dim_u> Ck;
    Eigen::Matrix<double, 6*n_leg, 1> ck;
    Eigen::Matrix<double, 3*n_leg, dim_u> Dk;
    Eigen::Matrix<double, 3*n_leg, 1> dk;
    int total_support_legs = 0;
    int total_swing_legs = 0;

    Eigen::Matrix<double, dim_s, dim_s> AT;
    Eigen::Matrix<double, dim_s, dim_u> BT;
    Eigen::Matrix<double, dim_s, dim_s> AT2;
    Eigen::Matrix<double, dim_s, dim_s> AT3;
    Eigen::Matrix<double, dim_s, dim_s> AT4;

    Eigen::MatrixXd empty;

public:
    QuadConvexMPC(double sim_dt = 0.001, int leap = 25);

    void set_back_solver(LocoMpc::QPSolver::SOLVER solver);
    void cal_weight_matrices();
    bool need_solve(int count);
    void update_mpc_matrices(
        const Eigen::Matrix<double, HZ, 3>& body_euler_seq,
        const Eigen::Matrix<double, HZ, n_leg * 3>& r_leg_seq,
        const Eigen::Matrix<double, HZ, n_leg>& support_state_seq,
        const Eigen::Matrix<double, dim_s, 1>& x_act,
        const Eigen::Matrix<double, HZ, dim_s>& x_ref_seq);
    void update_super_matrices();
    void cal_state_equation(    
        const Eigen::Vector3d& body_pos, 
        const Eigen::Vector3d& body_euler, 
        const Eigen::Matrix<double, dim_u, 1>& r_leg, 
        Eigen::Matrix<double, dim_s, dim_s>& A,
        Eigen::Matrix<double, dim_s, dim_u>& B);
    void discretized_state_equation(
        const Eigen::Matrix<double, dim_s, dim_s>& A, 
        const Eigen::Matrix<double, dim_s, dim_u>& B,
        Eigen::Matrix<double, dim_s, dim_s>& Ak,
        Eigen::Matrix<double, dim_s, dim_u>& Bk);
    void cal_friction_constraint(
        const Eigen::Matrix<double, n_leg, 1>& support_state,
        Eigen::Matrix<double, n_leg*6, n_leg*3>& C,
        Eigen::Matrix<double, n_leg*6, 1>& c, 
        int& n_support);
    void cal_swing_force_constraints(
        const Eigen::Matrix<double, n_leg, 1>& support_state,
        Eigen::Matrix<double, n_leg*3, n_leg*3>& D,
        Eigen::Matrix<double, n_leg*3, 1>& d,
        int& n_swing);
    void solve(Eigen::Matrix<double, dim_u, 1>& u_mpc);
    void reduce_solve(Eigen::Matrix<double, dim_u, 1>& u_mpc);
};

#endif