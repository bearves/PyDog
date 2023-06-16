#include "RobotMPC.h"
#include "MathTools.h"
#include <iostream>
#include <fstream>
#include <chrono>

int main(int argc, char** argv)
{
    using namespace Eigen;
    using namespace std;
    using clk = std::chrono::high_resolution_clock;

    srand(time(nullptr));
    QuadConvexMPC mpc(0.001, 25);
    //mpc.set_back_solver(LocoMpc::QPSolver::QUADPROG);
    mpc.set_back_solver(LocoMpc::QPSolver::QPOASES);

    Vector3d scale(3.14159265, 3.14159265/2, 3.14159265);
    //Vector3d rpy = Vector3d::Random().cwiseProduct(scale);
    Vector3d rpy(0.0, 0.0, 0.1);
    AngleAxisd rx(rpy[0], Vector3d::UnitX());
    AngleAxisd ry(rpy[1], Vector3d::UnitY());
    AngleAxisd rz(rpy[2], Vector3d::UnitZ());

    AngleAxisd body_rot(rz * ry * rx);
    Quaterniond body_quat(body_rot);
    Vector3d body_rpy = MathTool::quat2rpy(body_quat);

    Vector3d body_pos(0, 0, 0.3);
    Matrix<double, mpc.n_leg*3, 1> r_leg;
    r_leg << 0.183, -0.047, 0,
             0.183,  0.047, 0,
            -0.183, -0.047, 0,
            -0.813,  0.047, 0;

    // cout << rpy << "\n--------------\n" << body_rpy << '\n';

    Matrix<double, mpc.dim_s, mpc.dim_s> A;
    Matrix<double, mpc.dim_s, mpc.dim_u> B;
    mpc.cal_state_equation(body_pos, body_rpy, r_leg, A, B);
    
    //cout << "A = \n";
    //cout << A << endl;
    //cout << "B = \n";
    //cout << B << endl;

    int n_support, n_swing;
    Matrix<double, mpc.n_leg*6, mpc.dim_u> C;
    Matrix<double, mpc.n_leg*6, 1> c;
    Matrix<double, mpc.n_leg*3, mpc.dim_u> D;
    Matrix<double, mpc.n_leg*3, 1> d;
    mpc.cal_friction_constraint(Vector4d(0, 1, 0, 1), C, c, n_support);
    mpc.cal_swing_force_constraints(Vector4d(0, 1, 0, 1), D, d, n_swing);

    // cout << "n_spt = " << n_support << "\n"; 
    // cout << "C = \n";
    // cout << C << endl;
    // cout << "c = \n";
    // cout << c << endl;
    // cout << "n_swg = " << n_swing << "\n"; 
    // cout << "D = \n";
    // cout << D << endl;
    // cout << "d = \n";
    // cout << d << endl;

    Eigen::Matrix<double, mpc.HZ, 3> body_euler_list;
    Eigen::Matrix<double, mpc.HZ, mpc.n_leg * 3> r_leg_list;
    Eigen::Matrix<double, mpc.HZ, mpc.n_leg> support_state_list;
    Eigen::Matrix<double, mpc.HZ, mpc.dim_s> x_ref_list;

    body_euler_list.setZero();
    r_leg_list.setZero();
    support_state_list.setZero();
    x_ref_list.setZero();

    Matrix<double, mpc.dim_s, 1> x0;
    Eigen::Matrix<double, mpc.dim_u, 1> u_mpc, u_mpc_reduce;

    auto begin = clk::now();

    int total_loop = 1000;
    for (int n = 0; n < total_loop; n++)
    {
        x0 << 0,0.02,0, // R, P, Y
            0,0,0,    // x, y, z
            0,0,0,    // wx, wy, wz
            0,0,0,    // vx, vy, vz
            -9.81;    // gravity

        for (int i = 0; i < mpc.HZ; i++)
        {
            body_euler_list.row(i) = body_rpy;
            r_leg_list.row(i) << 0.183, -0.047, 0,
                                0.183,  0.047, 0,
                                -0.183, -0.047, 0,
                                -0.183,  0.047, 0;
            support_state_list.row(i) << 1,0,0,1;
            x_ref_list.row(i) << 0,0,0,
                                0,0,0,
                                0,0,0,
                                0,0,0,
                                -9.81;
        }

        mpc.cal_weight_matrices();
        mpc.update_mpc_matrices(body_euler_list, r_leg_list, support_state_list, x0, x_ref_list);
        mpc.update_super_matrices();
        
        mpc.reduce_solve(u_mpc_reduce);
        //mpc.solve(u_mpc);
    }

    auto end = clk::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cout << "Result:\n"; 
    cout << u_mpc << "\n";
    cout << u_mpc_reduce << "\n";

    cout << "Time used per loop: " << elapsed.count() * 1e-6 / total_loop << "ms\n";

    return 0;
}