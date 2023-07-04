#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include "RobotMPC.h"

namespace py = pybind11;

class QuadConvexMPCWrapper
{
public:
    QuadConvexMPCWrapper(double dt, int leap) : mpc_{dt, leap}
    {
        mpc_.set_back_solver(LocoMpc::QPSolver::SOLVER::QPOASES);
    }

    bool need_solve(int count)
    {
        return mpc_.need_solve(count);
    }

    void cal_weight_matrices()
    {
        mpc_.cal_weight_matrices();
    }

    Eigen::MatrixXd solve(
        const Eigen::MatrixXd &body_euler_seq,
        const Eigen::MatrixXd &r_leg_seq,
        const Eigen::MatrixXd &support_state_seq,
        const Eigen::MatrixXd &x_ref_seq,
        const Eigen::MatrixXd &x_act)
    {
        Eigen::Matrix<double, mpc_.HZ, 3> body_euler_list;
        Eigen::Matrix<double, mpc_.HZ, mpc_.n_leg * 3> r_leg_list;
        Eigen::Matrix<double, mpc_.HZ, mpc_.n_leg> support_state_list;
        Eigen::Matrix<double, mpc_.HZ, mpc_.dim_s> x_ref_list;
        Eigen::Matrix<double, mpc_.dim_s, 1> x0;
        Eigen::Matrix<double, mpc_.dim_u, 1> u_mpc_reduce;
        Eigen::Matrix<double, mpc_.dim_u, 1> u_mpc;

        body_euler_list = body_euler_seq;
        r_leg_list = r_leg_seq;
        support_state_list = support_state_seq;
        x0 = x_act;
        x_ref_list = x_ref_seq;
        
        mpc_.update_mpc_matrices(
            body_euler_list,
            r_leg_list,
            support_state_list,
            x0,
            x_ref_list);
        mpc_.update_super_matrices();
        //mpc_.solve(u_mpc);
        mpc_.reduce_solve(u_mpc_reduce);

        return u_mpc_reduce;
    }

    int horizon_length()
    {
        return mpc_.HZ;
    }
    int dim_s()
    {
        return mpc_.dim_s;
    }
    int dim_u()
    {
        return mpc_.dim_s;
    }
    int n_leg()
    {
        return mpc_.n_leg;
    }
    double dt_mpc()
    {
        return mpc_.dt_mpc;
    }

private:
    QuadConvexMPC mpc_;
};

PYBIND11_MODULE(robotMPC_pb, m)
{
    m.doc() = "RobotMPC bindings";
    py::class_<QuadConvexMPCWrapper, std::shared_ptr<QuadConvexMPCWrapper>>(m, "QuadConvexMPCWrapper")
        .def(py::init<double, int>())
        .def("need_solve", &QuadConvexMPCWrapper::need_solve)
        .def("cal_weight_matrices", &QuadConvexMPCWrapper::cal_weight_matrices)
        .def("solve", &QuadConvexMPCWrapper::solve)
        .def("horizon_length", &QuadConvexMPCWrapper::horizon_length)
        .def("dim_s", &QuadConvexMPCWrapper::dim_s)
        .def("dim_u", &QuadConvexMPCWrapper::dim_u)
        .def("n_leg", &QuadConvexMPCWrapper::n_leg)
        .def("dt_mpc", &QuadConvexMPCWrapper::dt_mpc);
}
