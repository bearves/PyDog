#include "QPSolver.h"
#include <qpOASES.hpp>
#include <QuadProg++/Array.hh>
#include <QuadProg++/QuadProg++.hh>
#include <iostream>

namespace LocoMpc
{
    QPSolver::QPSolver() {};

    void QPSolver::solve_qp(
        const Eigen::MatrixXd& HH,
        const Eigen::MatrixXd& gg,
        const Eigen::MatrixXd& CE,
        const Eigen::MatrixXd& ce,
        const Eigen::MatrixXd& CI,
        const Eigen::MatrixXd& ci,
        Eigen::MatrixXd& solution,
        QPSolver::SOLVER solver)
    {
        switch (solver)
        {
        case SOLVER::QPOASES:
            solve_qpoases(HH, gg, CE, ce, CI, ci, solution); 
            break;

        case SOLVER::QUADPROG:
            solve_quadprog(HH, gg, CE, ce, CI, ci, solution);
            break;
        
        default:
            break;
        }
    }

    void QPSolver::solve_qpoases(
        const Eigen::MatrixXd& HH,
        const Eigen::MatrixXd& gg,
        const Eigen::MatrixXd& CE,
        const Eigen::MatrixXd& ce,
        const Eigen::MatrixXd& CI,
        const Eigen::MatrixXd& ci,
        Eigen::MatrixXd& solution)
    {
        using namespace qpOASES;
        Eigen::MatrixXd A, At;
        Eigen::MatrixXd lbA;
        Eigen::MatrixXd ubA;

        int nv = gg.size();
        int nc = CI.rows() + CE.rows();

        A.resize(nc, nv); 
        At.resize(nv, nc); // Eigen use col-major storage in default, but qpOASES wants row-major matrix here
        lbA.resize(nc, 1);
        ubA.resize(nc, 1);

        A.block(0, 0, CI.rows(), nv) = CI;
        A.block(CI.rows(), 0, CE.rows(), nv) = CE;
        Eigen::MatrixXd ci_copy = ci;
        ci_copy.setConstant(-1e15);
        lbA.topRows(CI.rows()) = ci_copy;
        lbA.bottomRows(CE.rows()) = ce;
        ubA.topRows(CI.rows()) = ci;
        ubA.bottomRows(CE.rows()) = ce;

        At = A.transpose();

        /* Setting up QProblem object with n_variables and n_constraints*/
        QProblem problem(nv, nc, HST_POSDEF);

        Options options;
        options.setToMPC();
        options.printLevel = PL_LOW;
        problem.setOptions( options );

        /* Solve first QP. */
        int_t nWSR = 200; // number of active sets 
        problem.init( HH.data(), gg.data(), At.data(), nullptr, nullptr, lbA.data(), ubA.data(), nWSR );

        /* Get and print solution of first QP. */
        problem.getPrimalSolution( solution.data() );
    }


    void QPSolver::solve_quadprog(
        const Eigen::MatrixXd& HH,
        const Eigen::MatrixXd& gg,
        const Eigen::MatrixXd& CE,
        const Eigen::MatrixXd& ce,
        const Eigen::MatrixXd& CI,
        const Eigen::MatrixXd& ci,
        Eigen::MatrixXd& solution)
    {
        quadprogpp::Matrix<double> H_qpp(HH.data(), HH.cols(), HH.rows());
        quadprogpp::Vector<double> g_qpp(gg.data(), gg.size());
        quadprogpp::Matrix<double> CE_qpp(CE.data(), gg.size(), CE.rows());
        quadprogpp::Vector<double> ce_qpp(ce.data(), ce.size());
        quadprogpp::Matrix<double> CI_qpp(CI.data(), gg.size(), CI.rows());
        quadprogpp::Vector<double> ci_qpp(ci.data(), ci.size());
        quadprogpp::Vector<double> x(gg.size());

        quadprogpp::solve_quadprog(H_qpp, g_qpp, CE_qpp, ce_qpp, -CI_qpp, ci_qpp, x);

        for (int i = 0; i < gg.size(); i++)
        {
            solution(i, 0) = x[i];
        }
    }

}