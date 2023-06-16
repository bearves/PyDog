#include <Eigen/Dense>
#include <iostream>

namespace LocoMpc
{

class QPSolver
{
public:

    enum SOLVER 
    {
        QPOASES  = 0,
        QUADPROG = 1
    };

    QPSolver();

    void solve_qp(
        const Eigen::MatrixXd& HH,
        const Eigen::MatrixXd& gg,
        const Eigen::MatrixXd& CE,
        const Eigen::MatrixXd& ce,
        const Eigen::MatrixXd& CI,
        const Eigen::MatrixXd& ci,
        Eigen::MatrixXd& solution,
        SOLVER solver = QPOASES);

private:
    void solve_qpoases(
            const Eigen::MatrixXd& HH,
            const Eigen::MatrixXd& gg,
            const Eigen::MatrixXd& CE,
            const Eigen::MatrixXd& ce,
            const Eigen::MatrixXd& CI,
            const Eigen::MatrixXd& ci,
            Eigen::MatrixXd& solution);

    void solve_quadprog(
            const Eigen::MatrixXd& HH,
            const Eigen::MatrixXd& gg,
            const Eigen::MatrixXd& CE,
            const Eigen::MatrixXd& ce,
            const Eigen::MatrixXd& CI,
            const Eigen::MatrixXd& ci,
            Eigen::MatrixXd& solution);
};

}