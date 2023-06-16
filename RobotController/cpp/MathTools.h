#include <Eigen/Dense>

namespace MathTool
{
    inline Eigen::Matrix3d skew(const Eigen::Vector3d& r)
    {
        Eigen::Matrix3d rHat;
        rHat <<    0, -r(2),  r(1),
                r(2),     0, -r(0),
               -r(1),  r(0),     0;
        return rHat;
    }

    // Convert q = (qw, qx, qy, qz) to RPY euler angles (roll, pitch, yaw)
    inline Eigen::Vector3d quat2rpy(double qw, double qx, double qy, double qz)
    {
        double roll = atan2(qw*qx + qy*qz, 0.5 - qx*qx - qy*qy);
        double pitch = asin(-2.0 * (qx*qz - qw*qy));
        double yaw  = atan2(qx*qy + qw*qz, 0.5 - qy*qy - qz*qz);
        return Eigen::Vector3d(roll, pitch, yaw);
    }

    inline Eigen::Vector3d quat2rpy(const Eigen::Quaterniond& q)
    {
        return quat2rpy(q.w(), q.x(), q.y(), q.z());
    }
}

