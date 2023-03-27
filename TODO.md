1. MPC basic study (DONE)
    - MPC of CartPole
    - MPC of 1-D floating base, double input, with support switching
    - MPC of 6-D floating base, 12/18 input (configurable)
2. Dynamic simulation environment setup (DONE)
    - Pybullet installation and setup
    - Create a quadruped robot model A1 from urdf
    - Get body states and joint states as feedback
3. Joint PD controller (DONE)
4. MPC controller for quadruped robot (DONE)
    - Build continuous and discrete dynamics of the floating base
    - Create super matrices of the MPC problem
        - Abar, Bbar
        - Equation constraints: Du = d, based on support states
            - F_swing = 0
        - Inequality constrants: Cu = d, based on friction cone
            - F_stance_xy < mu * F_stance_z
            - F_stance_z > 0 && F_stance_z < fzmax
    - Init static MPC inputs
        - dt, dtmpc
        - Qk, Rk
    - Predict time varying MPC inputs
        - support state sequence
        - footholds sequence
        - body euler sequence (derived from below)
        - body reference pose sequence
        - current actual body pose
    - Solve QP and get Fr as the tip force for stance legs
    - tau = Rbody * J_leg * Fr

5. Gait scheduler (DONE)
    - Duration & offset definition
    - Predict support state sequence

6. Body trajectory planner (DONE)
    - Get cmd from RobotSteer
    - Generate body ref pose and euler sequence by integration

7. Footholds planner (DONE)
    - Hold last, current footholds
    - If lft, last footholds <- current leg tip pos
    - If td, current footholds <- current leg tip pos
    - Keep calculating next footholds no matter the leg is swinging or not
        - Shoulder position prediction
        - Raibert law footholds prediction
        - Quick spin correction, aka. the contrifugal term (DONE)
    - Give prediction of future tip pos

8. Leg trajectory planner (DONE)
    - Leg kinematics
        - Leg FK, IK and Jac in Leg CS
    - Interpolation from last footholds to next footholds using sin/cos curve (DONE)
    - Interpolation using bezier curve or spline （DONE）
    - Get tip ref pos, ref vel in WCS
    - Tip ref accel in WCS

9. Floating base dynamics for quadruped robot (DONE)
    - Setup pinocchino dynamic engine
    - Write robot dynamics

10. Whole body impluse control
    - Task Execution 
        - Task specific: (DONE)
            - Body ori
            - Body pos
            - Tip pos of swinging legs
        - Get body ref ori, pos, omega, vel, tip ref pos, vel, accel (DONE)
        - Get body actual ori, pos, omega, vel, tip actual pos, vel and update model (DONE)
        - Calculate state error (DONE)
        - Get Jc of each task from Dynamics, where vtip = Jc q (DONE) 
        - Calculate Nc = I - pinv(Jc)Jc, let N0 be Nc thus the task decomposition (DONE) 
            can take consideration of contact constrains in the kinematic chain computation
        - Get Ji and Ji^dot q_{i-1}^dot of each task from Dynamics (DONE) 
        - Calculate deltaq, q_des = q_act + deltaq (DONE) 
        - Calculate qdot_des and qddot_des (DONE) 
        - Find out joint target pos and vel, feeding to PD controller (NEED VERIFY AND REFRACTORING)
    - QP to find out joint feedforward trq
        - Set weights  (DONE)
        - Get Fr from MPC (DONE)
        - Get qddot_des from Task Execution (DONE) 
        - Get A, b, g (DONE)
        - Get friction cone constrains just like MPC (DONE)
        - Solve QP and get final fr and qddot (DONE)
        - Calculate tau as Aqddot + b + g - Jc.T fr, and get final joint ff trq (DONE) 
    - FIX: Qj error? (DONE)
        - Reason: Actual body pos and body_ref_pos has big different 
        - checkout why?
          - The body ref pos is integrated by a very rough way, which has large drift, it should be updated to align with the actual body pos to compensate the drift.
    - TODO: FIX: Non smooth joint pos trajectory
        - Reason: Late touchdown?
    - FIX: When turn 180 deg/large angle, the whole robot lost control (DONE)
        - Reason: wrong set to WBIC's body ref orn 
    - FIX: The wbic may cause the leg's knee/shoulder joint to flip 
        - Reason: multiple inverse solution of leg
        - fix: check if the joint position go beyond the allowable range

11. Change simulator and re-write simulation codes (DONE)
    - Setup Webots simulator interface
    - Setup accelerometer and gyro sensor

12.  Early/Late touchdown detection and reprogramming
    - For each leg, if always early/late touchdown, slowly adjust foothold height accordingly.
    - Sudden early touchdown, replan the leg trajectory.

13.  Body state estimator
    - Implement ETH's state estimator (DONE)
      - Extended Kalman Filter formulation(DONE)
    - Collect data for state estimation turning(DONE)
      - Measurements: time, accelerometer, gyro, joint pos, vel, trq, support state, phase of support state
      - True data: body pos, vel, orn, angvel
    - Turning(DONE)
      - Qbw, Qbf
      - Qw, Qf
      - Rs, Qpst, Qpsw
    - Integrate state estimation to gait controller 
      - Integrating(DONE)
      - Add log for estimated data analysis(DONE)
      - Modify foothold height according to body state and ground height(Done)
    - Simplified body state estimator (MIT's version)
      - Remove estimation of body orientation, just use IMU's output (Done)

14.  Terrain slope estimation and body adaption
    - Implement ground plane estimator (DONE)
    - Modify foothold height according to body state and ground height(Done)
    - Modify body height according to body state and ground height(Done)
    - Modify body orientation and future trajectory according to ground slope (Done)
    - Adjust reference body height according to the ground slope (Ongoing)

