cmake_minimum_required(VERSION 3.10)

project(RobotMPCLib)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RELEASE)

set(EIGEN_INC_DIR "C:\\Users\\Sun\\miniconda3\\envs\\pydog\\Library\\include\\eigen3")
set(QPOASES_INC_DIR ${PROJECT_SOURCE_DIR}\\qpOASES\\qpOASES_lib\\include)
set(QPOASES_LIB_DIR ${PROJECT_SOURCE_DIR}\\qpOASES\\qpOASES_lib\\lib)
set(QUADPROG_INC_DIR ${PROJECT_SOURCE_DIR}\\quadprog\\quadprog_lib\\include)
set(QUADPROG_LIB_DIR ${PROJECT_SOURCE_DIR}\\quadprog\\quadprog_lib\\lib)

include_directories(${EIGEN_INC_DIR} ${QPOASES_INC_DIR} ${QUADPROG_INC_DIR})
link_directories(${QPOASES_LIB_DIR} ${QUADPROG_LIB_DIR})

set(MPC_LIB_SOURCE RobotMPC.cpp QPSolver.cpp)

add_library(locoMpc ${MPC_LIB_SOURCE})
target_link_libraries(locoMpc qpOASES quadprog)

add_executable(testMPC TestMPC.cpp)
target_link_libraries(testMPC locoMpc)

add_executable(testQPOASES TestQPOASES.cpp)
target_link_libraries(testQPOASES qpOASES)