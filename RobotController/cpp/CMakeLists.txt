cmake_minimum_required(VERSION 3.10)

project(RobotMPCLib)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RELEASE)

set(EIGEN_INC_DIR "C:\\Users\\Sun\\miniconda3\\envs\\pydog\\Library\\include\\eigen3")
set(QPOASES_INC_DIR ${PROJECT_SOURCE_DIR}\\third_party\\qpOASES\\qpOASES_lib\\include)
set(QPOASES_LIB_DIR ${PROJECT_SOURCE_DIR}\\third_party\\qpOASES\\qpOASES_lib\\lib)
set(QUADPROG_INC_DIR ${PROJECT_SOURCE_DIR}\\third_party\\quadprog\\quadprog_lib\\include)
set(QUADPROG_LIB_DIR ${PROJECT_SOURCE_DIR}\\third_party\\quadprog\\quadprog_lib\\lib)

add_subdirectory(${PROJECT_SOURCE_DIR}\\third_party\\pybind11)

include_directories(${EIGEN_INC_DIR} ${QPOASES_INC_DIR} ${QUADPROG_INC_DIR})
link_directories(${QPOASES_LIB_DIR} ${QUADPROG_LIB_DIR})

set(MPC_LIB_SOURCE RobotMPC.cpp QPSolver.cpp)

add_library(locoMpc ${MPC_LIB_SOURCE})
target_link_libraries(locoMpc qpOASES quadprog)

add_executable(testMPC TestMPC.cpp)
target_link_libraries(testMPC locoMpc)

add_library(robotMPC_pb SHARED RobotMPCWrapper.cpp ${MPC_LIB_SOURCE})
target_link_libraries(robotMPC_pb qpOASES quadprog pybind11::module)
pybind11_extension(robotMPC_pb)
pybind11_strip(robotMPC_pb)
set_target_properties(robotMPC_pb PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden")