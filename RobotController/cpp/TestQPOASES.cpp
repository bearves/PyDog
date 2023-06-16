#include <stdlib.h>
#include <cstdio>
#include <qpOASES.hpp>

int main(int argc, char **argv) {
    USING_NAMESPACE_QPOASES;

	/* Setup data of first QP. */
	real_t H[2*2] = { 4.0, 1.0, 1.0, 2.0 };
	real_t g[2] = { 1.0, 1.0 };
	real_t lb[2] = { -1e6, -1e6 };
	real_t ub[2] = {  1e6,  1e6 };
    real_t A[3*2] = { 1.0, 1.0, 1.0, 0.0, 0.0, 1.0 };
	real_t lbA[3] = {  1.0, 0.0, 0.0 };
	real_t ubA[3] = {  1.0, 0.7, 0.7 };

	/* Setup data of second QP. */
	real_t g_new[2] = { 1.0, 1.5 };
	real_t lb_new[2] = { -20.0, 12.0 };
	real_t ub_new[2] = { -25.0, 10.5 };
	real_t lbA_new[3] = { -2.0, 0.0, 0.0 };
	real_t ubA_new[3] = {  1.0, 0.7, 0.7 };


	/* Setting up QProblem object. */
	QProblem example( 2,3 );

	Options options;
	example.setOptions( options );

	/* Solve first QP. */
	int_t nWSR = 10;
	example.init( H,g,A,lb,ub,lbA,ubA, nWSR );

	/* Get and print solution of first QP. */
	real_t xOpt[2];
	real_t yOpt[2+1];
	example.getPrimalSolution( xOpt );
	example.getDualSolution( yOpt );
	printf( "\nxOpt = [ %e, %e ];  yOpt = [ %e, %e, %e ];  objVal = %e\n\n", 
			xOpt[0],xOpt[1],yOpt[0],yOpt[1],yOpt[2],example.getObjVal() );
	
	/* Solve second QP. */
	//nWSR = 10;
	//example.hotstart( g_new,lb_new,ub_new,lbA_new,ubA_new, nWSR );

	/* Get and print solution of second QP. */
	//example.getPrimalSolution( xOpt );
	//example.getDualSolution( yOpt );
	//printf( "\nxOpt = [ %e, %e ];  yOpt = [ %e, %e, %e ];  objVal = %e\n\n", 
    //			xOpt[0],xOpt[1],yOpt[0],yOpt[1],yOpt[2],example.getObjVal() );

	//example.printOptions();
	/*example.printProperties();*/

	/*getGlobalMessageHandler()->listAllMessages();*/

	return 0;
    
};