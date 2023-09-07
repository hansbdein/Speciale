#include "src/include.h"

/*-------------------------------------------------------- */
/* Calculation of the abundance of the elements from BBN   */
/*-------------------------------------------------------- */

int main(int argc,char** argv)
{
	int failsafe;
	
	if(argc<2) 
  	{ 
    		printf(" This program needs 1 parameter:\n"
           	"   failsafe    0     = fast\n"
           	"               1...3 = more precise, stiff method\n"
           	"               5...7 = stiff method with precision tests (5=tolerance of 5%%, 6=tolerance of 1%%, 7=tolerance of 0.1%%)\n"
           	"               10...12 = RK4 method with adaptative stepsize (10=5%%, 11=1%%, 12=0.1%%)\n"
           	"               20...22 = Fehlberg RK4-5 method (20=5%%, 21=1%%, 22=0.1%%)\n"
           	"               30...32 = Cash-Karp RK4-5 method (30=1%%, 31=10^-4, 32=10^-5)\n");
      		exit(1); 
  	} 
	else 
  	{
  		sscanf(argv[1],"%d",&failsafe);
  	}
  	
	struct relicparam paramrelic;
	double ratioH[NNUC+1],cov_ratioH[NNUC+1][NNUC+1];
	double H2_H,He3_H,Yp,Li7_H,Li6_H,Be7_H;
	double sigma_H2_H,sigma_He3_H,sigma_Yp,sigma_Li7_H,sigma_Li6_H,sigma_Be7_H;
	
	Init_cosmomodel(&paramrelic);
	
	paramrelic.failsafe=failsafe;
	
	printf("\t Yp\t\t H2/H\t\t He3/H\t\t Li7/H\t\t Li6/H\t\t Be7/H\n");

	paramrelic.err=0;
	nucl(&paramrelic,ratioH);
	H2_H=ratioH[3];Yp=ratioH[6];Li7_H=ratioH[8];Be7_H=ratioH[9];He3_H=ratioH[5];Li6_H=ratioH[7];
	printf(" cent:\t %.3e\t %.3e\t %.3e\t %.3e\t %.3e\t %.3e\n",Yp,H2_H,He3_H,Li7_H,Li6_H,Be7_H); 
	
	return 1;
}
