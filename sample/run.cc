#include "itensor/all.h"
#include "tdvp.h"

using namespace itensor;

int main()
    {
    int N = 16;
    Real t = 0.1;
    Real t0 = 0.01;

    auto sites = SpinHalf(N); //make a chain of N spin 1/2's
    //auto sites = SpinOne(N); //make a chain of N spin 1's
    
	auto state = InitState(sites);
    for(int i = 1; i <= N; ++i) 
        {
        if(i%2 == 1) state.set(i,"Up");
        else state.set(i,"Dn");
        }
    auto psi1 = MPS(state);
    auto psi2 = psi1;


    // Artificially increase bond dimension by DMRG
	printfln("-------------------------------------DMRG warm-up----------------------------");
    auto sweeps0 = Sweeps(1);
    sweeps0.maxdim() = 100;
    sweeps0.cutoff() = 0;
    sweeps0.niter() = 4;
    
	double a = 1.0;

	for(int i = 1; i <= 11; ++i)
		{
		auto ampo1 = AutoMPO(sites);
    	for(int j = 1; j < N; ++j)
        	{
        	ampo1 += 0.5*a,"S+",j,"S-",j+1;
        	ampo1 += 0.5*a,"S-",j,"S+",j+1;
        	ampo1 +=     a,"Sz",j,"Sz",j+1;
        	}
		auto H1 = toMPO(ampo1);

		auto ampo2 = AutoMPO(sites);
		for(int j = 1; j <= N; ++j)
			{
			if(j%2 == 1) ampo2 += -1,"Sz",j;
			else ampo2 += "Sz",j;
			}
		auto H2 = toMPO(ampo2);

		auto Hset = std::vector<MPO>(2);
		Hset.at(0) = H1;
		Hset.at(1) = H2;

		dmrg(psi1,Hset,sweeps0,{"Quiet",true});

		a /= 10;
		}
	for(int i = 1; i <= 5; ++i)
		{
		auto ampo2 = AutoMPO(sites);
		for(int j = 1; j <= N; ++j)
			{
			if(j%2 == 1) ampo2 += -1,"Sz",j;
			else ampo2 += "Sz",j;
			}
		auto H2 = toMPO(ampo2);

		dmrg(psi1,H2,sweeps0,{"Quiet",true});
		}

	printfln("Check spin configuration after DMRG warm-up");
	for(int j = 1; j <= N; ++j)
        {
        psi1.position(j);
        Real szj = std::real((psi1.A(j) * sites.op("Sz",j) * dag(prime(psi1.A(j),"Site"))).cplx());
        printfln("%d %.10f",j,szj);
        }
    psi1.position(1);

	// start TDVP, either one site and two site algorithm can be used by adjust the "NumCenter" argument
    println("----------------------------------------TDVP---------------------------------------");
	auto ampo = AutoMPO(sites);
    for(int j = 1; j < N; ++j)
        {
        ampo += 0.5,"S+",j,"S-",j+1;
        ampo += 0.5,"S-",j,"S+",j+1;
        ampo +=     "Sz",j,"Sz",j+1;
        }
    auto H = toMPO(ampo);
    printfln("Maximum bond dimension of H is %d",maxLinkDim(H));

    printfln("Initial energy = %.5f", inner(psi1,H,psi1) );

    auto sweeps = Sweeps(5);
    sweeps.maxdim() = 2000;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 10;
    println(sweeps);

    auto energy = tdvp(psi1,H,-t,sweeps,{"DoNormalize",true,
                                         "IsHermitian",true,
                                         "Quiet",true,
                                         "NumCenter",2});

    printfln("\nEnergy after imaginary time evolution = %.10f",energy);
    printfln("\nUsing overlap = %.10f", inner(psi1,H,psi1) );

    println("-------------------------------------Zaletel 2nd order---------------------------------------");

    auto expH1 = toExpH(ampo,(1-1_i)/2*t0);
	auto expH2 = toExpH(ampo,(1+1_i)/2*t0);
    printfln("Maximum bond dimension of expH1 is %d",maxLinkDim(expH1));
    auto args = Args("Method=","DensityMatrix","Cutoff=",1E-10,"MaxDim=",2000);
    for(int n = 1; n <= 5*(t/t0); ++n)
        {
        psi2 = applyMPO(expH1,psi2,args);
		psi2.noPrime();
		psi2 = applyMPO(expH2,psi2,args);
        psi2.noPrime().normalize();
        if(n%int(t/t0) == 0)
            {
            printfln("Maximum bond dimension at time %.1f is %d ", n*t0, maxLinkDim(psi2));
            printfln("\nEnergy using overlap at time %.1f is %.10f", n*t0, real(innerC(psi2,H,psi2)) );
            }
        }

    return 0;
    }
