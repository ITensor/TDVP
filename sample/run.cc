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

    auto ampo = AutoMPO(sites);

	// chain
	for(int j = 1; j < N; ++j)
		{
		ampo += 0.5,"S+",j,"S-",j+1;
		ampo += 0.5,"S-",j,"S+",j+1;
        ampo +=     "Sz",j,"Sz",j+1;
		}

	auto H = toMPO(ampo);
	printfln("Maximum bond dimension of H is %d",maxLinkDim(H));

    auto state = InitState(sites);
    for(int i = 1; i <= N; ++i) 
        {
        if(i%2 == 1)
            state.set(i,"Up");
        else
            state.set(i,"Dn");
        }
	auto psi1 = MPS(state);
	auto psi2 = MPS(state);

    printfln("Initial energy = %.5f", inner(psi1,H,psi1) );

    auto sweeps = Sweeps(5);
    sweeps.maxdim() = 2000;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 10;
    println(sweeps);

	println("----------------------------------------TDVP---------------------------------------");

	auto energy = tdvp(psi1,H,-t,sweeps,{"DoNormalize",true,"IsHermitian",true,"Quiet",true,"NumCenter",2});

	printfln("\nEnergy after imaginary time evolution = %.10f",energy);
    printfln("\nUsing overlap = %.10f", inner(psi1,H,psi1) );

	println("----------------------------------------Zaletel---------------------------------------");

	auto expH = toExpH(ampo,t0);
	printfln("Maximum bond dimension of expH is %d",maxLinkDim(expH));
	auto args = Args("Method=","DensityMatrix","Cutoff=",1E-10,"MaxDim=",2000);
	for(int n = 1; n <= 5*(t/t0); ++n)
		{
		psi2 = applyMPO(expH,psi2,args);
		psi2.noPrime().normalize();
		if(n%int(t/t0) == 0)
			{
			printfln("Maximum bond dimension at time %.1f is %d ", n*t0, maxLinkDim(psi2));
			printfln("\nEnergy using overlap at time %.1f is %.10f", n*t0, inner(psi2,H,psi2) );
			}
		}

    return 0;
    }
