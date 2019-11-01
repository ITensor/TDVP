#ifndef __ITENSOR_TDVP_H
#define __ITENSOR_TDVP_H

#include "itensor/iterativesolvers.h"
#include "itensor/mps/localmposet.h"
#include "itensor/mps/sweeps.h"
#include "itensor/mps/DMRGObserver.h"
#include "itensor/util/cputime.h"


namespace itensor {

template <class LocalOpT>
Real
TDVPWorker(MPS & psi,
           LocalOpT& PH,
           Cplx t,
           const Sweeps& sweeps,
           const Args& args = Args::global());

template <class LocalOpT>
Real
TDVPWorker(MPS & psi,
           LocalOpT& PH,
           Cplx t,
           const Sweeps& sweeps,
           DMRGObserver & obs,
           Args args = Args::global());

//
// Available TDVP methods:
// second order integrator: sweep left-to-right and right-to-left
//

//
//TDVP with an MPO
//
Real inline
tdvp(MPS & psi, 
     MPO const& H,
     Cplx t, 
     const Sweeps& sweeps,
     const Args& args = Args::global())
    {
    LocalMPO PH(H,args);
    Real energy = TDVPWorker(psi,PH,t,sweeps,args);
    return energy;
    }

//
//TDVP with an MPO and custom DMRGObserver
//
Real inline
tdvp(MPS & psi, 
     MPO const& H, 
     Cplx t,
     const Sweeps& sweeps, 
     DMRGObserver & obs,
     const Args& args = Args::global())
    {
    LocalMPO PH(H,args);
    Real energy = TDVPWorker(psi,PH,t,sweeps,obs,args);
    return energy;
    }

//
//TDVP with an MPO and boundary tensors LH, RH
// LH - H1 - H2 - ... - HN - RH
//(ok if one or both of LH, RH default constructed)
//
Real inline
tdvp(MPS & psi, 
     MPO const& H, 
     Cplx t,
     ITensor const& LH, 
     ITensor const& RH,
     const Sweeps& sweeps,
     const Args& args = Args::global())
    {
    LocalMPO PH(H,LH,RH,args);
    Real energy = TDVPWorker(psi,PH,t,sweeps,args);
    return energy;
    }

//
//TDVP with an MPO and boundary tensors LH, RH
//and a custom observer
//
Real inline
tdvp(MPS & psi, 
     MPO const& H, 
     Cplx t,
     ITensor const& LH, 
     ITensor const& RH,
     const Sweeps& sweeps, 
     DMRGObserver& obs,
     const Args& args = Args::global())
    {
    LocalMPO PH(H,LH,RH,args);
    Real energy = TDVPWorker(psi,PH,t,sweeps,obs,args);
    return energy;
    }

//
//TDVP with a set of MPOs (lazily summed)
//(H vector is 0-indexed)
//
Real inline
tdvp(MPS& psi, 
     std::vector<MPO> const& Hset,
     Cplx t, 
     const Sweeps& sweeps,
     const Args& args = Args::global())
    {
    LocalMPOSet PH(Hset,args);
    Real energy = TDVPWorker(psi,PH,t,sweeps,args);
    return energy;
    }

//
//TDVP with a set of MPOs and a custom DMRGObserver
//(H vector is 0-indexed)
//
Real inline
tdvp(MPS & psi, 
     std::vector<MPO> const& Hset, 
     Cplx t,
     const Sweeps& sweeps, 
     DMRGObserver& obs,
     const Args& args = Args::global())
    {
    LocalMPOSet PH(Hset,args);
    Real energy = TDVPWorker(psi,PH,t,sweeps,obs,args);
    return energy;
    }


//
// TDVPWorker
//

template <class LocalOpT>
Real
TDVPWorker(MPS & psi,
           LocalOpT& PH,
	   Cplx t,
           Sweeps const& sweeps,
           Args const& args)
    {
    DMRGObserver obs(psi,args);
    Real energy = TDVPWorker(psi,PH,t,sweeps,obs,args);
    return energy;
    }

template <class LocalOpT>
Real
TDVPWorker(MPS & psi,
           LocalOpT& PH,
           Cplx t,
           Sweeps const& sweeps,
           DMRGObserver& obs,
           Args args)
    { 
    // Truncate blocks of degenerate singular values (or not)
    args.add("RespectDegenerate",args.getBool("RespectDegenerate",true));
  
    const bool silent = args.getBool("Silent",false);
    if(silent)
        {
        args.add("Quiet",true);
        args.add("PrintEigs",false);
        args.add("NoMeasure",true);
        args.add("DebugLevel",0);
        }
    const bool quiet = args.getBool("Quiet",false);
    const int debug_level = args.getInt("DebugLevel",(quiet ? 0 : 1));
    const int numCenter = args.getInt("NumCenter",2);

    const int N = length(psi);
    Real energy = NAN;

    psi.position(1);

    args.add("DebugLevel",debug_level);
    args.add("DoNormalize",true);

    for(int sw = 1; sw <= sweeps.nsweep(); ++sw)
        {
        cpu_time sw_time;
        args.add("Sweep",sw);
        args.add("NSweep",sweeps.nsweep());
        args.add("Cutoff",sweeps.cutoff(sw));
        args.add("MinDim",sweeps.mindim(sw));
        args.add("MaxDim",sweeps.maxdim(sw));
        args.add("MaxIter",sweeps.niter(sw));
 
        if(!PH.doWrite()
           && args.defined("WriteDim")
           && sweeps.maxdim(sw) >= args.getInt("WriteDim"))
            {
            if(!quiet)
                {
                println("\nTurning on write to disk, write_dir = ",
                        args.getString("WriteDir","./"));
                }
  
            //psi.doWrite(true);
            PH.doWrite(true,args);
            }

        if(numCenter==2)
            {
            for(int b = 1, ha = 1; ha <= 2; sweepnext(b,ha,N))
                {
                if(!quiet)
                    {
                    printfln("Sweep=%d, HS=%d, Bond=%d/%d",sw,ha,b,(N-1));
                    }
  
                PH.numCenter(2);
                PH.position(b,psi);//position2

                auto phi = psi.A(b)*psi.A(b+1);

                applyExp(PH,phi,t/2,args);

                if(args.getBool("DoNormalize",true))
                    {
                    phi/=norm(phi);
                    }
		    
                ITensor Aw;
                PH.product(phi,Aw);
                energy = real(eltC(dag(phi)*Aw));

                auto spec = psi.svdBond(b,phi,(ha==1?Fromleft:Fromright),PH,args);

                if((ha == 1 && b+1 != N) || (ha == 2 && b != 1))
                    {	
                    PH.numCenter(1);
                    PH.position((ha == 1? b+1: b),psi);//position1: fromleft: b+1,fromright: b
                    auto& M = (ha == 1? psi.Aref(b+1):psi.Aref(b));
                    applyExp(PH,M,-t/2,args);
                    if(args.getBool("DoNormalize",true))
                        {
                        M/=norm(M);
                        }
		    PH.product(M,Aw);
                    energy = real(eltC(dag(M)*Aw));
                    }

                if(!quiet)
                    { 
                    printfln("    Truncated to Cutoff=%.1E, Min_dim=%d, Max_dim=%d",
                               sweeps.cutoff(sw),
                               sweeps.mindim(sw), 
                               sweeps.maxdim(sw) );
                    printfln("    Trunc. err=%.1E, States kept: %s",
                               spec.truncerr(),
                               showDim(linkIndex(psi,b)) );
                    }

                obs.lastSpectrum(spec);

                args.add("AtBond",b);
                args.add("HalfSweep",ha);
                args.add("Energy",energy); 
                args.add("Truncerr",spec.truncerr()); 

                obs.measure(args);

                } //for loop over b
            }
        else if(numCenter==1)
            {
            ITensor M;
            for(int b = 1, ha = 1; ha <= 2; sweepnext1(b,ha,N))
                {
                if(!quiet)
                    {
                    printfln("Sweep=%d, HS=%d, Bond=%d/%d",sw,ha,b,(N-1));
                    }

                PH.numCenter(1);
                PH.position(b,psi);//position1

                ITensor phi;
                if((ha == 1 && b != 1) || (ha == 2 && b != N)) phi = M*psi.A(b);//const reference
                else phi = psi.A(b);

                applyExp(PH,phi,t/2,args);
                if(args.getBool("DoNormalize",true))
                    {
                    phi/=norm(phi);
                    }
		
		ITensor Aw;
                PH.product(phi,Aw);
                energy = real(eltC(dag(phi)*Aw));

                Spectrum spec;
                if((ha == 1 && b != N) || (ha == 2 && b != 1))
                    {
                    ITensor U,V,S;
                    if(ha == 1) V = ITensor(commonIndex(psi.A(b),psi.A(b+1),"Link"));
                    else V = ITensor(commonIndex(psi.A(b-1),psi.A(b),"Link"));
                    spec = svd(phi,U,S,V,args);// QR, oc tensor to be returned
                    psi.Aref(b) = U;
                    M = S*V;

                    PH.numCenter(0);
                    PH.position((ha == 1? b+1: b),psi);//position0

                    applyExp(PH,M,-t/2,args);
                    if(args.getBool("DoNormalize",true))
                        {
                        M/=norm(M);
                        }
                    PH.product(M,Aw);
                    energy = real(eltC(dag(M)*Aw));
                    }
                else
                    {
                    psi.Aref(b) = phi;
                    }

                if(!quiet)
                    { 
                    printfln("    Truncated to Cutoff=%.1E, Min_dim=%d, Max_dim=%d",
                               sweeps.cutoff(sw),
                               sweeps.mindim(sw), 
                               sweeps.maxdim(sw) );
                    printfln("    Trunc. err=%.1E, States kept: %s",
                               spec.truncerr(),
                               showDim(linkIndex(psi,b)) );
                    }
  
                obs.lastSpectrum(spec);
  
                args.add("AtBond",b);
                args.add("HalfSweep",ha);
                args.add("Energy",energy); 
                args.add("Truncerr",spec.truncerr()); 
  
                obs.measure(args);
  
            } //for loop over b
        } // End NumCenter=1 case
    else
        {
        Error("NumCenter must be 1 or 2");
        }
  
    if(!silent)
        {	
         auto sm = sw_time.sincemark();
         printfln("    Sweep %d/%d CPU time = %s (Wall time = %s)",
                    sw,sweeps.nsweep(),showtime(sm.time),showtime(sm.wall));
        }
    
    if(obs.checkDone(args)) break;
  
    } //for loop over sw
  
    if(args.getBool("DoNormalize",true))
        {
        if(numCenter==1) psi.position(1);
        psi.normalize();
        }

    return energy;
    }

} //namespace itensor

#endif
