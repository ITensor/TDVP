To run this sample TDVP code, you first need to have ITensor V3
installed. Then, edit the Makefile to point towards your installation
of ITensor as well as the location of the TDVP source code (the tdvp.h header file).

### TDVP

#### Function
``tdvp(MPS psi, MPO H, Cplx t, Sweeps sweeps, Args args) -> Real energy``

``tdvp(MPS psi, MPO H, Cplx t, Sweeps sweeps, DMRGObserver obs, Args args) -> Real energy``

Note there are other interfaces available for TDVP, which are similar to their [DMRG counterparts](http://itensor.org/docs.cgi?page=classes/dmrg&vers=cppv3).

#### Parameters

`psi`: the MPS to be time evolved.

`H`: the MPO of the Hamiltonian.

`t`: the time step of TDVP. It can be real, imaginary, or complex. The corresponding time evolution operator of a single time step will be <img src="https://render.githubusercontent.com/render/math?math=e^{tH}">. Therefore, to do real time evolution, `t` need to be purely imaginary; to do imaginary time evolution, `t` need to be purely real.

`sweeps`: Specify the sweep parameters of TDVP (similar to DMRG). `nsweeps` is the number of TDVP sweeps. A TDVP sweep = a sweep from left to right with half time step + a sweep from right to left with half time step. The total evolution time = `t*nsweep`. `maxdim` is the maximum bond dimension of the sweep. `cutoff` is the truncation error of the sweep (to allow truncation for the one-site TDVP, one needs to set the `Truncate` `args` to `true` (see below)). `niter` is the maximum number of lanczos iterations used when solving each local effective TDVP equations.

`obs`: is the observer one can customize to do measurement after each sweep without recalling the `tdvp` function. Similar to its use in DMRG.

`args`: `NumCenter` can either be 1 or 2, corresponding to the one-site and two-site TDVP respectively (default is `2`). `Truncate` choose whether or not truncate when doing the SVD (for `NumCenter=1`, default is `false`; for  `NumCenter=2`, default is `true`). `DoNormalize` choose whether or not normalize the MPS after a TDVP sweep (default is `true`). `Quiet` choose to whether or not print out the information of each local update. If `WriteDim` is specified, then the environment tensors PH's for the TDVP sweep will be written to disk when the bond dimension of the MPS is larger than `WriteDim`. `WriteDir` gives the directory to write those environment tensors PH's (default is the current directory).


### Global subspace expansion

#### Function
``addBasis(MPS phi, MPO H, vector<Real> truncK, Args args)``

``addBasis(MPS phi, MPO H, vector<int> maxdimK, Args args)``

#### Parameters
`phi`: the MPS will be global subspace expanded after calling the function.

`H`: the MPO of the operator to be used to construct the subspace, e.g. the Hamiltonian <img src="https://render.githubusercontent.com/render/math?math=H"> or the operator <img src="https://render.githubusercontent.com/render/math?math=1-\mathrm{i}\tau H">

`truncK` `maxdimK`: one can choose to either specify the truncation error `truncK` or the maximum bond dimension `maxdimK` when applying the MPO `H`. Each element in the vector corresponds to each order of application. For example, `truncK={1e-8,1e-6}` means the truncation error of `H` applying to `phi` is 1e-8 and the truncation error of `H` applying to `H*phi`(obtained from previous step) is 1e-6.

`args`: `Cutoff` set the truncation error when diagonalizing the sum of the reduced density matrices (default is `1e-15`). `KrylovOrd` is the dimension <img src="https://render.githubusercontent.com/render/math?math=k"> of the Krylov subspace <img src="https://render.githubusercontent.com/render/math?math=\{\phi,H\phi,...,H^{k-1}\phi\}"> (default is `2`). `Method` specify which method is used when applying the MPO, it can be e.g. `DensityMatrix` or `Fit` (default is `DensityMatrix`). `DoNormalize` choose whether or not to perform normalization after applying the MPO (default is `false`). `Nsweep` specify the number of sweeps if use the `Fit` method to apply MPO (default is `2`). `Quiet` choose whether or not print out the norm and bond dimension of the Krylov vectors and the warning messages, (default is `false`). If `WriteDim` is provided, `DensityMatrix` way of applying the MPO will write the intermediate environment tensors E's to disk when the bond dimension of the MPS is bigger than `WriteDim`, thus reducing the memory usage. `WriteDir` gives the directory to write those environment tensors E's (default is the current directory).
