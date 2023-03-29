#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from read_FLASH import Fields

# from
# https://bitbucket.org/mpi4py/mpi4py-fft/raw/67dfed980115108c76abb7e865860b5da98674f9/examples/spectral_dns_solver.py
# with modification for complex numbers
def getTransferWWAnyToAny(self, Result, KBins, QBins, Terms):
        """ return what
                    formalism -- determined by the definition of the spectral kinetic energy density
                "WW": E_kin(k) = 1/2 |FT(sqrt(rho)U)|^2
        
        Args:
            Result -- a (potentially empty) dictionary to store the results in
            Ks -- range of destination shell wavenumber
            Qs -- range of source shell wavenumbers
            Terms -- list of terms that should be analyzed, could be
                "UUA": Kinetic to kinetic by advection        
        """
        
        #self.populateResultDict(Result,KBins,"WW",Terms,"AnyToAny")
        self.calcBasicVars("WW")
        
        
        rho = self.rho
        U = self.U
        B = self.B
        S = self.S
        W = self.W
        FT_W = self.FT_W
        FT_S = self.FT_S
        FT_B = self.FT_B
        FT_P = self.FT_P
        FT_Acc = self.FT_Acc

        startTime = time.time()

        # clear Q terms
        W_Q = None
        S_Q = None
        B_Q = None
        SDivW_QoverGammaSqrtRho = None
        OneOverGammaSqrtRhogradSS_Q = None
        OneOverTwoSqrtRhogradBB_Q = None
        UdotGradW_Q = None
        UdotGradS_Q = None
        UdotGradB_Q = None
        bDotGradB_Q = None
        BdotGradW_QoverSqrtRho = None
        DivbW_Q = None
        bdotGradW_Q = None
        W_QoverSqrtRho = None
        W_QoverSqrtRhoDotGradB = None
        DivW_QoverSqrtRho = None        
        DivW_Qb = None
        W_QdotGradb = None
        DivW_Q = None
        BDivW_Qover2SqrtRho = None
        OneOverSqrtRhoGradP_Q = None
        SqrtRhoAcc_Q = None
        
        DivU = None
        b = None
        Divb = None
        
        for q in range(len(QBins)-1):
            QBin = "%.2f-%.2f" % (QBins[q],QBins[q+1])

            # clear K terms
            W_K = None
            S_K = None	
            B_K = None
            
            for k in range(len(KBins)-1):
                
                KBin = "%.2f-%.2f" % (KBins[k],KBins[k+1])

                #  - W_K * (U dot grad) W_Q - 0.5 W_K W_Q DivU
                if "UU" in Terms:
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                    
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])                        
                        
                    if UdotGradW_Q is None:
                        UdotGradW_Q = MPIXdotGradY(self.comm,U,W_Q)                        
                    
                    if DivU is None:
                        DivU = MPIdivX(self.comm,U)
                    
                    
                    localSum = - np.sum(W_K * UdotGradW_Q)              

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    localSum = - np.sum(0.5 * W_K * W_Q * DivU)                    

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                    
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UUA","AnyToAny",KBin,QBin,totalSumA)
                        self.addResultToDict(Result,"WW","UUC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","UU","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with UU for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime )) 


def get_local_wavenumbermesh(FFT, L):
    """Returns local wavenumber mesh."""
    s = FFT.local_slice()
    N = FFT.global_shape()
    # Set wavenumbers in grid
    if FFT.dtype() == np.complex128:
        k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N]
    else:
        k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1./N[-1]).astype(int))
    K = [ki[si] for ki, si in zip(k, s)]
    Ks = np.meshgrid(*K, indexing='ij', sparse=True)
    Lp = 2*np.pi/L
    for i in range(3):
        Ks[i] = (Ks[i]*Lp[i]).astype(float)
    return [np.broadcast_to(k, FFT.shape(True)) for k in Ks]

def getShellX(self,FTquant,Low,Up):
    """ extracts shell X-0.5 < K <X+0.5 of FTquant """

    if FTquant.shape[0] == 3:    
        Quant_X = newDistArray(self.FFT,False,rank=1)
        for i in range(3):
            tmp = np.where(np.logical_and(self.localKmag > Low, self.localKmag <= Up),FTquant[i],0.)
            Quant_X[i] = self.FFT.backward(tmp,Quant_X[i])
    else:
        Quant_X = newDistArray(self.FFT,False)
        tmp = np.where(np.logical_and(self.localKmag > Low, self.localKmag <= Up),FTquant,0.)
        Quant_X = self.FFT.backward(tmp,Quant_X)        

    return Quant_X

if __name__ == "__main__":
    field_dir = "/Users/beattijr/Documents/Research/2022/energy-transfer-analysis/test_data"
    field = Fields(f"{field_dir}/Turb_hdf5_plt_cnt_0050",reformat=True)
    field.read("dens")
    field.read("vel")
    field.fourier_transform("vel")