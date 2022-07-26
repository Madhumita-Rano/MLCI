gthresh,printci=0.00
memory,10,G
geometry={o;h1,o,r;h2,o,r,h1,theta}   !Z-matrix geometry input
r=4.8                              !bond length
theta=104.5                             !bond angle

{hf                                    !do scf calculation
print,20
}

put,molden,h2o.molden
