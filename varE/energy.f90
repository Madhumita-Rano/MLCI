subroutine energy(length,eval,dnorm)
  use commonarrays, only: c, h, ijh, s, ijs
  use dyn_par
  implicit  real*8  (a-h,o-z)       
  !common /hands/     h(maxh), ijh(maxh), s(maxs), ijs(maxs)
  !common /coef/      c(maxc) 

  if(ijh(1) .ne. length + 2) STOP 'energy: h array mismatch'
  if(ijs(1) .ne. length + 2) STOP 'energy: s array mismatch'
  
  open(41,file='csf_energy',status='unknown')
  eval  = 0.0
  dnorm = 0.0
  do ici=1, length
     write(41,*)h(ici)
     eval  = eval  + c(ici)*h(ici)*c(ici)
     dnorm = dnorm + c(ici)*s(ici)*c(ici)
     do jci = ijh(ici), ijh(ici+1) - 1
        eval  =  eval  + 2.0*c(ici)*h(jci)*c(ijh(jci))
     enddo
     do jci = ijs(ici), ijs(ici+1) - 1
        dnorm =  dnorm + 2.0*c(ici)*s(jci)*c(ijs(jci))
     enddo
  enddo

  eval = eval/dnorm
  close(41)
  return
end subroutine energy
