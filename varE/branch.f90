subroutine branch(nu_configs,length,llast,seed)
  use commonarrays, only: icij, nword, c, n_alpha, n_beta, nfreeze,mr_up,mr_dn,mrnum,mralpha
  use dyn_par
  !implicit none
  implicit real*8   (a-h,o-z)
  !common  /config/  icij(2,iword,maxc), nword
  !common  /coef/    c(maxc)
  !common  /occupy/  ntotal,n_alpha,n_beta,i_sx2
  !common  /froze/   nfreeze, ifreeze(maxocc)
  !common  /active/  nactive, iactive(maxocc)
  integer :: jmr
  integer, allocatable :: mrdet(:,:)

  parameter (one3= 1.0/3.0   )
  parameter (two3= 2.0/3.0   )
  ilength =length
  
  if (lref.eq.0) then
     lref_in_br=length
  else
     lref_in_br=lref  
  endif

  imrnum = 1
  do while (ilength-length .lt. nu_configs)

     call random(seed,rand) 
     i=int(dble(lref_in_br)*rand + 1)

     ilength = ilength + 1
     if(ilength.gt.maxc) STOP 'Error branch: exceeded maximum configs'

     ! setup new configurations based on old configs
111  continue
     do n=1,nword
        icij(1,n,ilength) = 0
        icij(2,n,ilength) = 0
     enddo
     do jmr=1,mralpha
        jshift      = mr_up(imrnum,jmr) - 1
        n           = jshift/int_bits +1
        jshift      = jshift -(n-1)*int_bits
        icij(1,n,ilength) = ibset(icij(1,n,ilength),jshift)                       ! BW
     enddo

     do jmr=1,mralpha
        jshift      = mr_dn(imrnum,jmr) - 1
        n           = jshift/int_bits +1
        jshift      = jshift -(n-1)*int_bits
        icij(2,n,ilength) = ibset(icij(2,n,ilength),jshift)                       ! BW
     enddo

     ! make a double or single substitution?
     call random(seed,rand)
     call random(seed,rand)
     imrnum = imrnum + 1
     if(i.gt.maxc) STOP 'Error branch: exceeded maximum configs'



     if(ilength-length .ge. nu_configs) goto 222
  enddo

222 llast  = length
  length = ilength

  do ici=llast+1, length
     c(ici) = 0.0
  enddo

  return
end subroutine branch
