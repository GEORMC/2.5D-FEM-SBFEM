      subroutine zriccati(n,a,zst)
      use filePathModule
      use inoutch_module
      implicit none 
      integer n,n2,i,j,info
      logical, allocatable :: bw(:)
      double complex a(2*n,2*n),zst(n,n),zzst(n,n)
      double precision ,allocatable ::  tmpmm(:)
      double complex,  allocatable :: z(:,:),eign(:,:),tmpm(:)
      integer, allocatable :: indx(:)
c
      external zselect
c
      n2=n+n
      allocate(z(n2,n2),eign(n2,2),indx(n2),bw(n2)
     &,tmpm(n2*max(4,n2)),tmpmm(n2))
c
C     POSITIVE-DEFINITE SOLUTION OF RICCATI EQUATION 
C     USING SCHUR DECOMPOSITION
C
c  n:         total number of dof
c  a(*,*):    coefficient matrix
c  z(*,*):    orthogonal matrix
c  eign(*,1): real part of eigenvalues
c  eign(*,2): imaginary part of eigenvalues
c  zst(*,*):  positive-definite solution of Riccati equation
c  select:    function for selecting the first n eigenvalues
c
c    Schur decomposition

      
       call ZGEES('V','S',zselect,n2,a,n2,j,eign,z,n2
     &,tmpm,n2*max(4,n2),tmpmm,bw,info)
    !  call DGEES('V','S',selectt,n2,a,n2,j,eign,eign(1,2),z,n2
    ! &,tmpm,n2*max(4,n2),bw,info)
    
    

    
      if(info.ne.0) stop ' Schur factorization failed'
    ! call ZSchurFormSort(n2, a, n2, z)
      
      
      if(j.lt.n) then
c      error in solving Riccati equation
c      check eigenvalues

      stop
      end if

      do i=1,n
       do j=1,n
        zst(j,i)=z(n+i,j)

       end do
      end do

      call ZGETRF(n,n,z,n2,indx,i)

      call ZGETRS('T',n,n,z,n2,indx,zst,n,i)
c
      deallocate(z,eign,bw,indx,tmpm,tmpmm)
c
      return
      end  
	  
	  
	       logical function zselect(w)
      implicit none
      double complex w
      double precision wr,wi
      wi=IMAG(w)
      wr=DBLE(w)

c
c     select eigenvalues with negative real parts 
c     in Schur decomposition
c
      if(wr.lt.0.d0) then 
       zselect=.TRUE.
      else
       zselect=.FALSE.
      end if
      return
      end  