subroutine init_phi(lo, hi, phi, philo, phihi, dx, prob_lo, prob_hi) bind(C, name="init_phi")
  use amrex_fort_module, only : amrex_real

  implicit none

  integer, intent(in) :: lo(2), hi(2), philo(2), phihi(2)
  real(amrex_real), intent(inout) :: phi(philo(1):phihi(1),philo(2):phihi(2))
  real(amrex_real), intent(in   ) :: dx(2) 
  real(amrex_real), intent(in   ) :: prob_lo(2) 
  real(amrex_real), intent(in   ) :: prob_hi(2) 

  integer          :: i,j
  double precision :: x,y,r2,tupi
  tupi=3.14159265358979323846d0*2d0

  do j = lo(2), hi(2)
     y = prob_lo(2) + (dble(j)+0.5d0) * dx(2)
     do i = lo(1), hi(1)
        x = prob_lo(1) + (dble(i)+0.5d0) * dx(1)
        phi(i,j) = sin(x*tupi)*sin(y*tupi)
     end do
  end do

end subroutine init_phi

subroutine err_phi(lo, hi, phi, philo, phihi, dx, prob_lo, prob_hi,time,v,nu) bind(C, name="err_phi")

  use amrex_fort_module, only : amrex_real

  implicit none

  integer, intent(in) :: lo(2), hi(2), philo(2), phihi(2)
  real(amrex_real), intent(inout) :: phi(philo(1):phihi(1),philo(2):phihi(2))
  real(amrex_real), intent(in   ) :: dx(2) 
  real(amrex_real), intent(in   ) :: prob_lo(2) 
  real(amrex_real), intent(in   ) :: prob_hi(2) 
  real(amrex_real), intent(in   ) :: time
  real(amrex_real), intent(in   ) :: v !  Velocity
  real(amrex_real), intent(in   ) :: nu !  diffusion coefficient

  integer          :: i,j
  double precision :: x,y, tranx,trany
  double precision :: sym,tupi

  tupi=3.14159265358979323846d0*2d0
  sym=(-4.0d0+2.0d0*cos(tupi*dx(1))+2.0d0*cos(tupi*dx(2)))/(dx(1)*dx(1))


  
  !  print *,Tfin,tupi,sym
  trany = v*(1.0d0-dx(2)*dx(2)*tupi*tupi/6)*time
  tranx = v*(1.0d0-dx(1)*dx(1)*tupi*tupi/6)*time          


  do j = lo(2), hi(2)
     y = prob_lo(2) + (dble(j)+0.5d0) * dx(2)+trany
     do i = lo(1), hi(1)
        x = prob_lo(1) + (dble(i)+0.5d0) * dx(1) + tranx

        phi(i,j) = phi(i,j)-sin(x*tupi)*sin(y*tupi)*exp(nu*time*sym)
     end do
  end do

end subroutine err_phi
