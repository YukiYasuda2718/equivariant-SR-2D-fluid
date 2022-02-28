module global
    implicit none
    character(200), parameter :: DATA_DIR_PATH = "/workspace/data/fortran/decaying_turbulence/test"

    integer, parameter :: KM = 42, LM = 42
    integer, parameter :: KMAX = KM
    integer, parameter :: IM = 2**7, JM = 2**7
    real(8), parameter :: R = 1d0
    integer, parameter :: NSTEP = 500
    real(8), parameter :: H = 0.01d0
    integer, parameter :: M = 50
    integer, parameter :: NV = 1
    real(8), parameter :: DNU = 1d-2
    
    integer(8), parameter :: INIT_SEED = 0
    integer, parameter :: NUM_SEEDS = 100

    real(8), parameter :: AMPLITUDE0 = 12.0d0
    real(8), parameter :: KSCALE0 = 26.5d0

    real(8), parameter :: PI = 3.1415926535897932385d0
    real(8), parameter :: PIx2 = 6.283185307179586477d0
        
    character(200), save :: VORTEX_FILE_PATH = ""
    real(8), save :: Z(-LM:LM, -KM:KM) = 0d0
    integer(8), save :: SEEDS(NUM_SEEDS) = -999 

end module global

module subprog
    use global
    use mt19937_64
    implicit none

    real(8), parameter :: DELTAT = H/M
    real(8), save :: DL(-LM:LM, -KM:KM)
    integer, save :: ITJ(4), ITI(4)
    real(8), save :: TJ(JM*6), TI(IM*8)
    real(8), save :: WS(-LM:LM, -KM:KM)
    real(8), save :: WG(0:JM - 1, 0:IM - 1, 3)

contains

    subroutine set_random_seeds()
        integer i
        integer(8) s
        
        call init_genrand64(INIT_SEED)

        do i = 1, NUM_SEEDS
            s = int8(genrand64_real3() * 1d15)
            SEEDS(i) = s
        end do
    end subroutine

    subroutine set_output_filename(seed)
        integer(8), intent(in) :: seed
        character(100) s1, s2

        write(s1,*) KM
        s1 = adjustl(s1)

        write(s2,*) seed
        s2 = adjustl(s2)

        VORTEX_FILE_PATH = trim(DATA_DIR_PATH) // "/T" // trim(s1) // "_seed" // trim(s2) // ".dat"

        write(*, *) "Output file path = ", VORTEX_FILE_PATH
    end subroutine

    subroutine initialize(istat, seed)
        integer, intent(out):: istat
        integer(8), intent(in):: seed
        integer k, l

        write (*, *) "Initialization: start"
        istat = 0

        if (IM <= 3*KM .or. JM <= 3*LM) then
            write (*, *) "Aliasing error would occur."
            istat = 1
            return
        end if

        if (IM > 1024 .or. JM > 2048) then
            write (*, *) "Number of grids is too large."
            istat = 1
            return
        end if

        if ((KM /= LM) .or. (IM /= JM) .or. (R /= 1d0)) then
            write (*, *) "Anisotropic case has not been debugged yet."
            istat = 1
            return
        end if

        ! Make coefficients of linear term (viscosity effect)
        do k = -KM, KM
            do l = -LM, LM
                DL(l, k) = exp(-DNU*DELTAT/2d0*(1d0*(k*k + l*l))**NV)
            end do
        end do

        call PZINIT(JM, IM, ITJ, TJ, ITI, TI)

        call set_random_initial_condition(seed)

        call set_output_filename(seed)

        write (*, *) "Initialization: end"
    end subroutine

    subroutine set_random_initial_condition(seed)
        integer(8), intent(in):: seed
        real(8) :: DIST(KM + LM) = 0d0, NORM(KM + LM) = 0d0
        real(8) :: tmp
        integer :: k, l, index_k

        DIST(:) = 0d0
        NORM(:) = 0d0

        ! Set energy spectrum
        do k = 1, KMAX
            DIST(K) = exp(-dble(k)**2/KSCALE0**2)
        end do
        DIST(:) = AMPLITUDE0*DIST(:)

        ! Set random phases
        write(*,*) "Random seed == ", seed
        call init_genrand64(seed)
        do k = -KMAX, KMAX
            do l = -KMAX, KMAX
                Z(l, k) = generate_noramal_rand()
            end do
        end do

        ! Calc normalization constants over wavenumber shells
        do k = -KM, KM
            do l = -LM, LM
                index_k = int(sqrt(dble(k)**2 + dble(l)**2) + 0.5d0)
                if (index_k /= 0) then
                    tmp = Z(l, k)**2/(dble(k)**2 + dble(l)**2)
                    NORM(index_k) = NORM(index_k) + tmp
                end if
            end do
        end do

        ! Set initial condition
        do k = -KM, KM
            do l = -LM, LM
                index_k = int(sqrt(dble(k)**2 + dble(l)**2) + 0.5d0)
                if (index_k /= 0) then
                    if (NORM(index_k) /= 0) then
                        Z(l, k) = Z(l, k)*sqrt(DIST(index_k)/NORM(index_k))
                    end if
                end if
            end do
        end do

        ! Set the average of vorticity to be zero
        Z(0, 0) = 0d0
    end subroutine

    function generate_noramal_rand() result(r)
        real(8) :: r, x1, x2
        real(8), save :: y1, y2
        integer, save :: iflag = 0

        ! Box-Muller method
        ! https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

        if (iflag == 0) then
            x1 = genrand64_real3()
            x2 = genrand64_real3()
            y1 = sqrt(-2d0*log(x1))*cos(x2*PIx2)
            y2 = sqrt(-2d0*log(x1))*sin(x2*PIx2)
            r = y1
            iflag = 1
        else
            r = y2
            iflag = 0
        end if
    end function

    subroutine linear_term(X, DX, Y)
        real(8), intent(in):: X, DX
        real(8), intent(inout) :: Y(-LM:LM, -KM:KM)

        Y(:, :) = DL(:, :)*Y(:, :)
    end subroutine

    subroutine nonlinear_term(X, Y, DY)
        real(8), intent(in):: X, Y(-LM:LM, -KM:KM)
        real(8), intent(out) :: DY(-LM:LM, -KM:KM)

        call PZAJBS(LM, KM, JM, IM, R, Y, DY, WS, WG, ITJ, TJ, ITI, TI)
    end subroutine

    subroutine get_grid_data(S, G)
        real(8), intent(in) :: S(-LM:LM, -KM:KM)
        real(8), intent(out) :: G(0:JM - 1, 0:IM - 1)

        call PZS2GA(LM, KM, JM, IM, S, G, WG(:, :, 1), ITJ, TJ, ITI, TI)
    end subroutine

end module subprog

program main
    use global
    use subprog
    implicit none

    integer, parameter :: VORTEX_FILE_NO = 10
    integer, parameter :: N = (2*LM + 1)*(2*KM + 1)
    
    integer :: iloop, istat, istep
    integer(8) :: seed
    real :: start_cpu_time, end_cpu_time
    real(8) :: t, VORTEX(0:JM - 1, 0:IM - 1),  W(N, 3)

    call set_random_seeds
    write(*,*)  "SEEDS = ", SEEDS

    do iloop = 1, NUM_SEEDS
        write(*,*) "--------------------"
        write(*,*) "progress = ", iloop, "/", NUM_SEEDS
        call cpu_time(time=start_cpu_time)

        ! Initialization
        seed = SEEDS(iloop)
        istat = 0
        Z(:,:) = 0d0
        call initialize(istat, seed)
        if (istat /= 0) then
            write (*, *) "Initialization failed!"
            return
        end if
        open (VORTEX_FILE_NO, file=VORTEX_FILE_PATH, form='unformatted', access='stream')

        ! Numerical integration
        call get_grid_data(Z, VORTEX)
        write (VORTEX_FILE_NO) VORTEX

        t = 0d0
        write (*, *) "Simulation: start"
        do istep = 1, NSTEP
            call TDRKNU(N, M, H, t, Z, W, linear_term, nonlinear_term)
            call get_grid_data(Z, VORTEX)
            write (VORTEX_FILE_NO) VORTEX
            if (mod(istep, 100) == 0) then
                write (*, *) "Time step = ", istep, ", Time = ", t
            end if
        end do
        write (*, *) "Simulation: end"

        ! Finalization
        close (VORTEX_FILE_NO)
        call cpu_time(time=end_cpu_time)
        write (*, *) "Total elapsed time = ", end_cpu_time - start_cpu_time, " [sec]"
    end do

end program main
