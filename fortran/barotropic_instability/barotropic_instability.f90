module global
    character(200), parameter :: DATA_DIR_PATH = &
        & "/workspace/data/fortran/barotropic_instability"
    
    real(8), parameter :: PI = 3.1415926535897932385d0
    real(8), parameter :: PIx2 = 6.283185307179586477d0
    
    integer, parameter :: VORTEX_GRID_FILE_NO = 10
    integer, parameter :: VORTEX_SPCT_FILE_NO = 11
    integer, parameter :: NUDGING_VORTEX_SPCT_FILE_NO = 12

    character(200), save :: CONFIG_NAME = ""
    character(200), save :: VORTEX_SPCT_FILE_PATH
    character(200), save:: VORTEX_GRID_FILE_PATH
    character(200), save :: NUDGING_VORTEX_SPCT_FILE_PATH

    integer, save :: KM, LM, IM, JM, NUDGE_KM, NUDGE_LM
    ! IM > 3*KM and JM > 3*LM/2
    
    real(8), allocatable, save :: Z(:,:)
    real(8), allocatable, save :: NUDGE_Z(:,:,:)
    real(8), allocatable, save :: CURRENT_Z(:,:), NEXT_Z(:,:)
    integer(8), allocatable, save :: SEEDS(:)
    integer(8), save :: INIT_SEED
    integer, save :: NUM_SEEDS

    real(8), save :: CURRENT_T, NEXT_T
    
    real(8), save :: WIDTH_VORTEX = -999d0
    real(8), save :: DECAY_VORTEX = -999d0
    real(8), save :: VELOCITY_VORTEX = -999d0
    ! In shear region, vorticity = VELOCITY_VORTEX / (WIDTH_VORTEX/2d0)

    integer(8), save :: K_MAX = -999
    real(8), save :: PERTURBATION_AMP = -999d0
    real(8), save :: RELAX_T = -999d0

    integer, save :: NSTEP = -999
    real(8), save :: H = -999d0
    integer, save :: M = -999
    real(8), save :: DELTAT ! = H/M

    integer, save :: NV = -999
    real(8), save :: DNU = -999d0
    
end module

module subprog
    use global
    use mt19937_64
    implicit none

    integer, save :: ITJ(5), ITI(5)
    real(8), allocatable, save :: TJ(:),TI(:)
    real(8), allocatable, save :: DL(:,:)
    real(8), allocatable, save :: WS(:,:), WG(:,:,:)
    real(8), allocatable, save :: BASE_Z(:,:)
contains

    subroutine set_random_seeds()
        integer i
        integer(8) s
        
        write(*,*) "All seeds are generated using INIT_SEED = ", INIT_SEED
        call init_genrand64(INIT_SEED)

        do i = 1, NUM_SEEDS
            s = int8(genrand64_real3() * 1d15)
            SEEDS(i) = s
        end do
    end subroutine

    subroutine validate_wavenumbers()
        if (IM <= 3d0*KM .or. JM <= 3d0*LM/2d0) then
            write(*,*) "Aliasing errors will occur!! Please set an appropriate grids or wavenumbers."
            return
        end if

        if (KM /= LM) then
            write(*,*) "Wavenumbers along x and y are different."
        end if

        if (NUDGE_KM /= NUDGE_LM) then
            write(*,*) "Wavenumbers for nudging along x and y are different."
        end if

        if (K_MAX > KM .or. K_MAX > LM) then
            write(*,*) "Perturbation wavenumber is invalid."
            return
        end if
    end subroutine

    subroutine read_config_csv(path)
        character(*), intent(in) :: path
        
        integer, parameter :: CSV_FILE_NO = 99
        character(200) dummy
        
        open (CSV_FILE_NO, file=path, action='read')
        read (CSV_FILE_NO, '()') ! skip header

        read(CSV_FILE_NO, *) dummy, WIDTH_VORTEX
        write(*,*) "WIDTH_VORTEX = ", WIDTH_VORTEX

        read(CSV_FILE_NO, *) dummy, DECAY_VORTEX
        write(*,*) "DECAY_VORTEX = ", DECAY_VORTEX
        
        read(CSV_FILE_NO, *) dummy, VELOCITY_VORTEX
        write(*,*) "VELOCITY_VORTEX = ", VELOCITY_VORTEX
        
        read(CSV_FILE_NO, *) dummy, NUM_SEEDS
        write(*,*) "NUM_SEEDS = ", NUM_SEEDS
        allocate(SEEDS(NUM_SEEDS))

        read(CSV_FILE_NO, *) dummy, INIT_SEED
        write(*,*) "INIT_SEED = ", INIT_SEED

        read(CSV_FILE_NO, *) dummy, K_MAX
        write(*,*) "K_MAX = ", K_MAX
        
        read(CSV_FILE_NO, *) dummy, PERTURBATION_AMP
        write(*,*) "PERTURBATION_AMP = ", PERTURBATION_AMP
        
        read(CSV_FILE_NO, *) dummy, RELAX_T
        write(*,*) "RELAX_T = ", RELAX_T

        read(CSV_FILE_NO, *) dummy, NSTEP
        write(*,*) "NSTEP = ", NSTEP
        
        read(CSV_FILE_NO, *) dummy, H
        write(*,*) "H = ", H
        
        read(CSV_FILE_NO, *) dummy, M
        write(*,*) "M = ", M
        
        DELTAT = H/dble(M)
        write(*,*)  "DELTAT = ", DELTAT

        read(CSV_FILE_NO, *) dummy, NV
        write(*,*) "NV = ", NV
        
        read(CSV_FILE_NO, *) dummy, DNU
        write(*,*) "DNU = ", DNU

        close (CSV_FILE_NO)
    end subroutine

    subroutine initialize(seed)
        integer(8), intent(in) :: seed
        integer k, l
        
        ! Set data paths
        call set_paths(seed)

        ! Allocate arrays
        allocate(Z(-KM:KM,LM))
        allocate(NUDGE_Z(0:NSTEP,-NUDGE_KM:NUDGE_KM,NUDGE_LM))
        allocate(CURRENT_Z(-NUDGE_KM:NUDGE_KM,NUDGE_LM))
        allocate(NEXT_Z(-NUDGE_KM:NUDGE_KM,NUDGE_LM))
        allocate(BASE_Z(-NUDGE_KM:NUDGE_KM,NUDGE_LM))
        allocate(TJ(JM*6))
        allocate(TI(IM*2))
        allocate(DL(-KM:KM,LM))        
        allocate(WS(-KM:KM,0:LM))
        allocate(WG(0:JM,0:IM-1,3))
        Z(:,:) = 0d0
        NUDGE_Z(:,:,:) = 0d0
        DL(:,:) = 0d0

        ! Initialize C2PACK
        call C2INIT(JM,IM,ITJ,TJ,ITI,TI)

        ! Set viscosity coefficients
        do l=1,LM
            do k=-KM,KM
                DL(k,l)=exp(-DNU*DELTAT/2*(1d0*(k*k+l*l))**NV)
            end do
        end do

        if (KM == NUDGE_KM .and. LM == NUDGE_LM) then
            write(*,*) "Initial condition is set at random"
            call set_initial_condition(seed)
        else
            ! Read data for nudging
            open (NUDGING_VORTEX_SPCT_FILE_NO, &
                file=NUDGING_VORTEX_SPCT_FILE_PATH, form='unformatted', access='stream', action='read')
            do k=0,NSTEP
                read(NUDGING_VORTEX_SPCT_FILE_NO) NUDGE_Z(k,:,:)
            end do
            close (NUDGING_VORTEX_SPCT_FILE_NO)

            Z(:,:) = 0d0
            Z(-NUDGE_KM:NUDGE_KM,1:NUDGE_LM) = NUDGE_Z(0,:,:)
            write(*,*) "Initial condition is set from data"
        end if
    end subroutine
    
    subroutine set_paths(seed)
        integer(8), intent(in) :: seed
        character(100) s1, s2

        write(s2,*) seed
        s2 = adjustl(s2)

        write(s1,*) KM
        s1 = adjustl(s1)

        VORTEX_SPCT_FILE_PATH = trim(DATA_DIR_PATH) // "/" // trim(CONFIG_NAME) // "/vortex_spct_T" // &
            trim(s1) // "_" // trim(CONFIG_NAME) // "_seed" // trim(s2) // ".dat"
        VORTEX_GRID_FILE_PATH = trim(DATA_DIR_PATH) // "/" // trim(CONFIG_NAME) // "/vortex_grid_T" // &
            trim(s1) // "_" // trim(CONFIG_NAME) // "_seed" // trim(s2) // ".dat"
        
        write(s1,*) NUDGE_KM
        s1 = adjustl(s1)
        
        NUDGING_VORTEX_SPCT_FILE_PATH = trim(DATA_DIR_PATH)  // "/" // trim(CONFIG_NAME) // "/vortex_spct_T" &
            // trim(s1) // "_" // trim(CONFIG_NAME) // "_seed" // trim(s2) // ".dat"

        write(*,*) "Output paths"
        write(*,*) VORTEX_SPCT_FILE_PATH
        write(*,*) VORTEX_GRID_FILE_PATH
        write(*,*) "Input paths"
        write(*,*) NUDGING_VORTEX_SPCT_FILE_PATH
    end subroutine

    subroutine set_initial_condition(seed)
        integer(8), intent(in) :: seed
        integer i, j, k, l
        real(8) x, y, y0, v

        ! Set initial condition
        y0 = PI/2        
        do i=0,IM-1
            x = PIx2*i/IM
            do j=0,JM
                y = PI*j/JM
                if (abs(y-y0) <= WIDTH_VORTEX/2d0) then
                    v = 1d0
                else if (y > y0) then
                    v = y - y0 - WIDTH_VORTEX/2d0
                    v = exp(-0.5d0 * (v/DECAY_VORTEX)**2)
                else
                    v = y0 - y - WIDTH_VORTEX/2d0
                    v = exp(-0.5d0 * (v/DECAY_VORTEX)**2)
                endif
                WG(j,i,1) = v * VELOCITY_VORTEX / (WIDTH_VORTEX/2d0)
            end do
        end do
        call C2G2SA(LM,KM,JM,IM,WG,Z,WG(0,0,3),ITJ,TJ,ITI,TI,1)

        call init_genrand64(seed)
        do l=1,LM
            do k=-KM,KM
                if (abs(k) <= K_MAX .and. l <= K_MAX) then
                    Z(k,l) = Z(k,l) + PERTURBATION_AMP * generate_noramal_rand()
                end if
            end do
        end do
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
        real(8), intent(inout) :: Y(-KM:KM,LM)

        Y(:, :) = DL(:, :)*Y(:, :)
    end subroutine

    subroutine nonlinear_term(X, Y, DY)
        real(8), intent(in):: X, Y(-KM:KM,LM)
        real(8), intent(out) :: DY(-KM:KM,LM)
        real(8) :: current_w, next_w
        
        call C2AJBS(LM,KM,JM,IM,1d0,Y,DY,WS,WG,ITJ,TJ,ITI,TI)

        if (KM == NUDGE_KM .and. LM == NUDGE_LM) then
            return
        else
            current_w = (NEXT_T - X) / (NEXT_T - CURRENT_T)
            next_w = (X - CURRENT_T) / (NEXT_T - CURRENT_T)
            BASE_Z(:,:) = next_w*NEXT_Z(:,:) + current_w*CURRENT_Z(:,:)

            DY(-NUDGE_KM:NUDGE_KM,1:NUDGE_LM) = DY(-NUDGE_KM:NUDGE_KM,1:NUDGE_LM) &
                - (Y(-NUDGE_KM:NUDGE_KM,1:NUDGE_LM) - BASE_Z(:,:)) / RELAX_T
        end if
    end subroutine

    subroutine get_grid_data(S, G)
        real(8), intent(in) :: S(-KM:KM,LM)
        real(8), intent(out) :: G(0:JM,0:IM-1)

        call C2S2GA(LM,KM,JM,IM,S,G,WG(0,0,3),ITJ,TJ,ITI,TI,1)
    end subroutine

    subroutine update_nudging_state(istep, time)
        integer, intent(in) :: istep
        real(8), intent(in) :: time

        CURRENT_T = time
        NEXT_T = CURRENT_T + H
    
        CURRENT_Z(:,:) = NUDGE_Z(istep,:,:)
        NEXT_Z(:,:) = NUDGE_Z(istep+1,:,:)
    end subroutine

end module subprog

program main
    use global
    use subprog
    implicit none

    integer, parameter :: ARG_POSITION = 1
    integer, parameter :: WAVENUMBERS(1) = (/ 42 /)
    integer, parameter :: X_GRID_NUMS(1) = (/ 128 /)
    integer, parameter :: Y_GRID_NUMS(1) = (/ 64 /)

    character(200) :: csv_path
    integer :: N, istep, iloop, iseed
    real :: start_cpu_time, end_cpu_time
    real(8) :: t
    real(8), allocatable :: W(:,:)
    real(8), allocatable :: VORTEX(:,:)

    call getarg(ARG_POSITION,CONFIG_NAME)
    csv_path = "/workspace/fortran/barotropic_instability/config/" // trim(CONFIG_NAME) // ".csv"
    write(*,*) "Config = ", csv_path
    
    NUDGE_KM = WAVENUMBERS(1)
    NUDGE_LM = WAVENUMBERS(1)
    write(*,*) "NUDGE_KM = ", NUDGE_KM, ", NUDGE_LM = ", NUDGE_LM
    call read_config_csv(csv_path)
    call set_random_seeds()
    
    do iseed = 1, size(SEEDS)
        write(*,*) ""
        write(*,*) "SEED = ", SEEDS(iseed)

        do iloop = 1, size(WAVENUMBERS)
            call cpu_time(time=start_cpu_time)
            
            istep = 0
            t = 0d0
            KM = WAVENUMBERS(iloop)
            LM = WAVENUMBERS(iloop)
            IM = X_GRID_NUMS(iloop)
            JM = Y_GRID_NUMS(iloop)
            N = LM*(2*KM+1)
            
            write(*,*) "KM = ", KM, ", LM = ", LM
            write(*,*) "IM = ", IM, ", JM = ", JM
            call validate_wavenumbers()

            allocate(W(N, 3))
            allocate(VORTEX(0:JM,0:IM-1))

            ! Initialization
            call initialize(SEEDS(iseed))

            open (VORTEX_GRID_FILE_NO, file=VORTEX_GRID_FILE_PATH, form='unformatted', access='stream')
            open (VORTEX_SPCT_FILE_NO, file=VORTEX_SPCT_FILE_PATH, form='unformatted', access='stream')

            call get_grid_data(Z, VORTEX)
            call update_nudging_state(0, t)
            write (VORTEX_GRID_FILE_NO) VORTEX
            write (VORTEX_SPCT_FILE_NO) Z
            ! write (*, *) "Time step = ", istep, ", Time = ", t

            ! Perform integration
            do istep = 1, NSTEP
                call TDRKNU(N,M,H,t,Z,W,linear_term,nonlinear_term)
                call get_grid_data(Z, VORTEX)
                if (istep < NSTEP) then
                    call update_nudging_state(istep, t)
                end if
                write (VORTEX_GRID_FILE_NO) VORTEX
                write (VORTEX_SPCT_FILE_NO) Z
                ! write (*, *) "Time step = ", istep, ", Time = ", t
            end do

            ! Finalization
            close (VORTEX_GRID_FILE_NO)
            close (VORTEX_SPCT_FILE_NO)
            deallocate(W, VORTEX, Z, NUDGE_Z, CURRENT_Z, NEXT_Z, BASE_Z, TJ, TI, DL, WS, WG)
            
            call cpu_time(time=end_cpu_time)
            write (*, *) "Total elapsed time = ", end_cpu_time - start_cpu_time, " [sec]"
            write(*, *) ""
        end do
    end do
end program main