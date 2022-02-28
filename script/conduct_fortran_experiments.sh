#!/bin/sh

docker-compose up -d fortran

# decaying turbulence experiment
docker-compose exec fortran /workspace/fortran/decaying_turbulence/compile_train.sh
docker-compose exec fortran /workspace/fortran/decaying_turbulence/a.out
docker-compose exec fortran /workspace/fortran/decaying_turbulence/compile_test.sh
docker-compose exec fortran /workspace/fortran/decaying_turbulence/a.out

# barotropic instability experiment
docker-compose exec fortran /workspace/fortran/barotropic_instability/compile.sh
docker-compose exec fortran /workspace/fortran/barotropic_instability/a.out default_negative
docker-compose exec fortran /workspace/fortran/barotropic_instability/a.out default_positive
docker-compose exec fortran /workspace/fortran/barotropic_instability/a.out shear_with_0p40_negative
docker-compose exec fortran /workspace/fortran/barotropic_instability/a.out shear_with_0p40_positive

# barotropic instability experiment with spectral nudging
docker-compose exec fortran /workspace/fortran/barotropic_instability_spectral_nudging/compile.sh
docker-compose exec fortran /workspace/fortran/barotropic_instability_spectral_nudging/a.out default_negative
docker-compose exec fortran /workspace/fortran/barotropic_instability_spectral_nudging/a.out default_positive
docker-compose exec fortran /workspace/fortran/barotropic_instability_spectral_nudging/a.out shear_with_0p40_negative
docker-compose exec fortran /workspace/fortran/barotropic_instability_spectral_nudging/a.out shear_with_0p40_positive

