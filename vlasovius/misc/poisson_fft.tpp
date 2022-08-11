/*
 * Copyright (C) 2022 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of Der Gerät, a solver for the Vlasov–Poisson equation.
 *
 * Der Gerät is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * Der Gerät is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Der Gerät; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */
#include <vlasovius/misc/poisson_fft.h>

#include <cmath>
#include <memory>
#include <cstdlib>
#include <stdexcept>

namespace vlasovius
{

namespace dim1
{

    poisson<double>::poisson(double Lx, size_t Nx):
        Lx { Lx }, Lx_inv{1.0 / Lx}, Nx{Nx}
    {
        using memptr = std::unique_ptr<double,decltype(std::free)*>;

        size_t mem_size  = sizeof(double) * Nx;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<double*>(tmp), std::free };

        plan = fftw_plan_r2r_1d( Nx, mem.get(), mem.get(), FFTW_DHT, FFTW_MEASURE );
    }

    poisson<double>::~poisson()
    {
        fftw_destroy_plan( plan );
    }

    void poisson<double>::solve( double *data ) const noexcept
    {
        fftw_execute_r2r( plan, data, data );

        const double fac_N = double(1) / double( Nx );
        const double fac_x = (2*M_PI*Lx_inv) * (2*M_PI*Lx_inv);
        for ( size_t i = 1; i < Nx; i++ )
        {
            double ii = (2*i < Nx) ? i : Nx - i; ii *= ii;
            double fac = fac_N/(ii*fac_x);
            data[i] *= fac;
        }
        data[0] = 0;
        
        fftw_execute_r2r( plan, data, data );
    }



    poisson<float>::poisson( float Lx, size_t Nx):
        Lx{Lx}, Lx_inv{1/Lx}, Nx{Nx}
    {
        using memptr = std::unique_ptr<float,decltype(std::free)*>;

        size_t mem_size  = sizeof(float) * Nx;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<float*>(tmp), std::free };

        plan = fftwf_plan_r2r_1d( Nx, mem.get(), mem.get(), FFTW_DHT, FFTW_MEASURE );
    }

    poisson<float>::~poisson()
    {
        fftwf_destroy_plan( plan );
    }

    void poisson<float>::solve( float *data ) const noexcept
    {
        fftwf_execute_r2r( plan, data, data );

        const float fac_N = float(1) / float( Nx );
        const float fac_x = (2*M_PI*Lx_inv) * (2*M_PI*Lx_inv);
        for ( size_t i = 1; i < Nx; i++ )
        {
            float ii = (2*i < Nx) ? i : Nx - i; ii *= ii;
            float fac = fac_N/(ii*fac_x);
            data[i] *= fac;
        }
        data[0] = 0;
        
        fftwf_execute_r2r( plan, data, data );
    }
}

namespace dim2
{

    poisson<double>::poisson( double Lx, double Ly, size_t Nx, size_t Ny ):
        Lx{Lx}, Lx_inv{1/Lx}, Ly{Ly}, Ly_inv{1/Ly}, Nx{Nx}, Ny{Ny}
    {
        using memptr = std::unique_ptr<double,decltype(std::free)*>;

        size_t mem_size  = sizeof(double) * Nx * Ny;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<double*>(tmp), std::free };

        plan = fftw_plan_r2r_2d( Ny, Nx, mem.get(), mem.get(),
                                 FFTW_DHT, FFTW_DHT, FFTW_MEASURE );
    }

    poisson<double>::~poisson()
    {
        fftw_destroy_plan( plan );
    }

    void poisson<double>::solve( double *data ) const noexcept
    {
        fftw_execute_r2r( plan, data, data );

        const double fac_N = double(1) / double( Nx * Ny );
        const double fac_x = (2*M_PI*Lx_inv) * (2*M_PI*Lx_inv);
        const double fac_y = (2*M_PI*Ly_inv) * (2*M_PI*Ly_inv);
        for ( size_t j = 0; j < Ny; j++ )
        for ( size_t i = 0; i < Nx; i++ )
        {
            double ii = (2*i < Nx) ? i : Nx - i; ii *= ii;
            double jj = (2*j < Ny) ? j : Ny - j; jj *= jj;
            double fac = fac_N/(ii*fac_x + jj*fac_y);
            data[ j*Nx + i ] *= fac;
        }
        data[0] = 0;
        
        fftw_execute_r2r( plan, data, data );
    }



    poisson<float>::poisson( float Lx, float Ly, size_t Nx, size_t Ny ):
        Lx{Lx},Lx_inv{1/Lx}, Ly{Ly}, Ly_inv{1/Ly}, Nx{Nx}, Ny{Ny}
    {
        using memptr = std::unique_ptr<float,decltype(std::free)*>;

        size_t mem_size  = sizeof(float) * Nx * Ny;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<float*>(tmp), std::free };
    
        plan = fftwf_plan_r2r_2d( Ny, Nx, mem.get(), mem.get(),
                                 FFTW_DHT, FFTW_DHT, FFTW_MEASURE );
    }

    poisson<float>::~poisson()
    {
        fftwf_destroy_plan( plan );
    }

    void poisson<float>::solve( float *data ) const noexcept
    {
        fftwf_execute_r2r( plan, data, data );

        const float fac_N = float(1) / float( Nx * Ny );
        const float fac_x = (2*M_PI*Lx_inv) * (2*M_PI*Lx_inv);
        const float fac_y = (2*M_PI*Ly_inv) * (2*M_PI*Ly_inv);
        for ( size_t j = 0; j < Ny; j++ )
        for ( size_t i = 0; i < Nx; i++ )
        {
            float ii = (2*i < Nx) ? i : Nx - i; ii *= ii;
            float jj = (2*j < Ny) ? j : Ny - j; jj *= jj;
            float fac = fac_N/(ii*fac_x + jj*fac_y);
            data[ j*Nx + i ] *= fac;
        }
        data[0] = 0;
        
        fftwf_execute_r2r( plan, data, data );
    }
}

namespace dim3
{

    poisson<double>::poisson( double Lx, double Ly, double Lz, size_t Nx, size_t Ny, size_t Nz ):
          Lx{Lx},Lx_inv{1/Lx}, Ly{Ly}, Ly_inv{1/Ly},
		  Lz{Lz}, Lz_inv{1/Lz}, Nx{Nx}, Ny{Ny}, Nz{Nz}
    {
        using memptr = std::unique_ptr<double,decltype(std::free)*>;

        size_t mem_size  = sizeof(double) * Nx * Ny * Nz;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<double*>(tmp), std::free };

        plan = fftw_plan_r2r_3d( Nz, Ny, Nx, mem.get(), mem.get(),
                                 FFTW_DHT, FFTW_DHT, FFTW_DHT, FFTW_MEASURE );
    }

    poisson<double>::~poisson()
    {
        fftw_destroy_plan( plan );
    }

    void poisson<double>::solve( double *data ) const noexcept
    {
        fftw_execute_r2r( plan, data, data );

        const double fac_N = double(1) / double( Nx * Ny * Nz );
        const double fac_x = (2*M_PI*Lx_inv) * (2*M_PI*Lx_inv);
        const double fac_y = (2*M_PI*Ly_inv) * (2*M_PI*Ly_inv);
        const double fac_z = (2*M_PI*Lz_inv) * (2*M_PI*Lz_inv);
        for ( size_t k = 0; k < Nz; k++ )
        for ( size_t j = 0; j < Ny; j++ )
        for ( size_t i = 0; i < Nx; i++ )
        {
            double ii = (2*i < Nx) ? i : Nx - i; ii *= ii;
            double jj = (2*j < Ny) ? j : Ny - j; jj *= jj;
            double kk = (2*k < Nz) ? k : Nz - k; kk *= kk;
            double fac = fac_N/(ii*fac_x + jj*fac_y + kk*fac_z);
            data[ k*Nx*Ny + j*Nx + i ] *= fac;
        }
        data[ 0 ] = 0;
        
        fftw_execute_r2r( plan, data, data );
    }



    poisson<float>::poisson( float Lx, float Ly, float Lz, size_t Nx, size_t Ny, size_t Nz ):
        Lx{Lx},Lx_inv{1/Lx}, Ly{Ly}, Ly_inv{1/Ly},
        Lz{Lz}, Lz_inv{1/Lz}, Nx{Nx}, Ny{Ny}, Nz{Nz}
    {
        using memptr = std::unique_ptr<float,decltype(std::free)*>;

        size_t mem_size  = sizeof(float) * Nx * Ny * Nz;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<float*>(tmp), std::free };
    
        plan = fftwf_plan_r2r_3d( Nz, Ny, Nx, mem.get(), mem.get(),
                                  FFTW_DHT, FFTW_DHT, FFTW_DHT, FFTW_MEASURE );
    }

    poisson<float>::~poisson()
    {
        fftwf_destroy_plan( plan );
    }

    void poisson<float>::solve( float *data ) const noexcept
    {
        fftwf_execute_r2r( plan, data, data );

        const float fac_N = float(1) / float( Nx * Ny * Nz );
        const float fac_x = (2*M_PI*Lx_inv) * (2*M_PI*Lx_inv);
        const float fac_y = (2*M_PI*Ly_inv) * (2*M_PI*Ly_inv);
        const float fac_z = (2*M_PI*Lz_inv) * (2*M_PI*Lz_inv);
        for ( size_t k = 0; k < Nz; k++ )
        for ( size_t j = 0; j < Ny; j++ )
        for ( size_t i = 0; i < Nx; i++ )
        {
            float ii = (2*i < Nx) ? i : Nx - i; ii *= ii;
            float jj = (2*j < Ny) ? j : Ny - j; jj *= jj;
            float kk = (2*k < Nz) ? k : Nz - k; kk *= kk;
            float fac = fac_N/(ii*fac_x + jj*fac_y + kk*fac_z);
            data[ k*Nx*Ny + j*Nx + i ] *= fac;
        }
        data[0] = 0;
        
        fftwf_execute_r2r( plan, data, data );
    }
}

}

