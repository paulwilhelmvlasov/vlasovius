/*
 * Copyright (C) 2021 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of vlasovius.
 *
 * vlasovius is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * vlasovius is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * vlasovius; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */

#include <cblas.h>
#include <iostream>
#include <armadillo>

#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/misc/periodic_poisson_1d.h>

inline
double f0( double x, double v ) noexcept
{
    using std::cos;
    using std::exp; 
    constexpr double alpha = 0.01;
    constexpr double k     = 0.5;
    constexpr double fac   = 0.39894228040143267793994;

    return (1+alpha*cos(k*x) *exp(-v*v/2);
}

int main()
{
    // User-defined parameters
    using poisson_t = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

    double L     = 4*3.14159265358979323846;
    size_t Nx    = 20, Nv = 200;
    double v_min = -6, v_max = 6;

    // Compute derived quatntities
    double dv = (v_max-v_min)/Nv;
    double dx = L/Nx;
    double dx_inv = 1/dx;
    double dv_inv = 1/dv;
    double L_inv  = 1/L;

    arma::mat xv, xv_stage, k_xv[ 4 ];
    arma::vec W(Nx*Nv); // Masses of particles
    xv.set_size( Nx*Nv, 2 );
    xv_stage.set_size( Nx*Nv, 2 );
    k_xv[0].set_size( Nx*Nv, 2 ); 
    k_xv[1].set_size( Nx*Nv, 2 ); 
    k_xv[2].set_size( Nx*Nv, 2 ); 
    k_xv[3].set_size( Nx*Nv, 2 ); 

    size_t num_threads = omp_get_max_threads();
    arma::vec rho(Nx);
    arma::mat rho_thread( Nx, num_threads );

    // Runge--Kutta Butcher tableau.
    constexpr double c_rk4[4][4] = { {   0,   0,  0, 0 },
                                     { 0.5,   0,  0, 0 },
                                     {   0, 0.5,  0, 0 },
                                     {   0,   0,  1, 0 } };
    constexpr double d_rk4[4] = { 1./6., 1./3., 1./3., 1./6. };



    poisson_t poisson(0,L,256);
    arma::vec quad_nodes = poisson.quadrature_nodes();
    arma::vec quad_vals( quad_nodes.size() );

    // Initialise xv.
    double v = v_min + dv/2; 
    double x = 0 + dx/2;
    for ( size_t i = 0; i < Nx; ++i )
    for ( size_t j = 0; j < Nv; ++j )
    {
        xv( j + Nv*i, 0 ) = x;
        xv( j + Nv*i, 1 ) = v;
         f( j + Nv*i)     = dx*dv*f0(x,v);
        x += dx; v += dv;
    }

    double t = 0, T = 100, dt = 1./8.;
    std::ofstream str("E.txt");
    while ( t < T )
    {
        std::cout << "t = " << t << ". " << std::endl;
        vlasovius::misc::stopwatch clock;
        for ( size_t stage = 0; stage < 4; ++stage )
        {
            xv_stage = xv;
            for ( size_t s = 0; s < stage; ++s )
                xv_stage += dt*c_rk4[stage][s]*k_xv[s];

            k_xv[ stage ].col(0) = xv_stage.col(1);


            // Compute densities.
            rho_thread.zeros();
            rho.zeros();

            #pragma omp parallel
            {
                size_t threadno = omp_get_thread_num();

                #pragma omp for
                for ( size_t k = 0; k < Nx*Nv; ++k )
                {
                    double x  = xv_stage(k,0);
                           x  = x - L*std::floor(x*L_inv);
                    size_t i = static_cast<size_t>(x*dx_inv);

                    rho_thread(i,thread_no) += w(k);
                }

                #pragma omp critical
                rho += rho_thread.col(thread_no);
            }
            rho *= dx_inv;

            #pragma omp parallel for
            for ( size_t k = 0; k < Nx*Nv; ++k )
            {
                double x = quad_nodes(k);
                quad_vals(k) = rho( static_cast<size_t>(x*dx_inv) );
            }

            poisson.update_rho( quad_vals );

            #pragma omp parallel for
            for ( size_t k = 0; k < Nx*Nv; ++k )
                k_xv[stage](k,1) = -poisson.E( xv_stage(k,0) );

            if ( stage == 0 )
            {
                str << t << " " << norm(k_xv[stage].col(1),"inf")  << std::endl;
                std::cout << "Max-norm of E: " << norm(k_xv[stage].col(1),"inf") << "." << std::endl;
            }
        }

        for ( size_t s = 0; s < 4; ++s )
            xv += dt*d_rk4[s]*k_xv[s];
        t += dt;

        double elapsed = clock.elapsed();
        std::cout << "Time for needed for time-step: " << elapsed << ".\n";
        std::cout << "---------------------------------------" << elapsed << ".\n";

        if ( t + dt > T ) dt = T - t;
    }
}

