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

    return fac*(1+alpha*cos(k*x))*exp(-v*v/2);
}

inline
double u1(double x, double dx, double dx_inv) noexcept
{
	x = std::abs(x);
	if(x > dx){
		return 0;
	}

	return dx_inv * (1 - x*dx_inv);
}

double rho_eval( double x, const arma::vec& v, double dx, double dx_inv, const arma::vec& W) noexcept
{
	arma::vec temp = v;
	//#pragma omp parallel for
	for(size_t i = 0; i < temp.n_rows; i++)
	{
		temp(i) = u1(temp(i), dx, dx_inv);
	}

	return arma::dot(W, temp);
}


int main()
{
    using std::floor;

    // User-defined parameters
    using poisson_t = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<2>;

    const double L  = 4*3.14159265358979323846;
    const size_t Nx = 128;
    const size_t Nv = 256;
    const double v_min = -6;
    const double v_max =  6;

    // Compute derived quantities
    const double dv = (v_max-v_min)/Nv;
    const double dx = L/Nx;
    const double dx_inv = 1/dx;
    const double dv_inv = 1/dv;
    const double L_inv  = 1/L;

    arma::mat xv, xv_stage, k_xv[ 4 ];
    arma::vec W(Nx*Nv); // Masses of particles
    xv.set_size(Nx*Nv,2);
    xv_stage.set_size(Nx*Nv, 2 );
    k_xv[0].set_size( Nx*Nv, 2 ); 
    k_xv[1].set_size( Nx*Nv, 2 ); 
    k_xv[2].set_size( Nx*Nv, 2 ); 
    k_xv[3].set_size( Nx*Nv, 2 ); 

    arma::vec rho(Nx);

    // Runge--Kutta Butcher tableau.
    constexpr double c_rk4[4][4] = { {   0,   0,  0, 0 },
                                     { 0.5,   0,  0, 0 },
                                     {   0, 0.5,  0, 0 },
                                     {   0,   0,  1, 0 } };
    constexpr double d_rk4[4] = { 1./6., 1./3., 1./3., 1./6. };

    poisson_t poisson(0,L,Nx);
    arma::vec quad_nodes = poisson.quadrature_nodes();
    arma::vec quad_vals( quad_nodes.size() );

    // Initialise xv.
    for ( size_t i = 0; i < Nx; ++i )
    for ( size_t j = 0; j < Nv; ++j )
    {
        double x =         (i+0.5)*dx;
        double v = v_min + (j+0.5)*dv;
        xv(j+Nv*i, 0 ) = x;
        xv(j+Nv*i, 1 ) = v;
         W(j+Nv*i)     = dx*dv*f0(x,v);
    }

    double t = 0, T = 30, dt = 1./8.;
    double t_total = 0;
    size_t t_step_count = 0;
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

            for(size_t i = 0; i < Nx*Nv; i++){
            	double x = xv_stage(i,0);
            	xv_stage(i,0) = x - L * std::floor(x*L_inv);
            }

            // Compute densities.
            // Using B-Spline of first order.
			#pragma omp parallel for
            for(size_t i = 0; i < quad_nodes.size(); i++)
            {
            	quad_vals(i) = 1 - rho_eval(quad_nodes(i), xv_stage.col(0), dx, dx_inv, W);
            }


            // Compute E.
            poisson.update_rho( quad_vals );

            // Update k_stage.
            #pragma omp parallel for
            for ( size_t k = 0; k < Nx*Nv; ++k )
                k_xv[stage](k,1) = -poisson.E(xv_stage(k,0));

            if ( stage == 0 )
            {
                str << t << " " << norm(k_xv[stage].col(1),"inf")  << std::endl;
                std::cout << "Max-norm of E: " << norm(k_xv[stage].col(1),"inf") << "." << std::endl;
                std::ofstream rho_str( "rho_" + std::to_string(t) + ".txt" );
                for(size_t i = 0; i < quad_nodes.size(); i++)
                {
                	rho_str << quad_nodes(i) << " " << quad_vals(i) << std::endl;
                }
            }
        }

        for ( size_t s = 0; s < 4; ++s )
            xv += dt*d_rk4[s]*k_xv[s];
        t += dt;

        double elapsed = clock.elapsed();
        t_total += elapsed;
        t_step_count++;
        std::cout << "Time for needed for time-step: " << elapsed << ".\n";
        std::cout << "---------------------------------------" << elapsed << ".\n";

        if ( t + dt > T ) dt = T - t;
    }

    std::cout << "Average time per time step: " << t_total/t_step_count << " s." << std::endl;
    std::cout << "Total time: " << t_total << " s." << std::endl;
}

