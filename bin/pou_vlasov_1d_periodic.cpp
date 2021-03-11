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

#include <iostream>
#include <armadillo>

#include <omp.h>

#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/kernels/periodised_kernel.h>
#include <vlasovius/integrators/gauss_konrod.h>
#include <vlasovius/interpolators/direct_interpolator.h>
#include <vlasovius/interpolators/pou_interpolator.h>
#include <vlasovius/interpolators/periodic_pou_interpolator.h>
#include <vlasovius/misc/periodic_poisson_1d.h>


int main()
{
	constexpr size_t order = 4;
	using kernel_t        = vlasovius::kernels::rbf_kernel<vlasovius::kernels::wendland<2,4>>; //vlasovius::xv_kernel<order,4>;
	using interpolator_t  = vlasovius::interpolators::periodic_pou_interpolator<kernel_t>;
	using poisson_t       = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

	using wendland_t 	  = vlasovius::kernels::wendland<1,4>;
	wendland_t W;

	constexpr double tikhonov_mu { 1e-13 };
	constexpr size_t min_per_box = 300;
	constexpr double enlarge 	 = 1.5;

	constexpr double v_max = 10.0;

	double L = 4*3.14159265358979323846, sigma  = 2.0;
	arma::rowvec bounding_box { 0, -v_max, L, v_max };

	kernel_t K( {}, sigma );

	size_t Nx = 50, Nv = 100;

	size_t num_threads = omp_get_max_threads();

	// Runge--Kutta Butcher tableau.
	constexpr double c_rk4[4][4] = { {   0,   0,  0, 0 },
	                                 { 0.5,   0,  0, 0 },
	                                 {   0, 0.5,  0, 0 },
	                                 {   0,   0,  1, 0 } };
	constexpr double d_rk4[4] = { 1./6., 1./3., 1./3., 1./6. };


	arma::mat xv, k_xv[ 4 ];
	arma::vec f;

	poisson_t poisson(0,L,256);
	arma::mat rho_points;
	{
		arma::vec rho_tmp = poisson.quadrature_nodes();
		rho_points.set_size( rho_tmp.n_rows, 2 );
		for ( size_t i = 0; i < rho_tmp.size(); ++i )
		{
			rho_points(i,0) = rho_tmp(i);
			rho_points(i,1) = 0;
		}
	}

	// Initialise xv.
	xv.set_size( Nx*Nv,2 );
	f.resize( Nx*Nv );
	for ( size_t i = 0; i < Nx; ++i )
	for ( size_t j = 0; j < Nv; ++j )
	{
		double x = i * (L/Nx);
		double v = -v_max + j*( 2 * v_max/(Nv-1));

		xv( j + Nv*i, 0 ) = x;
		xv( j + Nv*i, 1 ) = v;
		constexpr double alpha = 0.01;
		constexpr double k     = 0.5;
		f( j + Nv*i ) = 0.39894228040143267793994 * ( 1. + alpha*std::cos(k*x) ) * std::exp( -v*v/2. );
	}

	arma::mat plotX( 101*101, 2 );
	arma::vec plotf( 101*101 );
	for ( size_t i = 0; i <= 100; ++i )
		for ( size_t j = 0; j <= 100; ++j )
		{
			plotX(j + 101*i,0) = L * i/100.;
			plotX(j + 101*i,1) = 20.0 * j/100. - 10.0;
			plotf(j + 101*i) = 0;
		}

	double t = 0, T = 100, dt = 1./4.;
	std::ofstream str("E.txt");
	while ( t < T )
	{
		std::cout << "t = " << t << ". " << std::endl;
		vlasovius::misc::stopwatch clock;
		for ( size_t stage = 0; stage < 4; ++stage )
		{
			arma::mat xv_stage = xv;
			for ( size_t s = 0;  s+1 <= stage; ++s )
				xv_stage += dt*c_rk4[stage][s]*k_xv[s];

			k_xv[ stage ].resize( xv.n_rows, xv.n_cols );
			k_xv[ stage ].col(0) = xv_stage.col(1);

			xv_stage.col(0) -= L * floor(xv_stage.col(0) / L);

			interpolator_t sfx { K, xv_stage, f, bounding_box, enlarge, min_per_box, tikhonov_mu };

			arma::vec rho = vlasovius::integrators::num_rho_1d(sfx, rho_points.col(0),
					v_max, 1e-8, num_threads);

			poisson.update_rho( rho );

			for ( size_t i = 0; i < xv_stage.n_rows; ++i )
				k_xv[stage](i,1) = -poisson.E( xv_stage(i,0) );

			if ( stage == 0 )
			{
				str << t << " " << norm(k_xv[stage].col(1),"inf")  << std::endl;
				std::cout << "Max-norm of E: " << norm(k_xv[stage].col(1),"inf") << "." << std::endl;

				plotf = sfx(plotX);
				std::ofstream str( "f_" + std::to_string(t) + "s.txt" );
				for ( size_t i = 0; i <= 100; ++i )
				{
					for ( size_t j = 0; j <= 100; ++j )
					{
						str << plotX(j + 101*i,0) << " " << plotX(j + 101*i,1)
								<< " " << plotf(j+101*i) << std::endl;
					}
					str << "\n";
				}
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


