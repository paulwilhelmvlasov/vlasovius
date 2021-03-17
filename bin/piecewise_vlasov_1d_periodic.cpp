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

#include <omp.h>
#include <iostream>
#include <armadillo>

#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/tensorised_kernel.h>
#include <vlasovius/interpolators/piecewise_interpolator.h>
#include <vlasovius/integrators/gauss_konrod.h>
#include <vlasovius/misc/periodic_poisson_1d.h>


constexpr size_t order = 6;
using wendland_t       = vlasovius::kernels::wendland<1,order>;
using kernel_t         = vlasovius::kernels::tensorised_kernel<wendland_t>;
using interpolator_t   = vlasovius::interpolators::piecewise_interpolator<kernel_t>;
using poisson_t        = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

int main()
{
	constexpr double tikhonov_mu { 1e-12 };
	constexpr size_t min_per_box = 200;

	double L = 4*3.14159265358979323846;
	arma::rowvec sigma { 3 , 3 };
	wendland_t W;
	kernel_t   K ( W, sigma );
	kernel_t   Kx( W, arma::rowvec { sigma(0) } );
	size_t Nx = 200, Nv = 1000;

	size_t num_threads = omp_get_max_threads();

	// Runge--Kutta Butcher tableau.
	constexpr double c_rk4[4][4] = { {   0,   0,  0, 0 },
	                                 { 0.5,   0,  0, 0 },
	                                 {   0, 0.5,  0, 0 },
	                                 {   0,   0,  1, 0 } };
	constexpr double d_rk4[4] = { 1./6., 1./3., 1./3., 1./6. };


	arma::mat xv, k_xv[ 4 ];
	arma::vec f;

	poisson_t poisson(0,L,512);
	arma::vec rho_points = poisson.quadrature_nodes();
	arma::vec rho( rho_points.size() );
	vlasovius::geometry::kd_tree rho_tree(rho_points);

	// Initialise xv.
	xv.set_size( Nx*Nv,2 );
	f.resize( Nx*Nv );
	for ( size_t i = 0; i < Nx; ++i )
	for ( size_t j = 0; j < Nv; ++j )
	{
		double x = i * (L/Nx);
		double v = -6 + j*(12./(Nv-1));

		xv( j + Nv*i, 0 ) = x;
		xv( j + Nv*i, 1 ) = v;
		constexpr double alpha = 0.01;
		constexpr double k     = 0.5;
		f( j + Nv*i ) = 0.39894228040143267793994 * ( 1. + alpha*std::cos(k*x) ) * std::exp( -v*v/2. );
	}

	double t = 0, T = 100, dt = 1./8.;
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

			interpolator_t sfx { K, xv_stage, f, min_per_box, tikhonov_mu, num_threads };


			rho.fill(1);
			#pragma omp parallel
			{

			arma::vec my_rho(rho.size(),arma::fill::zeros);
			#pragma omp for schedule(dynamic)
			for ( size_t i = 0; i < sfx.cover.n_rows; ++i )
			{
				const auto &local = sfx.local_interpolants[i];
				double x_min = sfx.cover(i,0), v_min = sfx.cover(i,1),
				       x_max = sfx.cover(i,2), v_max = sfx.cover(i,3);

				const arma::mat &X = local.points();
				arma::vec coeff = local.coeffs();
				for ( size_t j = 0; j < coeff.size(); ++j )
				{
					double v = X(j,1);
					coeff(j) *= ( W.integral((v-v_min)/sigma(1)) +
					              W.integral((v_max-v)/sigma(1)) )*sigma(1);
				}

				arma::rowvec box { x_min, x_max };
				arma::uvec idx = rho_tree.index_query(box);
				my_rho(idx) += Kx(rho_points(idx),X)*coeff;
			}

			#pragma omp critical
			rho -= my_rho;
			}

			poisson.update_rho( rho );
			for ( size_t i = 0; i < xv_stage.n_rows; ++i )
				k_xv[stage](i,1) = -poisson.E( xv_stage(i,0) );

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
		std::cout << "---------------------------------------\n";

		if ( t + dt > T ) dt = T - t;
	}
}
