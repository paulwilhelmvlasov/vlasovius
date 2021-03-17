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


constexpr size_t order = 1;
using wendland_t       = vlasovius::kernels::wendland<1,order>;
using kernel_t         = vlasovius::kernels::tensorised_kernel<wendland_t>;
using interpolator_t   = vlasovius::interpolators::piecewise_interpolator<kernel_t>;
using poisson_t        = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

int main()
{
	constexpr double tikhonov_mu { 0 };
	constexpr size_t min_per_box = 200;

	double L = 4*3.14159265358979323846;

	wendland_t W;
	arma::rowvec sigma { L/2, 5 };
	kernel_t   K ( W, sigma );
	kernel_t   Kx( W, arma::rowvec { sigma(0) } );


	size_t Nx = 512, Nv = 1024;
	std::cout << "Number of particles: " << Nx*Nv << ".\n";

	size_t num_threads = omp_get_max_threads();

	arma::mat xv;
	arma::vec f;

	poisson_t poisson(0,L,256);
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
		double v = -5 + j*(10./(Nv-1));

		xv( j + Nv*i, 0 ) = x;
		xv( j + Nv*i, 1 ) = v;
		constexpr double alpha = 0.01;
		constexpr double k     = 0.5;
		f( j + Nv*i ) = 0.39894228040143267793994 * ( 1. + alpha*std::cos(k*x) ) * std::exp( -v*v/2. ) * v*v;
	}

	arma::mat plotX( 401*401, 2 );
	arma::vec plotf( 401*401 );
	for ( size_t i = 0; i <= 400; ++i )
		for ( size_t j = 0; j <= 400; ++j )
		{
			plotX(j + 401*i,0) = L * i/400.;
			plotX(j + 401*i,1) = 8.0 * j/400. - 4.0;
			plotf(j + 401*i) = 0;
		}

	size_t count = 0;
	double t = 0, T = 100, dt = 1./16.;
	std::ofstream str("E.txt");
	while ( t < T )
	{
		std::cout << "t = " << t << ". " << std::endl;
		vlasovius::misc::stopwatch clock;

		if ( t + dt > T ) dt = T - t;

		xv.col(0) += dt*xv.col(1);             // Move particles
		xv.col(0) -= L * floor(xv.col(0) / L); // Set to periodic positions.
		interpolator_t sfx { K, xv, f, min_per_box, tikhonov_mu, num_threads };
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

		poisson.update_rho( rho ); double max_e = 0;
		for ( size_t i = 0; i < xv.n_rows; ++i )
		{
			double E = poisson.E(xv(i,0));
			xv(i,1) += -dt*E;
			max_e = std::max(max_e,std::abs(E));
		}
		str << t << " " << max_e  << std::endl;
		std::cout << "Max-norm of E: " << max_e << "." << std::endl;

		plotf = sfx(plotX);
		if ( count++ % 16 == 0 )
		{
		std::ofstream fstr( "f_" + std::to_string(t) + ".txt" );
		for ( size_t i = 0; i <= 400; ++i )
		{
			for ( size_t j = 0; j <= 400; ++j )
			{
				fstr << plotX(j + 401*i,0) << " " << plotX(j + 401*i,1)
				     << " " << plotf(j+401*i) << std::endl;
			}
			fstr << "\n";
		}

		}

		double elapsed = clock.elapsed();
		std::cout << "Time for needed for time-step: " << elapsed << ".\n";
		std::cout << "---------------------------------------\n";

		t += dt;
	}
}
