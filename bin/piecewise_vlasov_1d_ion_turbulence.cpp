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


constexpr size_t order = 2;
using wendland_t       = vlasovius::kernels::wendland<1,order>;
using kernel_t         = vlasovius::kernels::tensorised_kernel<wendland_t>;
using interpolator_t   = vlasovius::interpolators::piecewise_interpolator<kernel_t>;
using poisson_t        = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

int main()
{
	constexpr double tikhonov_mu { 1e-10 };
	constexpr size_t min_per_box = 200;

	double L = 40 * 3.14159265358979323846;
	double vmax = 8;
	double Mr = 1000;
	double Ue = -2;

	wendland_t W;
	arma::rowvec sigma { 4, 2 };
	kernel_t   K ( W, sigma );
	kernel_t   Kx( W, arma::rowvec { sigma(0) } );

	size_t Nx_ion = 128, Nv_ion = 256;
	size_t Nx_electron = 512, Nv_electron = 512;
	std::cout << "Number of ion-particles: " << Nx_ion*Nv_ion << ".\n";
	std::cout << "Number of ion-particles: " << Nx_electron*Nv_electron << ".\n";

	size_t num_threads = omp_get_max_threads();


	arma::mat xv_ion;
	arma::vec f_ion;
	arma::mat xv_electron;
	arma::vec f_electron;


	poisson_t poisson(0,L,256);
	arma::vec rho_points = poisson.quadrature_nodes();
	arma::vec rho( rho_points.size() );
	arma::vec rho_ion( rho_points.size() );
	arma::vec rho_electron( rho_points.size() );
	vlasovius::geometry::kd_tree rho_tree(rho_points);

	// Initialise Ion-distribution:
	xv_ion.set_size( Nx_ion*Nv_ion,2 );
	f_ion.resize( Nx_ion*Nv_ion );
	for ( size_t i = 0; i < Nx_ion; ++i )
	for ( size_t j = 0; j < Nv_ion; ++j )
	{
		double x = (i+0.5) * (L/Nx_ion);
		double v = -vmax + j*(2*vmax/(Nv_ion-1));

		xv_ion( j + Nv_ion*i, 0 ) = x;
		xv_ion( j + Nv_ion*i, 1 ) = v;

		f_ion( j + Nv_ion*i ) = std::sqrt(Mr / (2.0 * M_PI)) * std::exp(-0.5 * Mr * v * v);
	}

	// Initialise Electron-distribution:
	xv_electron.set_size( Nx_electron*Nv_electron,2 );
	f_electron.resize( Nx_electron*Nv_electron );
	for ( size_t i = 0; i < Nx_electron; ++i )
	for ( size_t j = 0; j < Nv_electron; ++j )
	{
		double x = (i+0.5) * (L/Nx_electron);
		double v = -vmax + j*(2*vmax/(Nv_electron-1));

		xv_electron( j + Nv_electron*i, 0 ) = x;
		xv_electron( j + Nv_electron*i, 1 ) = v;

		f_electron( j + Nv_electron*i ) = std::sqrt(1.0 / (2.0 * M_PI))
				* std::exp(-0.5 * (v - Ue) * (v - Ue) )
				* 0.01 * (std::sin(x) + std::sin(0.5 * x)
						+ std::sin(0.1 * x) + std::sin(0.15 * x)
						+ std::sin(0.2 * x) + std::cos(0.25 * x)
						+ std::cos(0.3 * x) + std::cos(0.35 * x));
	}

	size_t res = 200;
	arma::mat plotX( (res + 1)*(res + 1), 2 );
	arma::vec plotf( (res + 1)*(res + 1));
	for ( size_t i = 0; i <= res; ++i )
		for ( size_t j = 0; j <= res; ++j )
		{
			plotX(j + (res + 1)*i,0) = L * i/double(res);
			plotX(j + (res + 1)*i,1) = 2*vmax * j/double(res) - vmax;
			plotf(j + (res + 1)*i) = 0;
		}

	size_t count = 0;
	double t = 0, T = 30.25, dt = 1./32.;
	std::ofstream str("E.txt");
	vlasovius::misc::stopwatch global_clock;
	while ( t < T )
	{
		std::cout << "t = " << t << ". " << std::endl;
		vlasovius::misc::stopwatch clock;

		if ( count++ % 32 == 0 )
		{
			interpolator_t sfx_ion { K, xv_ion, f_ion, min_per_box, tikhonov_mu, num_threads };
			interpolator_t sfx_electron { K, xv_electron, f_electron, min_per_box, tikhonov_mu, num_threads };

			plotf = sfx_ion(plotX);
			std::ofstream f_ion_str( "f_ion_" + std::to_string(t) + ".txt" );
			for ( size_t i = 0; i <= res; ++i )
			{
				for ( size_t j = 0; j <= res; ++j )
				{
					double x = plotX(j + (res + 1)*i,0);
					double v = plotX(j + (res + 1)*i,1);
					double f = plotf(j + (res+1)*i);

					f_ion_str << x << " " << v << " " << f << std::endl;
				}
				f_ion_str << "\n";
			}
			plotf = sfx_electron(plotX);
			std::ofstream f_electron_str( "f_electron_" + std::to_string(t) + ".txt" );
			for ( size_t i = 0; i <= res; ++i )
			{
				for ( size_t j = 0; j <= res; ++j )
				{
					double x = plotX(j + (res + 1)*i,0);
					double v = plotX(j + (res + 1)*i,1);
					double f = plotf(j + (res+1)*i);

					f_electron_str << x << " " << v << " " << f << std::endl;
				}
				f_electron_str << "\n";
			}
		}


		// Move x-particles:
		xv_ion.col(0) += dt*xv_ion.col(1);             // Move particles
		xv_ion.col(0) -= L * floor(xv_ion.col(0) / L); // Set to periodic positions.

		xv_electron.col(0) += dt*xv_electron.col(1);             // Move particles
		xv_electron.col(0) -= L * floor(xv_electron.col(0) / L); // Set to periodic positions.

		//Interpolate f_ion and f_electron
		interpolator_t sfx_ion { K, xv_ion, f_ion, min_per_box, tikhonov_mu, num_threads };
		interpolator_t sfx_electron { K, xv_electron, f_electron, min_per_box, tikhonov_mu, num_threads };

		// Compute rho_ion:
		rho_ion.fill(0);
		#pragma omp parallel
		{
			arma::vec my_rho(rho_ion.size(),arma::fill::zeros);
			arma::vec coeff( 2*min_per_box );

			#pragma omp for schedule(dynamic)
			for ( size_t i = 0; i < sfx_ion.cover.n_rows; ++i )
			{
				const auto &local = sfx_ion.local_interpolants[i];
				double x_min = sfx_ion.cover(i,0), v_min = sfx_ion.cover(i,1),
						x_max = sfx_ion.cover(i,2), v_max = sfx_ion.cover(i,3);

				size_t n = local.points().n_rows;
				const arma::mat &X = local.points();

				if ( n > coeff.size() ) coeff.resize(n);

				for ( size_t j = 0; j < n; ++j )
				{
					double v = X(j,1);
					coeff(j) = local.coeffs()(j) * sigma(1) *
							   ( W.integral((v-v_min)/sigma(1)) +
					             W.integral((v_max-v)/sigma(1)) );
				}

				arma::rowvec box { x_min, x_max };
				arma::uvec idx = rho_tree.index_query(box);
				my_rho(idx) += Kx(rho_points(idx),X)*coeff( arma::span(0,n-1) );
			}

			#pragma omp critical
			rho_ion -= my_rho;
		}

		// Compute rho_electron:
		rho_electron.fill(0);
		#pragma omp parallel
		{
			arma::vec my_rho(rho_electron.size(),arma::fill::zeros);
			arma::vec coeff( 2*min_per_box );

			#pragma omp for schedule(dynamic)
			for ( size_t i = 0; i < sfx_electron.cover.n_rows; ++i )
			{
				const auto &local = sfx_electron.local_interpolants[i];
				double x_min = sfx_electron.cover(i,0), v_min = sfx_electron.cover(i,1),
					x_max = sfx_electron.cover(i,2), v_max = sfx_electron.cover(i,3);

				size_t n = local.points().n_rows;
				const arma::mat &X = local.points();

				if ( n > coeff.size() ) coeff.resize(n);

				for ( size_t j = 0; j < n; ++j )
				{
					double v = X(j,1);
					coeff(j) = local.coeffs()(j) * sigma(1) *
							   ( W.integral((v-v_min)/sigma(1)) +
				             W.integral((v_max-v)/sigma(1)) );
				}

				arma::rowvec box { x_min, x_max };
				arma::uvec idx = rho_tree.index_query(box);
				my_rho(idx) += Kx(rho_points(idx),X)*coeff( arma::span(0,n-1) );
			}

			#pragma omp critical
			rho_electron -= my_rho;
		}

		// Global rho:
		rho = rho_ion - rho_electron;

		// Solving Poisson equation:
		poisson.update_rho( rho );

		// Advancing v-particles:
		double max_e = 0;
		for ( size_t i = 0; i < xv_ion.n_rows; ++i )
		{
			double E = poisson.E(xv_ion(i,0));
			xv_ion(i,1) += (1.0 / Mr) * dt*E;
			max_e = std::max(max_e,std::abs(E));
		}
		for ( size_t i = 0; i < xv_electron.n_rows; ++i )
		{
			double E = poisson.E(xv_electron(i,0));
			xv_electron(i,1) += -dt*E;
			max_e = std::max(max_e,std::abs(E));
		}
		str << t << " " << max_e  << std::endl;
		std::cout << "Max-norm of E: " << max_e << "." << std::endl;
		double elapsed = clock.elapsed();
		std::cout << "Time for needed for time-step: " << elapsed << ".\n";
		std::cout << "---------------------------------------\n";

		t += dt;
		}

	double global_elapsed = global_clock.elapsed();
	std::cout << "Total comp time: " << global_elapsed << "s." << std::endl;
}
