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
	constexpr double tikhonov_mu { 1e-12 };
	constexpr size_t min_per_box = 200;

	double L = 4 * 3.14159265358979323846;
	//double L = 2*3.14159265358979323846 / 0.3; // Bump on tail

	wendland_t W;
	arma::rowvec sigma { 2, 1 };
	kernel_t   K ( W, sigma );
	kernel_t   Kx( W, arma::rowvec { sigma(0) } );


	size_t Nx = 32, Nv = 4*Nx;
	std::cout << "Number of particles: " << Nx*Nv << ".\n";

	size_t num_threads = 6;//omp_get_max_threads();

	arma::mat xv;
	arma::vec f;

	size_t N_poisson = 256;
	poisson_t poisson(0,L,N_poisson);
	arma::vec rho_points = poisson.quadrature_nodes();
	arma::vec rho( rho_points.size() );
	vlasovius::geometry::kd_tree rho_tree(rho_points);

	double vmax = 8;

	// Initialise xv.
	xv.set_size( Nx*Nv,2 );
	f.resize( Nx*Nv );
	for ( size_t i = 0; i < Nx; ++i )
	for ( size_t j = 0; j < Nv; ++j )
	{
		double x = (i+0.5) * (L/Nx);
		double v = -vmax + j*(2*vmax/(Nv-1));

		xv( j + Nv*i, 0 ) = x;
		xv( j + Nv*i, 1 ) = v;
		constexpr double alpha = 0.01;
		constexpr double k     = 0.5;
		// Linear Landau damping:
/*
		f( j + Nv*i ) = 0.39894228040143267793994 * ( 1. + alpha*std::cos(k*x) )
					* std::exp(-0.5 * v * v);
*/

		// Two Stream Instability:

		f( j + Nv*i ) = 0.39894228040143267793994 * ( 1. + alpha*std::cos(k*x) )
						* v * v * std::exp(-0.5 * v * v);



		// Bump on tail benchmark:
		/*
		constexpr double alpha = 0.04;
		constexpr double k     = 0.3;
		constexpr double np = 0.9;
		constexpr double nb = 0.2;
		constexpr double vb = 4.5;
		constexpr double vt = 0.5;
		f( j + Nv*i ) = 0.39894228040143267793994 * ( 1. + alpha*std::cos(k*x) )
				* (np * std::exp( -0.5 * v*v )
				+  nb * std::exp( -0.5 * (v-vb)*(v-vb) / (vt * vt)) );
		*/
		// Expansion into a uniform ion background:
		//f( j + Nv*i ) = 0.39894228040143267793994 * std::exp(-0.5 * v*v) * std::exp(-0.5 * (x-2*M_PI)*(x-2*M_PI));
	}

	size_t res = 256;
	double hx_plot = L / res;
	double hv_plot = 2.0 * vmax / res;
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
	size_t timeStepCounter = 0;
	double totalTime = 0;
	double t = 0, T = 100, dt = 1./16.;
	std::ofstream e_amp_str("E.txt");
	std::ofstream e_l2_str("E_l2.txt");
    std::ofstream statistics_file( "statistics.csv" );
    statistics_file << R"("Time"; "L1-Norm"; "L2-Norm"; "Electric Energy"; "Kinetic Energy"; "Total Energy"; "Entropy")";
    statistics_file << std::endl;

    statistics_file << std::scientific;
    std::cout << std::scientific;
    double l1_norm = 0;
    double l2_norm = 0;
    double kinetic_energy = 0;
    double entropy = 0;
	while ( t <= T )
	{
		std::cout << "t = " << t << ". " << std::endl;
		vlasovius::misc::stopwatch clock;

		//if ( t + dt > T ) dt = T - t;


		if ( count % (25*16) == 0)
		{
			interpolator_t sfx { K, xv, f, min_per_box, tikhonov_mu, num_threads };
			plotf = sfx(plotX);
			std::ofstream fstr( "f_" + std::to_string(t) + ".txt" );
			for ( size_t i = 0; i <= res; ++i )
			{
				for ( size_t j = 0; j <= res; ++j )
				{
					double x = plotX(j + (res + 1)*i,0);
					double v = plotX(j + (res + 1)*i,1);
					double f = plotf(j + (res+1)*i);

					fstr << x << " " << v << " " << f << std::endl;

					kinetic_energy += v*v*f;
					if(f > 1e-10)
					{
						entropy += f*std::log(f);
					}
					l1_norm += std::abs(f);
					l2_norm += f*f;
				}
				fstr << "\n";
			}

			kinetic_energy *= 0.5*hx_plot*hv_plot;
			entropy *= hx_plot*hv_plot;
			l1_norm *= hx_plot*hv_plot;
			l2_norm = std::sqrt(l2_norm);
			l2_norm *= hx_plot*hv_plot;
		}



		xv.col(0) += dt*xv.col(1);             // Move particles
		xv.col(0) -= L * floor(xv.col(0) / L); // Set to periodic positions.

		interpolator_t sfx { K, xv, f, min_per_box, tikhonov_mu, num_threads };
		rho.fill(1);
		#pragma omp parallel
		{
			arma::vec my_rho(rho.size(),arma::fill::zeros);
			arma::vec coeff( 2*min_per_box );

			#pragma omp for schedule(dynamic)
			for ( size_t i = 0; i < sfx.cover.n_rows; ++i )
			{
				const auto &local = sfx.local_interpolants[i];
				double x_min = sfx.cover(i,0), v_min = sfx.cover(i,1),
				       x_max = sfx.cover(i,2), v_max = sfx.cover(i,3);

				size_t n = local.points().n_rows;
				const arma::mat &X = local.points();
				const arma::mat &Xx = X.col(0);

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
				my_rho(idx) += Kx(rho_points(idx),Xx)*coeff( arma::span(0,n-1) );
			}

			#pragma omp critical
			rho -= my_rho;
		}

		poisson.update_rho( rho );
		double max_e = 0;
		for ( size_t i = 0; i < xv.n_rows; ++i )
		{
			double E = poisson.E(xv(i,0));
			xv(i,1) += -dt*E;
			max_e = std::max(max_e,std::abs(E));
		}

		double l2_e = 0;
		for(size_t i = 0; i < res; i++)
		{
			double x = i*hx_plot;
			double E = poisson.E(x);
			l2_e += E*E;
		}

		l2_e *= 0.5*hx_plot;
		e_amp_str << t << " " << max_e  << std::endl;
		e_l2_str << t << " " << l2_e  << std::endl;
		std::cout << "Max-norm of E: " << max_e << "." << std::endl;

		/*
		if ( count % (2*16) == 0 )
		{
            statistics_file << t       << "; "
                            << l1_norm      << "; "
                            << l2_norm      << "; "
                            << l2_e << "; "
                            << kinetic_energy << "; "
                            << l2_e + kinetic_energy << "; "
                            << entropy << std::endl;

            l1_norm = 0;
            l2_norm = 0;
            kinetic_energy = 0;
            entropy = 0;

			std::ofstream Estr( "E_" + std::to_string(t) + ".txt" );
			for ( size_t i = 0; i <= res; ++i )
			{
				double x = i*hx_plot;
				double E = poisson.E(x);

				Estr << x << " " << E << std::endl;
			}
		}
		*/

		double elapsed = clock.elapsed();
		timeStepCounter++;
		totalTime += elapsed;
		std::cout << "Time for needed for time-step: " << elapsed << ".\n";
		std::cout << "---------------------------------------\n";

		t += dt;
		count++;
	}
	std::cout << "Total simulation time: " << totalTime << std::endl;
	std::cout << "Average time for a time-step: " << totalTime / timeStepCounter << std::endl;
}
