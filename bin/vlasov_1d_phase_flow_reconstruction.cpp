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
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/kernels/periodised_kernel.h>
#include <vlasovius/interpolators/direct_interpolator.h>
#include <vlasovius/integrators/gauss_konrod.h>
#include <vlasovius/interpolators/pou_interpolator.h>
#include <vlasovius/misc/periodic_poisson_1d.h>
#include <vlasovius/misc/xv_kernel.h>

double maxwellian(double v)
{
	return 0.39894228040143267793994 * std::exp(-0.5 * v * v);
}

double lin_landau_f0(double x, double v, double alpha = 0.01, double k = 0.5)
{
	return 0.39894228040143267793994 * ( 1 + alpha * std::cos(k * x) )
			* std::exp( -0.5 * v*v );
}

arma::vec lin_landau_f0(const arma::mat& xv, double alpha = 0.01, double k = 0.5)
{
	return 0.39894228040143267793994 * ( 1 + alpha * arma::cos(k * xv.col(0)) )
				% arma::exp( -0.5 * xv.col(1) % xv.col(1) );
}

double two_stream_f0(double x, double v, double alpha = 0.01, double k = 0.5)
{
    return 0.39894228040143267793994 * v * v * std::exp(-0.5 * v * v )
	        * (1.0 + alpha * std::cos(k * x));
}

arma::vec two_stream_f0(arma::mat xv, double alpha = 0.01, double k = 0.5)
{
    return 0.39894228040143267793994 * xv.col(1) % xv.col(1) % arma::exp(-0.5 * xv.col(1) % xv.col(1) )
	        % (1.0 + alpha * arma::cos(k * xv.col(0)));
}

double bump_on_tail_f0(double x, double v, double alpha = 0.01, double k = 0.5, double nb = 0.1, double vb = 4.5)
{
    return 0.39894228040143267793994 * ((1.0 - nb) * std::exp(-0.5 * v * v)
        + nb * std::exp(-0.5 * (v - vb) * (v - vb)))
        * (1.0 + alpha * std::cos(k * x));
}

template<typename fct_2d_1d>
arma::vec f_comp_phi(const arma::vec& xv, const fct_2d_1d& phi_x, const fct_2d_1d& phi_v)
{
	constexpr double alpha = 0.01;
	constexpr double k = 0.5;
	arma::mat phi_xv = xv;
	phi_xv.col(0) = phi_x(xv);
	phi_xv.col(1) = phi_v(xv);
	return lin_landau_f0(phi_xv);
}


int main()
{
	constexpr size_t order_x = 2;
	constexpr size_t order_v = 2;
	using kernel_t        = vlasovius::xv_kernel<order_x,order_v>;
	using interpolator_t  = vlasovius::interpolators::direct_interpolator<kernel_t>;
	using poisson_t       = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

	using wendland_t = vlasovius::kernels::wendland<1,order_x>;
	wendland_t W;

	size_t res_n = 400;

	double L = 4*3.14159265358979323846, sigma_x  = 2, sigma_v = 2;
	double mu = 1e-10;
	double v_max = 6;
	kernel_t K( sigma_x, sigma_v, L );

	size_t Nx = 32, Nv = 64;

	size_t num_threads = omp_get_max_threads();

	arma::mat init_xv;
	arma::mat curr_xv;
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
	init_xv.set_size( Nx*Nv,2 );
	curr_xv.set_size( Nx*Nv,2 );
	for ( size_t i = 0; i < Nx; ++i )
		for ( size_t j = 0; j < Nv; ++j )
		{
			double x = i * (L/Nx);
			double v = -v_max + j*( 2 * v_max/(Nv-1));

			init_xv( j + Nv*i, 0 ) = x;
			init_xv( j + Nv*i, 1 ) = v;
		}
	curr_xv = init_xv;

	arma::mat plotX( (res_n + 1)*(res_n + 1), 2 );
	arma::vec plotf( (res_n + 1)*(res_n + 1) );
	for ( size_t i = 0; i <= res_n; ++i )
		for ( size_t j = 0; j <= res_n; ++j )
		{
			plotX(j + (res_n + 1)*i,0) = L * i/double(res_n);
			plotX(j + (res_n + 1)*i,1) = 2*v_max * j/double(res_n) - v_max;
			plotf(j + (res_n + 1)*i) = 0;
		}

	size_t count = 0;
	double t = 0, T = 50, dt = 1./4.;
	std::ofstream str("E.txt");
	while ( t < T )
	{
		std::cout << "t = " << t << ". " << std::endl;
		vlasovius::misc::stopwatch clock;

		if ( t + dt > T ) dt = T - t;

		// v-positions are already updated correctly. x positions are updated now:
		curr_xv.col(0) += dt*curr_xv.col(1);             // Move particles
		curr_xv.col(0) -= L * floor(curr_xv.col(0) / L); // Set to periodic positions.

		// Reconstruct inverse phase flow:
		interpolator_t phi_x(K, curr_xv, init_xv.col(0), mu, num_threads);
		interpolator_t phi_v(K, curr_xv, init_xv.col(1), mu, num_threads);


		auto comp = [&](arma::vec xv){
			return f_comp_phi<interpolator_t>(xv, phi_x, phi_v);
		};

		arma::vec rho = vlasovius::integrators::num_rho_1d(comp, rho_points.col(0), v_max, 1e-6, num_threads); // Does this work?

		poisson.update_rho( rho );
		double max_e = 0;
		for ( size_t i = 0; i < curr_xv.n_rows; ++i )
		{
			double E = poisson.E(curr_xv(i,0));
			curr_xv(i,1) += -dt*E;
			max_e = std::max(max_e,std::abs(E));
		}
		str << t << " " << max_e  << std::endl;
		std::cout << "Max-norm of E: " << max_e << "." << std::endl;

		plotf = comp(plotX);
		if ( count++ % 4 == 0 )
		{
			std::ofstream fstr( "f_" + std::to_string(t) + "s.txt" );
			for ( size_t i = 0; i <= res_n; ++i )
			{
				for ( size_t j = 0; j <= res_n; ++j )
				{
					arma::uword index = j + (res_n + 1)*i;
					double x = plotX(index,0);
					double v = plotX(index,1);
					double f = plotf(index) - lin_landau_f0(x,v);
					fstr << x << " " << v
						 << " " << f << std::endl;
				}
				str << "\n";
			}
		}

		double elapsed = clock.elapsed();
		std::cout << "Time for needed for time-step: " << elapsed << ".\n";
		std::cout << "---------------------------------------\n";

		t += dt;

	}

	return 0;
}
