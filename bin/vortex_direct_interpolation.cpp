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
#include <vlasovius/interpolators/direct_interpolator.h>

double w0(double x, double y)
{
	double r = std::sqrt(x*x + y*y);
	if (r < 0.8)
		return 20 * (1 - r/0.8);

	return 0;
}

arma::rowvec K_h_sigma(arma::rowvec x, arma::rowvec y, double sigma )
{
	arma::vec dist = x - y;
	double r = arma::norm(dist, 2);

	arma::rowvec result = {dist(1), -dist(0)};

	// For ease only for sigma = 1.

	if(r > 1)
	{
		return -1.0/(6*r*r)/3.0 * result;
	}

	return (result / 18.0) * ( 21*std::pow(r,8) - 128*std::pow(r,7)
			+ 315*std::pow(r,6) - 384*std::pow(r,5) + 210*std::pow(r,4)
			- 42*r*r + 9);
}

arma::rowvec u_h_sigma(arma::rowvec x, const arma::mat& xy, const arma::vec& coeff, double sigma)
{
	arma::rowvec u = {0,0};
	size_t N = xy.n_rows;

	for(size_t i = 0; i < N; i++)
	{
		u += coeff(i) * K_h_sigma(x,xy.row(i),sigma);
	}

	return u;
}




int main()
{

	using wendland_t	  = vlasovius::kernels::wendland<2,2>;
	using kernel_t        = vlasovius::kernels::rbf_kernel<wendland_t>;
	using interpolator_t  = vlasovius::interpolators::direct_interpolator<kernel_t>;

	double Lx = 1;
	double Ly = 1;
	double sigma = 1;
	double mu = 1e-12;
	kernel_t K( wendland_t {}, sigma);

	//wendland_t W;
	//std::cout << W(0) << std::endl;
	//std::cout << W(1) << std::endl;

	size_t num_threads = omp_get_max_threads();

	double visc_nu = 0;

	// Initialise xy.
	arma::mat xy;
	arma::vec w;
	size_t Nx = 32, Ny = 32;
	size_t N = Nx*Ny;
	xy.set_size( N,2 );
	w.resize( N );
	for ( size_t i = 0; i < Nx; ++i )
	for ( size_t j = 0; j < Ny; ++j )
	{
		double x = -1 +  i * (2*Lx/Nx);
		double y = -1 +  j * (2*Ly/Ny);

		xy( j + Ny*i, 0 ) = x;
		xy( j + Ny*i, 1 ) = y;
		w ( j + Ny*i) = w0(x,y);
	}

	// Plot-stuff.
	size_t res = 100;
	double hx_plot = 2.0 / res;
	double hy_plot = 2.0 / res;
	arma::mat plotX( (res + 1)*(res + 1), 2 );
	arma::vec plotf( (res + 1)*(res + 1));
	for ( size_t i = 0; i <= res; ++i )
		for ( size_t j = 0; j <= res; ++j )
		{
			plotX(j + (res + 1)*i,0) = -1 +  i * hx_plot;
			plotX(j + (res + 1)*i,1) = -1 +  j * hy_plot;
			plotf(j + (res + 1)*i) = 0;
		}


	size_t count = 0;
	size_t one_step = 1024;
	double t = 0, T = 10.25, dt = 1./double(one_step);
	while ( t < T )
	{
		std::cout << "t = " << t << ". " << std::endl;

		vlasovius::misc::stopwatch clock;
		// We use explicit Euler for simplicity:
		// Compute the interpolant:
		interpolator_t omega(K, xy, w, mu, num_threads);
		double h_la = 0.1;
		arma::rowvec left = {-h_la, 0};
		arma::rowvec right = {h_la, 0};
		arma::rowvec top = {0, -h_la};
		arma::rowvec bot = {0, h_la};
		#pragma omp parallel for
		for(size_t i = 0; i < N; i++)
		{
			w(i) = w(i) + dt * visc_nu / (h_la * h_la)
					* ( omega(xy.row(i) + left)(0,0)
					+ omega(xy.row(i) + right)(0,0)
					+ omega(xy.row(i) + top)(0,0)
					+ omega(xy.row(i) + bot)(0,0)
					- 4*omega(xy.row(i))(0,0) );
			xy.row(i) += dt * u_h_sigma(xy.row(i), omega.points(), omega.coeffs(), sigma);
		}

		// Plotting:
		if ( count++ % one_step == 0 )
		{
			plotf = omega(plotX);
			std::ofstream omega_str( "omega_" + std::to_string(t) + ".txt" );
			for ( size_t i = 0; i <= res; ++i )
			{
				for ( size_t j = 0; j <= res; ++j )
				{
					double x = plotX(j + (res + 1)*i,0);
					double y = plotX(j + (res + 1)*i,1);
					double f = plotf(j + (res+1)*i);
					omega_str << x << " " << y << " " << f << std::endl;
				}
				omega_str << "\n";
			}
		}


		t += dt;

		double elapsed = clock.elapsed();
		std::cout << "Time for needed for time-step: " << elapsed << ".\n";
		std::cout << "---------------------------------------" << elapsed << ".\n";

		if ( t + dt > T ) dt = T - t;
	}
}
