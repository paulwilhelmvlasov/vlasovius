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
#include <cmath>
#include <fstream>
#include <iostream>
#include <armadillo>


#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/misc/xv_kernel.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/interpolators/direct_interpolator.h>

double maxwell(double x, double v, double alpha = 0.01, double k = 0.5)
{
	return 0.39894228040143267793994 * ( 1 + alpha * std::cos(k * x) )
			* std::exp( -0.5 * v*v );
}

int main()
{
	constexpr size_t dim { 2 }, k { 4 };
	constexpr size_t N { 2'000 };

	constexpr double tikhonov_mu { 1e-12 };
	constexpr double twopi { 2*3.1415926535 };

	double L = 4 * M_PI;
	double v_max = 10.0;

	size_t threads { (size_t) omp_get_max_threads() };

	using wendland_t     = vlasovius::kernels::wendland<dim,k>;
	//using kernel_t       = vlasovius::kernels::rbf_kernel<wendland_t>;
	using kernel_t = vlasovius::xv_kernel<4, 4>;
	using interpolator_t = vlasovius::interpolators::direct_interpolator<kernel_t>;

	arma::mat X( N, 2, arma::fill::randu );
	X.col(0) *= L;
	X.col(1) = 2 * v_max * X.col(1)
			- v_max * arma::vec(N, arma::fill::ones);
	arma::vec f( N );
	for ( size_t i = 0; i < N; ++i )
	{
		f(i) = maxwell(X(i,0), X(i,1), 0.01, 0.5);
	}

	//kernel_t K { wendland_t {}, 4.0 };
	kernel_t K {4.0, 4.0, L};

	vlasovius::misc::stopwatch clock;
	interpolator_t sfx { K, X, f, tikhonov_mu, threads };
	double elapsed { clock.elapsed() };
	std::cout << "Time for computing RBF-Approximation: " << elapsed << ".\n";
	std::cout << "Maximal interpolation error: " << norm(f-sfx(X),"inf") << ".\n";

	arma::mat plotX( 101*101, 2 );
	arma::vec plotf_true( 101*101 );
	for ( size_t i = 0; i <= 100; ++i )
	for ( size_t j = 0; j <= 100; ++j )
	{
		double x = plotX(j + 101*i,0) = L * i/100.;
		double y = plotX(j + 101*i,1) = 2 * v_max * j/100. - v_max;
		plotf_true(j + 101*i) = maxwell(x, y, 0.01, 0.5);
	}

	clock.reset();
	arma::vec plotf = sfx(plotX,threads);
	elapsed = clock.elapsed();
	std::cout << "Time for evaluating RBF-approximation at plotting points: " << elapsed << ".\n";
	std::cout << "Maximum encountered error at plotting points: " << norm(plotf-plotf_true,"inf") << ".\n";

	std::ofstream str( "test_direct_interpolator.txt" );
	for ( size_t i = 0; i <= 100; ++i )
	{
		for ( size_t j = 0; j <= 100; ++j )
		{
			double x = plotX(j + 101*i,0);
			double y = plotX(j + 101*i,1);
			double err  = plotf(j+101*i) - plotf_true(j+101*i);
			str << x << " " << y << " " << err << std::endl;
		}
		str << "\n";
	}

	std::ofstream str2( "test_boundary.txt" );
	for ( size_t i = 0; i <= 100; ++i )
	{
		double v = 2 * v_max * i / 100. - v_max;
		double err  = sfx(arma::rowvec{0, v})(0) - sfx(arma::rowvec{L, v})(0);
		str2 << v << " " << err << std::endl;
	}

	std::ofstream str3( "test_points.txt" );
	for ( size_t i = 0; i < N; ++i )
	{
		str3 << X(i, 0) << " " << X(i, 1) << std::endl;
	}

	return 0;
}

