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
#include <cmath>
#include <fstream>
#include <iostream>
#include <armadillo>

#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/interpolators/direct_interpolator.h>

int main()
{
	constexpr size_t dim { 2 }, k { 4 };
	constexpr size_t N { 10000 };
	constexpr double tikhonov_mu { 1e-9 };
	constexpr double twopi { 2*3.1415926535 };

	using wendland_t     = vlasovius::kernels::wendland<dim,k>;
	using kernel_t       = vlasovius::kernels::rbf_kernel<wendland_t>;
	using interpolator_t = vlasovius::interpolators::direct_interpolator<kernel_t>;

	arma::mat X( N, 2, arma::fill::randu );
	arma::vec f( N );
	for ( size_t i = 0; i < N; ++i )
	{
		double x = twopi*X(i,0);
		double y = twopi*X(i,1);
		f(i) = std::sin(x)*std::sin(y);
	}

	vlasovius::misc::stopwatch clock;
	interpolator_t sfx { kernel_t {}, X, f, tikhonov_mu };
	double elapsed { clock.elapsed() };
	std::cout << "Time for computing RBF-Approximation: " << elapsed << ".\n";
	std::cout << "Maximal interpolation error: " << norm(f-sfx(X),"inf") << ".\n";

	arma::mat plotX( 101*101, 2 );
	arma::vec plotf_true( 101*101 );
	for ( size_t i = 0; i <= 100; ++i )
	for ( size_t j = 0; j <= 100; ++j )
	{
		double x = plotX(j + 101*i,0) = i/100.;
		double y = plotX(j + 101*i,1) = j/100.;
		plotf_true(j + 101*i) = std::sin(twopi*x)*std::sin(twopi*y);
	}

	clock.reset();
	arma::vec plotf = sfx(plotX);
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
			double err  = plotf(j+101*i)-plotf_true(j+101*i);
			str << x << " " << y << " " << err << std::endl;
		}
		str << "\n";
	}

}
