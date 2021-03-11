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

#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/interpolators/pou_interpolator.h>

int main()
{
	constexpr size_t dim { 2 }, k { 4 };
	constexpr size_t N { 1'000'000 };
	constexpr double tikhonov_mu { 1e-12 };
	constexpr size_t min_per_box = 100;
	constexpr double enlarge = 1.5;
	constexpr double twopi { 2*3.1415926535 };

	std::cout << "N = " << N << std::endl;
	std::cout << "min per Box = " << min_per_box << std::endl;

	using wendland_t     = vlasovius::kernels::wendland<dim,k>;
	using kernel_t       = vlasovius::kernels::rbf_kernel<wendland_t>;
	using interpolator_t = vlasovius::interpolators::pou_interpolator<kernel_t>;

	arma::mat X( N, 2, arma::fill::randu );
	arma::vec f( N );
	for ( size_t i = 0; i < N; ++i )
	{
		double x = twopi*X(i,0);
		double y = twopi*X(i,1);
		f(i) = std::sin(x)*std::sin(y);
	}

	arma::rowvec bounding_box { 0, 0, 1, 1 };

	kernel_t K { wendland_t {}, 0.5 };
	vlasovius::misc::stopwatch clock;
	interpolator_t sfx { K, X, f, bounding_box, enlarge, min_per_box, tikhonov_mu };
	double elapsed { clock.elapsed() };
	std::cout << "Time for computing RBF-Approximation: " << elapsed << ".\n";
	clock.reset();
	double error = norm(f-sfx(X),"inf");
	elapsed = clock.elapsed();
	std::cout << "Maximal interpolation error: " << error << ".\n";
	std::cout << "Time for evaluating interpolation error: " << elapsed << ".\n";

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

	std::ofstream str( "test_pou_interpolator.txt" );
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


	return 0;
}
