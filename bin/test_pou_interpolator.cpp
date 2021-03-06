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

#include <vlasovius/interpolators/pou_interpolator.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/misc/stopwatch.h>

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
	vlasovius::misc::stopwatch clock;
	kernel_t K { wendland_t {}, 0.5 };
	interpolator_t sfx { K, X, f, bounding_box, enlarge, min_per_box, tikhonov_mu };
	double elapsed { clock.elapsed() };
	std::cout << "Time for computing RBF-Approximation: " << elapsed << ".\n";
	clock.reset();
	double error = norm(f-sfx(X),"inf");
	elapsed = clock.elapsed();
	std::cout << "Maximal interpolation error: " << error << ".\n";
	std::cout << "Time for evaluating interpolation error: " << elapsed << ".\n";

	arma::mat plotX( 1001*1001, 2 );
	arma::vec plotf_true( 1001*1001 );
	for ( size_t i = 0; i <= 1000; ++i )
		for ( size_t j = 0; j <= 1000; ++j )
		{
			double x = plotX(j + 1001*i,0) = i/1000.;
			double y = plotX(j + 1001*i,1) = j/1000.;
			plotf_true(j + 1001*i) = std::sin(twopi*x)*std::sin(twopi*y);
		}

	clock.reset();
	arma::vec plotf = sfx(plotX);
	elapsed = clock.elapsed();
	std::cout << "Time for evaluating RBF-approximation at plotting points: " << elapsed << ".\n";
	std::cout << "Maximum encountered error at plotting points: " << norm(plotf-plotf_true,"inf") << ".\n";

	std::ofstream str( "test_pou_interpolator.txt" );
	for ( size_t i = 0; i <= 1000; ++i )
	{
		for ( size_t j = 0; j <= 1000; ++j )
		{
			double x = plotX(j + 1001*i,0);
			double y = plotX(j + 1001*i,1);
			double err  = plotf(j+1001*i)-plotf_true(j+1001*i);
			str << x << " " << y << " " << err << std::endl;
		}
		str << "\n";
	}


	return 0;
}
