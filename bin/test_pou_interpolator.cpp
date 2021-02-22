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
	constexpr size_t N { 10000 };
	constexpr double tikhonov_mu { 1e-9 };
	constexpr size_t min_per_box = 500;
	constexpr size_t max_per_box = 500;
	constexpr double enlarge = 1.5;
	constexpr double twopi { 2*3.1415926535 };

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

	vlasovius::misc::stopwatch clock;
	interpolator_t sfx { kernel_t {}, X, f, tikhonov_mu, min_per_box, max_per_box, enlarge};
	double elapsed { clock.elapsed() };
	std::cout << "Time for computing RBF-Approximation: " << elapsed << ".\n";

	return 0;
}
