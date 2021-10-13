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

double two_stream_f0(double x, double v, double alpha = 0.01, double k = 0.5)
{
    return 0.39894228040143267793994 * v * v * std::exp(-0.5 * v * v )
	        * (1.0 + alpha * std::cos(k * x));
}

double bump_on_tail_f0(double x, double v, double alpha = 0.01, double k = 0.5, double nb = 0.1, double vb = 4.5)
{
    return 0.39894228040143267793994 * ((1.0 - nb) * std::exp(-0.5 * v * v)
        + nb * std::exp(-0.5 * (v - vb) * (v - vb)))
        * (1.0 + alpha * std::cos(k * x));
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

	size_t num_threads = omp_get_max_threads();


	return 0;
}
