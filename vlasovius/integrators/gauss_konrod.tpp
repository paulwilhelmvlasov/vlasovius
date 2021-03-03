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



template<typename function>
arma::vec vlasovius::integrators::gauss_konrod_1d
(const function& f, double a, double b, double eps = 1e-5)
{
	// Note: f:[a,b] -> \R

	#ifndef NDEBUG
	if ( a == b)
	{
		throw std::runtime_error { "vlasovius::integrators::gauss_konrod_1d: "
		   "a == b" };
	} else if( a > b)
	{
		return gauss_konrod_1d(f, b, a, eps);
	}
	#endif

	// Change of intervall: [a,b] -> [-1,1]
	arma::vec f_evalf = 0.5 * (b - a) * f(0.5 * (b - a) * x_gauss_konrod_7_15 + 0.5 * (a + b));

	// Compute integral with gaussian and konrod:
	double int_gauss  = arma::dot( f_evalf.subvec(0, 6), vlasovius::integrators::w_gauss_7 );
	double int_konrod = arma::dot( f_evalf, vlasovius::integrators::w_konrod_15 );

	if(std::abs(int_gauss - int_konrod) <= 1e-5)
	{
		return int_konrod;
	}else
	{
		double center = a + 0.5 *  (b - a);
		return gauss_konrod_1d(f, a, center, eps) + gauss_konrod_1d(f. center, b, eps);
	}
}
