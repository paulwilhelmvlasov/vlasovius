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

#ifndef VLASOVIUS_INTEGRATORS_GAUSS_KONROD_H
#define VLASOVIUS_INTEGRATORS_GAUSS_KONROD_H

#include <armadillo>

namespace vlasovius
{

namespace integrators
{

	template<typename function_1d_xv>
	arma::vec num_rho_1d(const function_1d_xv& f, const arma::mat& x,
			double vmax = 10.0, double eps = 1e-5, size_t threads = 1 );


	template<typename function_1d_xv>
	arma::vec gauss_konrod_1d(const function_1d_xv& f, const arma::mat& x,
			double a = -10.0, double b = 10.0, double eps = 1e-5, size_t threads = 1 );

	template<typename function_1d>
	arma::vec gauss_konrod_1d(const function& f, double a,
			double b, double eps = 1e-5);

	// Nodes for konrod-15-quadrature. Sorted such that the first
	// 7 nodes coincide with the gauss-7-quadrature:
	arma::vec x_gauss_konrod_7_15
	{
		-0.949107912342759,
		-0.741531185599394,
		-0.405845151377397,
		0.0,
		0.405845151377397,
		0.741531185599394,
		0.949107912342759,

		-0.991455371120813,
		-0.864864423359769,
		-0.586087235467691,
		-0.207784955007898,
		0.207784955007898,
		0.586087235467691,
		0.864864423359769,
		0.991455371120813
	};

	arma::vec w_gauss_7
	{
		0.129484966168870,
		0.279705391489277,
		0.381830050505119,
		0.417959183673469,
		0.381830050505119,
		0.279705391489277,
		0.129484966168870,
	};

	arma::vec w_konrod_15
	{
		0.063092092629979,
		0.140653259715525,
		0.190350578064785,
		0.209482141084728,
		0.190350578064785,
		0.140653259715525,
		0.063092092629979,

		0.022935322010529,
		0.104790010322250,
		0.169004726639267,
		0.204432940075298,
		0.204432940075298,
		0.169004726639267,
		0.104790010322250,
		0.022935322010529
	};

}

}

#include <vlasovius/integrators/gauss_konrod.tpp>
#endif
