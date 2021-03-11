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

#include <vlasovius/integrators/gauss_konrod.h>


arma::vec test_1(const arma::mat& xv)
{
	arma::uword N = xv.n_rows;
	arma::vec r(N);

	for(arma::uword i = 0; i < N; i++)
	{
		r(i) = std::sin(xv(i, 1));
	}

	return r;
}

arma::vec test_2(const arma::mat& xv)
{
	arma::uword N = xv.n_rows;
	arma::vec r(N);

	for(arma::uword i = 0; i < N; i++)
	{
		r(i) = xv(i,0) * std::sin(xv(i, 1));
	}

	return r;
}


int main()
{

	arma::mat x(100, 1, arma::fill::randu);

	arma::vec r = vlasovius::integrators::gauss_konrod_1d(test_1, x, 0, 3.14, 1e-6, 6);
	std::cout << r;

	arma::vec v = vlasovius::integrators::gauss_konrod_1d(test_2, x, 0, 3.14, 1e-6, 6);
	std::cout << v - 2 * x;

	return 0;
}
