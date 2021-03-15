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


template<typename function_1d_xv>
	arma::vec vlasovius::integrators::num_rho_1d(const function_1d_xv& f, const arma::mat& x,
			double vmax, double eps, size_t threads)
{
	arma::uword N = x.n_rows;
	return arma::vec(N, arma::fill::ones) - gauss_konrod_1d(f, x, -vmax, vmax, eps, threads);
}


template<typename function_1d_xv>
	arma::vec vlasovius::integrators::gauss_konrod_1d(const function_1d_xv& f, const arma::mat& x,
			double a, double b, double eps, size_t threads)
{
	// Gets a function: f: [0,L] x \R -> \R, a N-list of points x to integrate f(x,*)
	// and returns a N-vec of rho-values.
	arma::uword N = x.n_rows;
	arma::uword m_konrod = 15;
	arma::uword m_gauss  = 7;
	arma::mat z_eval_konrod(N * m_konrod, 2);

	#pragma omp parallel for num_threads(threads)
	for(arma::uword i = 0; i < N; i++)
	{
		// Fill the evaluation matrix with the x-points and the v-quadrature-points.
		// Also change of intervall: [a, b] -> [-1, 1].
		z_eval_konrod.col(0).subvec(i * m_konrod, (i + 1) * m_konrod - 1).fill(x(i, 0));
		z_eval_konrod.col(1).subvec(i * m_konrod, (i + 1) * m_konrod - 1)
				= 0.5 * (b - a) * x_gauss_konrod_7_15
				+ 0.5 * (a + b) * arma::vec(m_konrod, arma::fill::ones);
	}

	// Now evaluate f at all Konrod-Points:
	// (Note that the Gauss points are a subset.)
	arma::vec f_eval = 0.5 * (b - a) * f(z_eval_konrod);

	// ...and distribute the points to the evaluation matrices:
	arma::mat f_eval_konrod(f_eval.memptr(), m_konrod, N);
	f_eval_konrod = f_eval_konrod.t();
	arma::mat f_eval_gauss = f_eval_konrod.submat(0, 0, N - 1, m_gauss - 1);

	// Compute integrals:
	arma::vec int_konrod = f_eval_konrod * w_konrod_15;
	arma::vec int_gauss  = f_eval_gauss * w_gauss_7;

	// If the difference of konrod and gauss is smaller eps, then return konrod result.
	// Else split the integral into two sub-integrals:
	if(arma::norm(int_konrod - int_gauss, "inf") <= eps)
	{
		return int_konrod;
	}else
	{
		double center = a + 0.5 * (b - a);
		return gauss_konrod_1d(f, x, a, center, 0.5 * eps, threads)
			 + gauss_konrod_1d(f, x, center, b, 0.5 * eps, threads);
	}
}

template<typename function_1d>
double vlasovius::integrators::gauss_konrod_1d
(const function_1d& f, double a, double b, double eps)
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

	if(std::abs(int_gauss - int_konrod) <= eps)
	{
		return int_konrod;
	}else
	{
		double center = a + 0.5 *  (b - a);
		return gauss_konrod_1d(f, a, center, 0.5 * eps) + gauss_konrod_1d(f, center, b, 0.5 * eps);
	}
}
