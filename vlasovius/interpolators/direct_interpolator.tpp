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

#include <lapacke.h>

namespace vlasovius
{

namespace interpolators
{

template <typename kernel>
direct_interpolator<kernel>::direct_interpolator
( kernel p_K, arma::mat p_X, arma::vec b, double tikhonov_mu ):
K { p_K }, X { std::move(p_X) }, coeff { std::move(b) }
{
	if ( X.empty() )
	{
		throw std::runtime_error { "vlasovius::direct_interpolator::direct_interpolator(): "
				                   "No interpolation-points passed." };
	}

	if ( X.n_rows != coeff.size() )
	{
		throw std::runtime_error { "vlasovius::direct_interpolator::direct_interpolator(): "
			                       "Number of points does not match value data vector length." };
	}

	arma::mat V = K(X,X);
	V.diag() += tikhonov_mu * arma::vec( X.n_rows, arma::fill::ones );

	// Directly calling LAPACK's Cholesky routines is faster than:
	// coeff = arma::solve( K(X,X), b );

	// Compute Cholesky decomposition V = L * L^T.
	size_t n = V.n_rows; lapack_int info;
	info = LAPACKE_dpotrf( LAPACK_COL_MAJOR, 'L', n, V.memptr(), n );
	if ( info )
	{
		throw std::runtime_error { "vlasovius::direct_interpolator::direct_interpolator(): "
			                       "Error while computing Cholesky factorisation of Vandermonde matrix." };
	}

	// Solve Ly = b; We already initialised coeff with b.
	info = LAPACKE_dtrtrs( LAPACK_COL_MAJOR, 'L', 'N', 'N', n, 1, V.memptr(), n, coeff.memptr(), n );
	if ( info )
	{
		throw std::runtime_error { "vlasovius::direct_interpolator::direct_interpolator(): "
			                       "Error while solving triangular system Ly = b." };
	}

	// Solve L^T x = y;
	info = LAPACKE_dtrtrs( LAPACK_COL_MAJOR, 'L', 'T', 'N', n, 1, V.memptr(), n, coeff.memptr(), n );
	if ( info )
	{
		throw std::runtime_error { "vlasovius::direct_interpolator::direct_interpolator(): "
			                       "Error while solving triangular system L^Tx = y." };
	}
}

template <typename kernel>
arma::vec direct_interpolator<kernel>::operator()( const arma::mat &Y ) const
{
	if ( Y.empty() )
	{
		throw std::runtime_error { "vlasovius::direct_interpolator::operator(): "
				                   "No evaluation points passed." };
	}

	if ( X.n_cols != Y.n_cols )
	{
		throw std::runtime_error { "vlasovius::direct_interpolator::operator(): "
		                           "Evaluation and Interpolation points have differing dimensions." };
	}

	return K(Y,X)*coeff;
}

}

}
