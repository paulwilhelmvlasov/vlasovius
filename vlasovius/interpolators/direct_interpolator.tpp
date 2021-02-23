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

#include <vlasovius/misc/stopwatch.h>

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

	const size_t n = X.n_rows, dim = X.n_cols, ldX  = n;
	const size_t ar_rows = (n & 1) ?  (n      ) : (n+1);
	const size_t ar_cols = (n & 1) ?  (n/2 + 1) : (n/2);

	// AR is stored in Rectangular Full Packed Format (RFPF)
	// See reference [gstv2010]
	arma::mat AR( ar_rows, ar_cols );

	misc::stopwatch clock;
	if ( n & 1 )
	{
		// First column
		K.eval( dim, n, 1, &AR(0,0), ar_rows, &X(0,0), ldX, &X(0,0), ldX );

		// Remaining columns.
		#pragma omp parallel for
		for ( size_t j = 1; j < ar_cols; ++j )
		{
			// Upper part of column
			K.eval( dim, j, 1, &AR(0,j), ar_rows, &X(n/2+1,0), ldX, &X(n/2+j,0), ldX );

			// Lower part of column
			K.eval( dim, n-j, 1, &AR(j,j), ar_rows, &X(j,0), ldX, &X(j,0), ldX );
			AR(j,j) += tikhonov_mu;
		}
	}
	else
	{
		#pragma omp parallel for
		for ( size_t j = 0; j < ar_cols; ++j )
		{
			// Upper part of column j
			K.eval( dim, j+1, 1, &AR(0,j), ar_rows, &X(n/2,0), ldX, &X(n/2+j,0), ldX );

			// Lower part of column j
			K.eval( dim, n-j, 1, &AR(j+1,j), ar_rows, &X(j,0), ldX, &X(j,0), ldX );
			AR(j+1,j) += tikhonov_mu;
		}
	}
	double elapsed = clock.elapsed();
	std::cout << "Time for matrix assembly: " << elapsed << ".\n";

	clock.reset();
	lapack_int info = LAPACKE_dpftrf( LAPACK_COL_MAJOR, 'N', 'L', n, AR.memptr() );
	if ( info )
	{
		throw std::runtime_error { "vlasovius::direct_interpolator::direct_interpolator(): "
		                           "Error while computing Cholesky factorisation of Vandermonde matrix." };
	}

	info = LAPACKE_dpftrs( LAPACK_COL_MAJOR, 'N', 'L', n, 1, AR.memptr(), coeff.memptr(), n );
	if ( info )
	{
		throw std::runtime_error { "vlasovius::direct_interpolator::direct_interpolator(): "
				                   "Error while solving triangular systems L^Tx = y, Ly = b." };
	}

	elapsed = clock.elapsed();
	std::cout << "Time for solving: " << elapsed << ".\n";
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

	size_t dim = X.n_cols, n = Y.n_rows, m = X.n_rows;
	arma::vec result( Y.n_rows, arma::fill::zeros );

	#pragma omp parallel
	{
		arma::vec tmp( Y.n_rows ), thread_sum( Y.n_rows, arma::fill::zeros );

		#pragma omp for
		for ( size_t i = 0; i < X.n_rows; ++i )
		{
			K.eval( dim, n, 1, tmp.memptr(), n, Y.memptr(), n, &X(i,0), m );
			thread_sum += tmp*coeff(i);
		}

		#pragma omp critical
		result += thread_sum;
	}
	return result;
}

}

}
