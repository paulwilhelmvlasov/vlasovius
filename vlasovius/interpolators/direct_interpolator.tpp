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

	size_t n = X.n_rows;

	using arma::span;
	arma::mat V(n,n);
	constexpr size_t block_size = 64;
	size_t num_blocks = n/block_size;

	misc::stopwatch clock;

	arma::mat buf(block_size,block_size);

	#pragma omp for schedule(dynamic)
	for ( size_t i_block = 0; i_block < num_blocks; ++i_block )
	{
		for ( size_t j_block = 0; j_block <= i_block; ++j_block )
		{
			size_t i_begin = i_block*block_size, i_end = i_begin + block_size - 1;
			size_t j_begin = j_block*block_size, j_end = j_begin + block_size - 1;

			V( span(i_begin,i_end), span(j_begin,j_end) ) = K( X(span(i_begin,i_end), span(0,X.n_cols-1) ),
															   X(span(j_begin,j_end), span(0,X.n_cols-1) ));
		}
	}


	size_t i_begin = num_blocks*block_size, i_end = n-1;
	V( span(i_begin,i_end), span(0,n-1) ) = K( X(span(i_begin,i_end), span(0,X.n_cols-1) ),
			                                   X(span(0,n-1), span(0,X.n_cols-1) ));

	V.diag() += tikhonov_mu * arma::vec( X.n_rows, arma::fill::ones );
	double elapsed = clock.elapsed();
	std::cout << "Time for matrix assembly: " << elapsed << ".\n";


	clock.reset();
	// Compute Cholesky decomposition V = L * L^T.
	lapack_int info;
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

	return K(Y,X)*coeff;
}

}

}
