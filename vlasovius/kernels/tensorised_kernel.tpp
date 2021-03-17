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
#include <vlasovius/kernels/tensorised_kernel.h>

namespace vlasovius
{

namespace kernels
{

template <typename rbf_function>
tensorised_kernel<rbf_function>::tensorised_kernel( rbf_function p_F, arma::rowvec sigma ):
F { std::move(p_F) }, inv_sigma { std::move(sigma) }
{
	for ( size_t d = 0; d < inv_sigma.size(); ++d )
		inv_sigma(d) = 1./inv_sigma(d);
}

template <typename rbf_function>
arma::mat tensorised_kernel<rbf_function>::operator()( const arma::mat &X, const arma::mat &Y, size_t threads ) const
{
	#ifndef NDEBUG
	if ( X.n_cols != inv_sigma.n_cols || Y.n_cols != inv_sigma.n_cols )
	{
		throw std::logic_error { "vlasovius::kernels::tensorised_kernel::operator(): "
								 "Dimension mismatch." };
	}
	#endif

	arma::mat result( X.n_rows, Y.n_rows );
	eval( inv_sigma.n_cols, X.n_rows, Y.n_rows,
		  result.memptr(), X.n_rows,
		       X.memptr(), X.n_rows,
		       Y.memptr(), Y.n_rows, threads );
	return result;
}

template <typename rbf_function>
void tensorised_kernel<rbf_function>::eval( size_t dim, size_t n, size_t m,
		                                          double *K, size_t ldK,
                                            const double *X, size_t ldX,
                                            const double *Y, size_t ldY, size_t threads ) const
{
	#ifndef NDEBUG
	if ( dim != inv_sigma.n_cols )
	{
		throw std::logic_error { "vlasovius::kernels::tensorised_kernel::eval(): "
			                     "Dimension mismatch." };
	}
	#endif

	#pragma omp parallel if(threads>1),num_threads(threads)
	for ( size_t j = 0; j < m; ++j )
	for ( size_t i = 0; i < n; ++i )
	{
		double val { 1 };
		for ( size_t d = 0; d < dim; ++d )
		{
			double r { std::abs( X[ i + d*ldX ] - Y[ j + d*ldY ] ) };
			val *= F( r*inv_sigma(d) );
		}
		K[ i + j*ldK ] = val;
	}
}

template <typename rbf_function>
void tensorised_kernel<rbf_function>::mul( size_t dim, size_t n, size_t m, size_t nrhs,
		                                         double *R, size_t ldR,
                                           const double *X, size_t ldX,
                                           const double *Y, size_t ldY,
		                                   const double *C, size_t ldC, size_t threads ) const
{
	#ifndef NDEBUG
	if ( dim != inv_sigma.n_cols )
	{
		throw std::logic_error { "vlasovius::kernels::tensorised_kernel::mul(): "
								 "Dimension mismatch." };
	}
	#endif

	#pragma omp parallel if(threads>1),num_threads(threads)
	for ( size_t i = 0; i < n; ++i )
	{
		for ( size_t k = 0; k < nrhs; ++k )
			R[ i + k*ldR ] = 0;

		for ( size_t j = 0; j < m; ++j )
		{
			double kernelval { 1 };
			for ( size_t d = 0; d < dim; ++d )
			{
				double r { std::abs( X[ i + d*ldX ] - Y[ j + d*ldY ] ) };
				kernelval *= F( r*inv_sigma(d) );
			}

			for ( size_t k = 0; k < nrhs; ++k )
				R[ i + k*ldR ] += kernelval*C[ j + k*ldC ];
		}
	}
}

}

}
