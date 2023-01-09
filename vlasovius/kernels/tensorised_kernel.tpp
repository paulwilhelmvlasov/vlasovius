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
	if ( X.n_cols != inv_sigma.n_cols )
	{
		throw std::logic_error { "vlasovius::kernels::tensorised_kernel::operator(): "
								 "Dimension mismatch. X.n_cols != inv_sigma.n_cols" };
	}
	if ( Y.n_cols != inv_sigma.n_cols )
	{
		std::cout << Y.n_cols << std::endl;
		std::cout << inv_sigma.n_cols << std::endl;
		throw std::logic_error { "vlasovius::kernels::tensorised_kernel::operator(): "
									 "Dimension mismatch. Y.n_cols != inv_sigma.n_cols" };
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


	if ( threads > 1 )
	{
		#pragma omp parallel num_threads(threads)
		for ( size_t j = 0; j < m; ++j )
			eval_column( dim, n, j, K, ldK, X, ldX, Y, ldY );
	}
	else
	{
		for ( size_t j = 0; j < m; ++j )
			eval_column( dim, n, j, K, ldK, X, ldX, Y, ldY );
	}
}

template <typename rbf_function>
void tensorised_kernel<rbf_function>::eval_column( size_t dim, size_t n, size_t j,
		                                                 double* K, size_t ldK,
		                                           const double* X, size_t ldX,
									               const double* Y, size_t ldY ) const
{
	using simd_t = vlasovius::misc::simd<double>;

	size_t i { 0 };
	if constexpr ( simd_t::size() > 1 )
	{
		simd_t val, x, y;
		for ( ; i + simd_t::size() < n; i += simd_t::size() )
		{
			val.fill(1.0);
			for ( size_t d = 0; d < dim; ++d )
			{
				x.load( X + i + d*ldX );
				y.fill( Y + j + d*ldY );
				val = val * F( abs(x-y)*inv_sigma(d) );
			}
			val.store( K + i + j*ldK );
		}
	}

	for ( ; i < n; ++i )
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
template <typename simd_t>
void tensorised_kernel<rbf_function>::micro_kernel( size_t dim, simd_t mat[ simd_t::size() ],
			                                        const double *X, size_t ldX,
	                                                const double *Y, size_t ldY ) const
{
	simd_t x, y, tmp[ simd_t::size() ];

	// d = 1.
	x.load( X );
	for ( size_t k = 0; k < simd_t::size(); ++k )
	{
		y.fill( Y + k );
		mat[k] = abs(x-y)*inv_sigma(0);
	}
	F.eval(mat);

	for ( size_t d = 1; d < dim; ++d )
	{
		x.load( X + d*ldX );
		for ( size_t k = 0; k < simd_t::size(); ++k )
		{
			y.fill( Y + k + d*ldY );
			tmp[k] = abs(x-y)*inv_sigma(d);
		}
		F.eval(tmp);
		for ( size_t k = 0; k < simd_t::size(); ++k )
		{
			mat[k] = mat[k] * tmp[k];
		}
	}
}

//template <typename rbf_function>
//void tensorised_kernel<rbf_function>::mul( size_t dim, size_t n, size_t m, size_t nrhs,
//		                                         double *R, size_t ldR,
//                                           const double *X, size_t ldX,
//                                           const double *Y, size_t ldY,
//		                                   const double *C, size_t ldC, size_t threads ) const
//{
//	#ifndef NDEBUG
//	if ( dim != inv_sigma.n_cols )
//	{
//		throw std::logic_error { "vlasovius::kernels::tensorised_kernel::mul(): "
//								 "Dimension mismatch." };
//	}
//	#endif
//
//	#pragma omp parallel if(threads>1),num_threads(threads)
//	for ( size_t i = 0; i < n; ++i )
//	{
//		for ( size_t k = 0; k < nrhs; ++k )
//			R[ i + k*ldR ] = 0;
//
//		for ( size_t j = 0; j < m; ++j )
//		{
//			double kernelval { 1 };
//			for ( size_t d = 0; d < dim; ++d )
//			{
//				double r { std::abs( X[ i + d*ldX ] - Y[ j + d*ldY ] ) };
//				kernelval *= F( r*inv_sigma(d) );
//			}
//
//			for ( size_t k = 0; k < nrhs; ++k )
//				R[ i + k*ldR ] += kernelval*C[ j + k*ldC ];
//		}
//	}
//}

template <typename rbf_function>
void tensorised_kernel<rbf_function>::mul( size_t dim, size_t n, size_t m, size_t nrhs,
		                            double *__restrict__  R, size_t ldR,
		                            const double         *X, size_t ldX,
									const double         *Y, size_t ldY,
									const double         *C, size_t ldC, size_t threads ) const
{
	// Block size.
	using simd_t = vlasovius::misc::simd<double>;
	constexpr size_t bs = simd_t::size();

	#pragma omp parallel if(threads>1), num_threads(threads)
	{

	arma::mat Xbuf( bs, dim  );
	arma::mat Ybuf( bs, dim  );
	arma::mat Rbuf( bs, nrhs );
	simd_t    Kbuf[ bs ];

	const double  *xx,  *yy;
	      size_t ldxx, ldyy;

	const size_t num_iblocks { (n/bs) + ((n%bs)?1:0) };
	const size_t num_jblocks { (m/bs) + ((m%bs)?1:0) };

	#pragma omp for
	for ( size_t i_block = 0; i_block < num_iblocks; ++i_block )
	{
		const size_t i_curr { i_block*bs };
		const size_t i_max  { std::min(n,i_curr+bs) };

		// Pack X
		if ( i_curr + bs < n )
		{
			  xx = X + i_curr;
			ldxx = ldX;
		}
		else
		{
			for ( size_t d = 0; d < dim; ++d )
			for ( size_t i = 0; i + i_curr < i_max; ++i )
				Xbuf(i,d) = X[ i + i_curr + d*ldX ];

			  xx = Xbuf.memptr();
			ldxx = bs;
		}

		Rbuf.zeros();
		for ( size_t j_block = 0; j_block < num_jblocks; ++j_block )
		{
			size_t j_curr { j_block*bs };
			size_t j_max  { std::min(m,j_curr+bs) };

			// Pack Y
			if ( j_curr + bs < m )
			{
				  yy = Y + j_curr;
				ldyy = ldY;
			}
			else
			{
				for ( size_t d = 0; d < dim; ++d )
					for ( size_t j = 0; j + j_curr < j_max; ++j )
						Ybuf(j,d) = Y[ j + j_curr + d*ldY ];

				  yy = Ybuf.memptr();
				ldyy = bs;
			}

			// Actual computation of the kernel sub-matrix Kbuf.
			micro_kernel( dim, Kbuf, xx, ldxx, yy, ldyy );

			// Application of Kbuf.
			for ( size_t kk = 0; kk < nrhs; ++kk )
			{
				simd_t k { Rbuf.memptr() + kk*bs };
				for ( size_t jj = 0; jj + j_curr < j_max; ++jj )
				{
					simd_t c; c.fill( C[jj+j_curr+kk*ldC] );
					k = fmadd(c,Kbuf[jj],k);
				}
				k.store( Rbuf.memptr() + kk*bs );
			}
		}

		for ( size_t kk = 0; kk < nrhs;  ++kk )
		for ( size_t ii = 0; i_curr + ii < i_max; ++ii )
			R[ i_curr + ii + kk*ldR ] = Rbuf(ii,kk);
	}

	}
}

}

}
