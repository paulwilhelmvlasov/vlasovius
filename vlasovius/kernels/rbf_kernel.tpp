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
#include <vlasovius/kernels/rbf_kernel.h>

namespace vlasovius
{

namespace kernels
{

template <typename rbf_function>
rbf_kernel<rbf_function>::rbf_kernel( rbf_function p_F, double sigma ):
F { p_F }, inv_sigma { 1./sigma }
{}

template <typename rbf_function>
arma::mat rbf_kernel<rbf_function>::operator()( const arma::mat &X,
		                                        const arma::mat &Y ) const
{
	if ( X.n_cols != Y.n_cols )
	{
		throw std::logic_error { "vlasovius::kernels::rbf_kernel::operator(): "
			                     "X and Y contain points of differing dimension." };
	}
	const size_t N { X.n_rows }, M { Y.n_rows }, dim { Y.n_cols };
	arma::mat result( N, M );
	eval( dim, N, M, result.memptr(), N, X.memptr(), N, Y.memptr(), M );
	return result;

}

template <typename rbf_function>
void rbf_kernel<rbf_function>::eval( size_t dim, size_t n, size_t m,
		                                   double *K, size_t ldK,
		                             const double *X, size_t ldX,
									 const double *Y, size_t ldY, size_t threads ) const
{
	if ( threads > 1 )
	{
		#pragma omp parallel for num_threads(threads)
		for ( size_t j = 0; j < m; ++j )
			eval_column( dim, n, j, K, ldK, X, ldX, Y, ldY );
	}
	else
	{
		for ( size_t j = 0; j < m; ++j )
			eval_column( dim, n, j, K, ldK, X, ldX, Y, ldY );
	}
}


/*
 * \brief Computes R = K(X,Y)*C.
 * \param dim     Dimensionality of the points in X and Y.
 * \param n       Number of points in X,      i.e., number of rows    of K(X,Y)
 * \param m       Number of points in Y,      i.e., number of columns of K(X,Y)
 * \param nhrs    Number of right hand sides, i.e., number of columns of C.
 * \param R       The result matrix R.
 * \param ldR     Column stride of R.
 * \param X       n*dim dimensional array with n points.
 * \param ldX     Column stride of X.
 * \param Y       m*dim dimensional array with m points.
 * \param ldY     Column stride of Y.
 * \param C       m*nrhs dimensional coefficient matrix C.
 * \param ldC     Column stride of C.
 * \param threads Number of threads to use for computing the result in parallel. (default: 1)
 *
 * This method is cache and memory friendly: it does not explicitly assemble and
 * store the entire matrix K(X,Y). Instead it proceeds block-wise: only a small block
 * of K(X,Y) is assembled and stored at once, thereby achieving high performance and
 */
template <typename rbf_function>
void rbf_kernel<rbf_function>::mul( size_t dim, size_t n, size_t m, size_t nrhs,
		                            double *__restrict__  R, size_t ldR,
		                            const double         *X, size_t ldX,
									const double         *Y, size_t ldY,
									const double         *C, size_t ldC, size_t threads ) const
{
	// Block size.
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

template <typename rbf_function>
void rbf_kernel<rbf_function>::eval_column( size_t dim, size_t n, size_t j,
		                                          double* K, size_t ldK,
		                                    const double* X, size_t ldX,
									        const double* Y, size_t ldY ) const
{
	using simd_t = typename rbf_function::simd_type;

	size_t i = 0;

	if constexpr ( simd_t::size() > 1 )
	{
		simd_t x, y, r;

		for ( ; i + simd_t::size() < n; i += simd_t::size() )
		{
			y.fill(Y + j);
			x.load(X + i);
			r = abs(x-y);

			if ( dim > 1 )
			{
				r = r*r;
				for ( size_t d = 1; d < dim; ++d )
				{
					x.load(X + i + d*ldX);
					y.fill(Y + j + d*ldY);
					r = fmadd(x-y,x-y,r);
				}
				r = sqrt(r);
			}
			r = F(r*inv_sigma);
			r.store( K + i + j*ldK );
		}
	}

	for ( ; i < n; ++i )
	{
		double x = X[i];
		double y = Y[j];
		double r = std::abs(x-y);

		if ( dim > 1 )
		{
			r = r*r;
			for ( size_t d = 1; d < dim; ++d )
			{
				x = X[ i + d*ldX ];
				y = Y[ j + d*ldY ];
				r = std::fma(x-y,x-y,r);
			}
			r = sqrt(r);
		}
		K[ i + j*ldK ] = F(r*inv_sigma);
	}
}

template <typename rbf_function>
void rbf_kernel<rbf_function>::micro_kernel( size_t dim, simd_t mat[ simd_t::size() ],
		                                     const double *X, size_t ldX,
									         const double *Y, size_t ldY ) const
{
	simd_t x,y;

	x.load( X );
	for ( size_t k = 0; k < simd_t::size(); ++k )
	{
		y.fill( Y + k );
		mat[k] = abs(x-y);
	}

	if ( dim > 1 )
	{
		for ( size_t k = 0; k < simd_t::size(); ++k )
			mat[k] = mat[k]*mat[k];

		for ( size_t d = 1; d < dim; ++d )
		{
			x.load( X + d*ldX );
			for ( size_t k = 0; k < simd_t::size(); ++k )
			{
				y.fill( Y + d*ldY + k );
				mat[k] = fmadd(x-y,x-y,mat[k]);
			}

			for ( size_t k = 0; k < simd_t::size(); ++k )
				mat[k] = sqrt(mat[k]);
		}
	}

	simd_t scale { inv_sigma };
	for ( size_t k = 0; k < simd_t::size(); ++k )
		mat[k] = mat[k] * scale;

	F.eval(mat);
}

}

}

