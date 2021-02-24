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
		                             double *__restrict__ K, size_t ldK,
		                             const double        *X, size_t ldX,
									 const double        *Y, size_t ldY, size_t threads ) const
{
	if ( threads > 1 )
	{
		#pragma omp parallel for num_threads(threads)
		for ( size_t j = 0; j < m; ++j )
			eval_column( dim, n, m, j, K, ldK, X, ldX, Y, ldY );
	}
	else
	{
		for ( size_t j = 0; j < m; ++j )
			eval_column( dim, n, m, j, K, ldK, X, ldX, Y, ldY );
	}
}

template <typename rbf_function>
void rbf_kernel<rbf_function>::eval_column( size_t dim, size_t n, size_t m, size_t j,
		                                    double *__restrict__ K, size_t ldK,
		                                    const double        *X, size_t ldX,
									        const double        *Y, size_t ldY ) const
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


}

}
