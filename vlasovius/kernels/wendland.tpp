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

namespace wendland_impl
{

void compute_coefficients( size_t dim, size_t k, double *result, double *result_int );

template <size_t N>
double clenshaw( const double c[N], double x )
{
	// Clenshaw's algorithm for evaluating Chebyshev expansions.
	double f      { 4*x - 2 };
	double z_prev { c[0] };
	double z      { std::fma(f,c[0],c[1]) };
	for ( size_t i = 2; i < N-1; ++i )
	{
		double tmp = std::fma(f,z,c[i]) - z_prev;
		z_prev = z;
		z      = tmp;
	}
	f = 0.5*f;
	return std::fma(f,z,c[N-1]) - z_prev;
}

template <size_t N, size_t M, typename simd_t>
void clenshaw_simd( const double c[N], simd_t x[M] ) noexcept
{
	// Clenshaw's algorithm for evaluating Chebyshev expansions,
	// vectorised SIMD version.
	simd_t tmp, c_tmp, z[ M ], z_prev[ M ];

	tmp.fill(1.0);
	for ( size_t m = 0; m < M; ++m )
	{
		x[m] = x[m] + x[m] - tmp;
		x[m] = x[m] + x[m];
	}

	tmp.fill(c[0]); c_tmp.fill(c[1]);
	for ( size_t m = 0; m < M; ++m )
	{
		z_prev[m] = tmp;
		z[m] = fmadd(x[m],tmp,c_tmp);
	}

	for ( size_t n = 2; n < N-1; ++n )
	{
		c_tmp.fill(c[n]);
		for ( size_t m = 0; m < M; ++m )
		{
			tmp = fmadd(x[m],z[m],c_tmp)-z_prev[m];
			z_prev[m] = z[m];
			z[m]      = tmp;
		}
	}

	tmp.fill(0.5); c_tmp.fill(c[N-1]);
	for ( size_t m = 0; m < M; ++m )
	{
		x[m] = tmp*x[m];
		x[m] = fmadd(x[m],z[m],c_tmp) - z_prev[m];
	}
}

}

template <size_t dim, size_t k, typename simd_t>
wendland<dim,k,simd_t>::wendland()
{
	wendland_impl::compute_coefficients( dim, k, c, c_int );
}

template <size_t dim, size_t k, typename simd_t>
double wendland<dim,k,simd_t>::operator()( double r ) const noexcept
{
	constexpr size_t N { (dim/2) + 3*k + 2 };

	r = std::abs(r);
	if ( r >= 1 ) return 0;
	else return wendland_impl::clenshaw<N>(c,r);
}

template <size_t dim, size_t k, typename simd_t>
simd_t wendland<dim,k,simd_t>::operator()( simd_t r ) const noexcept
{
	constexpr size_t N { (dim/2) + 3*k + 2 };

	r = abs(r);
	simd_t mask = less_than(r,simd_t(1.0));
	wendland_impl::clenshaw_simd<N,1,simd_t>(c,&r);
	return r & mask;
}

template <size_t dim, size_t k, typename simd_t>
arma::vec wendland<dim,k,simd_t>::operator()( arma::vec rvec ) const
{
	eval( rvec.memptr(), rvec.size() );
	return rvec;
}

template <size_t dim, size_t k, typename simd_t>
void wendland<dim,k,simd_t>::eval( double *__restrict__ result, size_t n ) const noexcept
{
	size_t i { 0 };
	if constexpr ( simd_t::size() > 1 )
	{
		simd_t r;
		for ( ; i + simd_t::size() < n; i += simd_t::size() )
		{
			r.load( result + i );
			r = (*this)(r);
			r.store( result + i );
		}
	}

	// Compute the remaining entries in scalar mode.
	for ( ; i < n; ++i )
		result[i] = (*this)( result[i] );
}

template <size_t dim, size_t k, typename simd_t>
void wendland<dim,k,simd_t>::eval( simd_t r[ vecsize ] ) const noexcept
{
	constexpr size_t N { (dim/2) + 3*k + 2 };

	simd_t mask[ vecsize ]; simd_t ones { 1.0 };
	for ( size_t m = 0; m < vecsize; ++m )
	{
		r   [m] = abs(r[m]);
		mask[m] = less_than( r[m], ones );
	}

	wendland_impl::clenshaw_simd<N,vecsize,simd_t>(c,r);

	for ( size_t m = 0; m < vecsize; ++m )
	{
		r[m] = r[m] & mask[m];
	}
}

/*!
 * \brief Computes the definite integral of the Wendland function from 0 to r.
 * \int_{0}^{r} W(t)\,{\mathrm dt}
 */
template <size_t dim, size_t k, typename simd_t> inline
double wendland<dim,k,simd_t>::integral( double r ) const noexcept
{
	constexpr size_t N { (dim/2) + 3*k + 2 };

	bool sign { r < 0.0 };
	r = std::min(std::abs(r),1.0);
	r = wendland_impl::clenshaw<N+1>( c_int, r );
	return sign ? -r : r;
}

}

}
