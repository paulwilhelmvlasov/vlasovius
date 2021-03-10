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

void compute_coefficients( size_t dim, size_t k, double *result, double *integral );

}

template <size_t dim, size_t k, typename simd_t>
wendland<dim,k,simd_t>::wendland()
{
	wendland_impl::compute_coefficients( dim, k, c, &integral_ );
}

template <size_t dim, size_t k, typename simd_t>
double wendland<dim,k,simd_t>::operator()( double r ) const noexcept
{
	constexpr size_t N { (dim/2) + 3*k + 2 };

	r = std::abs(r);
	if ( r >= 1 ) return 0;

	// Clenshaw's algorithm for evaluating Chebyshev expansions.
	double f      { 4*r - 2 };
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

template <size_t dim, size_t k, typename simd_t>
simd_t wendland<dim,k,simd_t>::operator()( simd_t r ) const noexcept
{
	constexpr size_t N { (dim/2) + 3*k + 2 };

	r = abs(r);
	r = r + r - 1.0; r = r + r;
	simd_t z_prev { c[0] }, z { fmadd(r,c[0],c[1]) };
	for ( size_t i = 2; i < N-1; ++i )
	{
		simd_t tmp = fmadd(r,z,c[i])-z_prev;
		z_prev = z;
		z      = tmp;
	}
	r = 0.5*r;
	z = (fmadd(r,z,c[N-1]) - z_prev) & less_than(r,1.0);

	return z;
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
	simd_t tmp, c_tmp, z[ vecsize ], z_prev[ vecsize ];

	tmp.fill(1.0);
	for ( size_t l = 0; l < vecsize; ++l )
	{
		r[l] = abs(r[l]);
		r[l] = r[l] + r[l] - tmp;
		r[l] = r[l] + r[l];
	}

	tmp.fill(c[0]); c_tmp.fill(c[1]);
	for ( size_t l = 0; l < vecsize; ++l )
	{
		z_prev[l] = tmp;
		z[l] = fmadd(r[l],tmp,c_tmp);
	}

	for ( size_t i = 2; i < N-1; ++i )
	{
		c_tmp.fill(c[i]);
		for ( size_t l = 0; l < vecsize; ++l )
		{
			tmp = fmadd(r[l],z[l],c_tmp)-z_prev[l];
			z_prev[l] = z[l];
			z[l]      = tmp;
		}
	}

	tmp.fill(0.5);
	for ( size_t l = 0; l < vecsize; ++l ) r[l] = tmp*r[l];

	c_tmp.fill(c[N-1]);
	for ( size_t l = 0; l < vecsize; ++l ) z[l] = fmadd(r[l],z[l],c_tmp) - z_prev[l];

	tmp.fill(1.0);
	for ( size_t l = 0; l < vecsize; ++l ) r[l] = z[l] & less_than(r[l],tmp);
}

/*!
 * \brief Computes the integral of the Wendland function over the positive reals.
 * \int_{0}^{\infty} W(r)\,{\mathrm dr} = \int_{0}^{1} W(r)\,{\mathrm d}r.
 */
template <size_t dim, size_t k, typename simd_t> inline
double wendland<dim,k,simd_t>::integral() const noexcept
{
	return integral_;
}

}

}
