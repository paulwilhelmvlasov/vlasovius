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

void compute_coefficients( size_t dim, size_t k, double *result );

}

template <size_t dim, size_t k>
wendland<dim,k>::wendland()
{
	wendland_impl::compute_coefficients( dim, k, c );
}

template <size_t dim, size_t k>
double wendland<dim,k>::operator()( double r ) const noexcept
{
	constexpr size_t N { (dim/2) + 3*k + 2 };

	r = std::abs(r);

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
	z = std::fma(f,z,c[N-1]) - z_prev;
	return ( r < 1 ) ? z : 0;
}

template <size_t dim, size_t k>
arma::vec wendland<dim,k>::operator()( arma::vec r ) const
{
	for ( size_t i = 0; i < r.size(); ++i )
		r[i] = (*this)(r[i]);
	return r;
}

}

}
