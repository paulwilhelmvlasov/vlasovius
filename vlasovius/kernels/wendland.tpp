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
#include <array>
#include <cmath>

#include <vlasovius/config.h>

#ifdef HAVE_AVX2_INSTRUCTIONS
#include <immintrin.h>
#endif

namespace vlasovius
{

namespace kernels
{

namespace wendland_impl
{

template <size_t dim, size_t k>
constexpr size_t num_coeffs()
{
    return (dim/2) + 3*k + 2;
}

template <size_t dim, size_t k>
using wendland_coeffs = std::array< double, num_coeffs<dim,k>() >;

template <size_t dim, size_t k> constexpr
wendland_coeffs<dim,k> coeffs()
{
	constexpr size_t l { dim/2 + k + 1 };

	wendland_coeffs<dim,k> tmp1 {}, tmp2 {}, result {};
	double *data_old { tmp1.data() };
	double *data_new { tmp2.data() };

	// s = 0; Computes the binomial coefficients.
    data_old[0] = 1;
    for ( size_t j = 1; j <= l; ++j )
    {
        data_old[j] = data_old[j-1] * ( - static_cast<double>(l+1-j) /
                                          static_cast<double>(j)    );
    }

    for ( size_t s = 0; s < k; ++s )
    {
        data_new[0] = data_new[1] = 0;
        for ( size_t j = 0; j <= l + 2*s; ++j )
            data_new[0] += data_old[j]/(j+2);

        for ( size_t j = 2; j <= l + 2*s  + 2; ++j )
            data_new[j] = -data_old[j-2]/j;

        // Swap new and old.
        double *tmp = data_old;
        data_old = data_new;
        data_new = tmp;
    }

    // Scale such that W(0) = 1.
    double scale = 1/data_old[0];

    for ( size_t j = 0; j < num_coeffs<dim,k>(); ++j )
    {
    	result[ j ] = data_old[ num_coeffs<dim,k>() - 1 - j ]*scale;
    }

    return result;
};

}

template <size_t dim, size_t k>
double wendland( double r )
{
	using wendland_impl::wendland_coeffs;
	constexpr    size_t                    N { wendland_impl::num_coeffs<dim,k>() };
	constexpr const wendland_coeffs<dim,k> c { wendland_impl::    coeffs<dim,k>() };

	r = std::abs(r);
    r = std::min(r,1.0);
    double result { c[0] };
    for ( size_t i = 1; i < N; ++i )
    	result = std::fma( r, result, c[i] );
    return result;
}

template <size_t dim, size_t k>
arma::vec scalar_wendland( const arma::vec &r )
{
	arma::vec result( r.size() );
	for ( size_t i = 0; i < r.size(); ++i )
		result(i) = wendland<dim,k>(r[i]);
	return result;
}


#ifdef HAVE_AVX2_INSTRUCTIONS
namespace wendland_impl
{

template <size_t dim, size_t k>
struct alignas(32) wendland_coeffs_avx
{
	__m256d data[ num_coeffs<dim,k>() ];
};


template <size_t dim, size_t k> constexpr
wendland_coeffs_avx<dim,k> avx_coeffs()
{
	wendland_coeffs_avx<dim,k> result {};
	wendland_coeffs    <dim,k> c { coeffs<dim,k>() };

	for ( size_t i = 0; i < c.size(); ++i )
	{
        result.data[i] = __m256d { c[i], c[i], c[i], c[i] };
	}
	return result;
}

}

template <size_t dim, size_t k>
arma::vec avx_wendland( const arma::vec &rr )
{
	using wendland_impl::wendland_coeffs;
	using wendland_impl::wendland_coeffs_avx;
	constexpr size_t                     N  { wendland_impl::num_coeffs<dim,k>() };
	constexpr wendland_coeffs    <dim,k> c  { wendland_impl::    coeffs<dim,k>() };
    constexpr wendland_coeffs_avx<dim,k> cc { wendland_impl::avx_coeffs<dim,k>() };

	__m256d sign_mask = _mm256_set1_pd(-0.0 ); // Used to compute absolute value.
	__m256d ones      = _mm256_set1_pd( 1.0 ); // Used to determine if abs(r)>1.

	arma::vec result( rr.size() ); size_t chunk;
	for ( chunk = 0; chunk < rr.size()/4; ++chunk )
	{
		__m256d r = _mm256_loadu_pd( rr.memptr() + 4*chunk );
		r = _mm256_andnot_pd( sign_mask, r ); // r = abs(r);
	    r = _mm256_min_pd( r, ones );         // r = min(r,1);

	    __m256d res = cc.data[0];
	    for ( size_t j = 1; j < N; ++j )
	    	res = _mm256_fmadd_pd( res, r, cc.data[j] );

	    _mm256_store_pd( result.memptr() + 4*chunk, res );
	}

	for ( size_t i = 4*chunk; i < rr.size(); ++i )
	{
		double r = rr(i);
		r = std::abs(r);
		r = std::min(r,1.0);
		double res { c[0] };
		for ( size_t j = 1; j < N; ++j )
			res = c[j] + r*res;
		result(i) = res;
	}

	return result;
}

template <size_t dim, size_t k>
arma::vec wendland( const arma::vec &rr )
{
	return avx_wendland<dim,k>(rr);
}
#else
template <size_t dim, size_t k>
arma::vec wendland( const arma::vec &rr )
{
	return scalar_wendland<dim,k>(rr);
}
#endif

}

}
