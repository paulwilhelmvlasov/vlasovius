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

#include <immintrin.h>

namespace vlasovius
{

namespace misc
{

template <>
class simd<double,simd_abi::avx>
{
public:
	simd() noexcept = default;
	simd( const simd&  ) noexcept = default;
	simd(       simd&& ) noexcept = default;
	simd& operator=( const simd&  ) noexcept = default;
	simd& operator=(       simd&& ) noexcept = default;

	simd( double* data ) noexcept: v { _mm256_loadu_pd( data ) } {}
	simd( double  data ) noexcept: v { _mm256_set1_pd ( data ) } {}
	simd( __m256d data ) noexcept: v { data }                    {}


	static constexpr size_t size() noexcept { return 4; }

	void zeros()                  noexcept { v = _mm256_xor_pd(v,v);       }
	void fill (       double  x ) noexcept { v = _mm256_set1_pd     ( x ); }
	void fill ( const double *x ) noexcept { v = _mm256_broadcast_sd( x ); }
	void load ( const double *x ) noexcept { v = _mm256_loadu_pd    ( x ); }
	void store(       double *x ) noexcept { _mm256_storeu_pd(x,v);        }


	__m256d v;
};


using simd_avx_double = simd<double,simd_abi::avx>;

inline
simd_avx_double operator+( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_add_pd( v1.v, v2.v );
}

inline
simd_avx_double operator-( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_sub_pd( v1.v, v2.v );
}

inline
simd_avx_double operator*( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_mul_pd( v1.v, v2.v );
}

inline
simd_avx_double operator/( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_div_pd( v1.v, v2.v );
}

inline
simd_avx_double operator&( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_and_pd( v1.v, v2.v );
}

inline
simd_avx_double operator|( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_or_pd( v1.v, v2.v );
}

inline
simd_avx_double operator^( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_xor_pd( v1.v, v2.v );
}

inline
simd_avx_double fmadd( simd_avx_double v1, simd_avx_double v2, simd_avx_double v3 ) noexcept
{
	return _mm256_fmadd_pd( v1.v, v2.v, v3.v );
}

inline
simd_avx_double fmsub( simd_avx_double v1, simd_avx_double v2, simd_avx_double v3 ) noexcept
{
	return _mm256_fmsub_pd( v1.v, v2.v, v3.v );
}

inline
simd_avx_double sqrt( simd_avx_double v ) noexcept
{
	return _mm256_sqrt_pd(v.v);
}

inline
simd_avx_double abs( simd_avx_double v ) noexcept
{
	return _mm256_andnot_pd( _mm256_set1_pd(-0.0), v.v );
}

inline
simd_avx_double less_than( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_cmp_pd(v1.v,v2.v,_CMP_LT_OQ);
}

inline
simd_avx_double greater_than( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_cmp_pd(v1.v,v2.v,_CMP_GT_OQ);
}

inline
simd_avx_double lessequal_than( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_cmp_pd(v1.v,v2.v,_CMP_LE_OQ);
}

inline
simd_avx_double greaterequal_than( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_cmp_pd(v1.v,v2.v,_CMP_GE_OQ);
}

inline
simd_avx_double equals( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_cmp_pd(v1.v,v2.v,_CMP_EQ_OQ);
}

inline
simd_avx_double unequals( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_cmp_pd(v1.v,v2.v,_CMP_NEQ_OQ);
}

inline
simd_avx_double max( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_max_pd(v1.v,v2.v);
}

inline
simd_avx_double min( simd_avx_double v1, simd_avx_double v2 ) noexcept
{
	return _mm256_min_pd(v1.v,v2.v);
}

}

}
