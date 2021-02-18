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

#include <cmath>

namespace vlasovius
{

namespace misc
{

template <>
class simd<double,simd_abi::scalar>
{
public:
	constexpr simd() noexcept = default;
	constexpr simd( const simd&  ) noexcept = default;
	constexpr simd(       simd&& ) noexcept = default;
	simd& operator=( const simd&  ) noexcept = default;
	simd& operator=(       simd&& ) noexcept = default;

	simd( double* data ) noexcept: v { *data } {}
	constexpr simd( double  data ) noexcept: v {  data } {}


	static constexpr size_t size() noexcept { return 1; }

	void zeros()                  noexcept {  v =  0; }
	void fill (       double  x ) noexcept {  v =  x; }
	void fill ( const double *x ) noexcept {  v = *x; }
	void load ( const double *x ) noexcept {  v = *x; }
	void store(       double *x ) noexcept { *x =  v; }

	simd operator~() const noexcept
	{
		double result = v;
		unsigned char* a = reinterpret_cast<unsigned char*>( &result );
		for ( size_t i = 0; i < sizeof(double); ++i )
			a[i] = ~(a[i]);
		return result;
	}

	double v {};
};


using simd_scalar_double = simd<double,simd_abi::scalar>;

inline
simd_scalar_double operator+( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	return v1.v + v2.v;
}

inline
simd_scalar_double operator-( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	return v1.v - v2.v;
}

inline
simd_scalar_double operator*( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	return v1.v * v2.v;
}

inline
simd_scalar_double operator/( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	return v1.v / v2.v;
}

inline
simd_scalar_double operator&( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	simd_scalar_double result;
	unsigned char* a1 = reinterpret_cast<unsigned char*>( &v1.v );
	unsigned char* a2 = reinterpret_cast<unsigned char*>( &v2.v );
	unsigned char* a3 = reinterpret_cast<unsigned char*>( &result.v );

	for ( size_t i = 0; i < sizeof(double); ++i )
		a3[i] = a1[i] & a2[i];

	return result;
}

inline
simd_scalar_double operator|( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	simd_scalar_double result;
	unsigned char* a1 = reinterpret_cast<unsigned char*>( &v1.v );
	unsigned char* a2 = reinterpret_cast<unsigned char*>( &v2.v );
	unsigned char* a3 = reinterpret_cast<unsigned char*>( &result.v );

	for ( size_t i = 0; i < sizeof(double); ++i )
		a3[i] = a1[i] | a2[i];

	return result;
}

inline
simd_scalar_double operator^( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	simd_scalar_double result;
	unsigned char* a1 = reinterpret_cast<unsigned char*>( &v1.v );
	unsigned char* a2 = reinterpret_cast<unsigned char*>( &v2.v );
	unsigned char* a3 = reinterpret_cast<unsigned char*>( &result.v );

	for ( size_t i = 0; i < sizeof(double); ++i )
		a3[i] = a1[i] ^ a2[i];

	return result;
}

inline
simd_scalar_double fmadd( simd_scalar_double v1, simd_scalar_double v2, simd_scalar_double v3 ) noexcept
{
	return std::fma(v1.v, v2.v, v3.v );
}

inline
simd_scalar_double fmsub( simd_scalar_double v1, simd_scalar_double v2, simd_scalar_double v3 ) noexcept
{
	return std::fma(v1.v, v2.v, -v3.v );
}

inline
simd_scalar_double sqrt( simd_scalar_double v ) noexcept
{
	return std::sqrt(v.v);
}

inline
simd_scalar_double abs( simd_scalar_double v ) noexcept
{
	return std::abs(v.v);
}

inline
simd_scalar_double less_than( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	constexpr simd_scalar_double FALSE { 0.0 };
	return (v1.v<v2.v) ? ~FALSE : FALSE;
}

inline
simd_scalar_double greater_than( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	constexpr simd_scalar_double FALSE { 0.0 };
	return (v1.v>v2.v) ? ~FALSE : FALSE;
}

inline
simd_scalar_double lessequal_than( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	constexpr simd_scalar_double FALSE { 0.0 };
	return (v1.v<=v2.v) ? ~FALSE : FALSE;
}

inline
simd_scalar_double greaterequal_than( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	constexpr simd_scalar_double FALSE { 0.0 };
	return (v1.v>=v2.v) ? ~FALSE : FALSE;
}

inline
simd_scalar_double equals( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	constexpr simd_scalar_double FALSE { 0.0 };
	return (v1.v==v2.v) ? ~FALSE : FALSE;
}

inline
simd_scalar_double unequals( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	constexpr simd_scalar_double FALSE { 0.0 };
	return (v1.v!=v2.v) ? ~FALSE : FALSE;
}

inline
simd_scalar_double max( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	return std::max(v1.v,v2.v);
}

inline
simd_scalar_double min( simd_scalar_double v1, simd_scalar_double v2 ) noexcept
{
	return std::min(v1.v,v2.v);
}

}

}
