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
#include <vlasovius/kernels/wendland.h>
#include <boost/multiprecision/gmp.hpp>

#include <vector>

namespace
{

using rational = boost::multiprecision::mpq_rational;

// This function computes the coefficients of the Wendland function
// in the monomial basis. [wend2005,Theorem 9.12]
// Because the monomial basis is horribly ill-conditioned, these computations
// need to be carried out in exact, rational arithmetic.
std::vector<rational>
monomial_coeffs( size_t dim, size_t k )
{
	size_t N { (dim/2) + 3*k + 2 };
	size_t l { (dim/2) +   k + 1 };

	std::vector<rational> tmp1(N), tmp2(N), result(N);
	rational *data_old { tmp1.data() };
	rational *data_new { tmp2.data() };

	// s = 0; Computes the binomial coefficients.
    data_old[0] = 1;
    for ( size_t j = 1; j <= l; ++j )
        data_old[j] = - data_old[j-1] * rational(l+1-j,j);

    for ( size_t s = 0; s < k; ++s )
    {
        data_new[0] = data_new[1] = 0;
        for ( size_t j = 0; j <= l + 2*s; ++j )
            data_new[0] += data_old[j]/(j+2);

        for ( size_t j = 2; j <= l + 2*s  + 2; ++j )
            data_new[j] = -data_old[j-2]/j;

        std::swap( data_old, data_new );
    }

    // Scale such that W(0) = 1.
    rational scale = data_old[0];

    for ( size_t j = 0; j < N; ++j )
    {
    	result[ j ] = data_old[ j ]/scale;
    }
    return result;
}

// Converts a coefficient vector given in the monomial basis
// to the corresponding coefficients in the Chebyshev expansion.
// We use the Chebyshev polynomials of the first kind, rescaled to the interval
// [0,1]. The Chebyshev polynomials form a stable basis for the space of
// polynomials and are thus much more suitable for numerical computation.
// Also carried out in exact rational arithmetic.
void monomial2chebyshev( std::vector<rational> &coeff )
{
	size_t N { coeff.size() };
	if ( N == 1 ) return;

	// Coefficients of the Chebyshev polynomials in monomial basis.
	// N*N upper triangular matrix in column major ordering.
	std::vector<rational> A(N*N);
	A[ 0 + N*0 ] =  1;
	A[ 0 + N*1 ] = -1;	A[ 1 + N*1 ] =  2;
	for ( size_t i = 2; i < N; ++i )
	{
		// Tn+1 = (4x-2) Tn - Tn-1
		for ( size_t j = 0; j < i; ++j )
		{
			A[j+1 + N*i] += 4*A[j + N*(i-1)];
			A[j   + N*i] -= 2*A[j + N*(i-1)];
		}
		for ( size_t j = 0; j < i-1; ++j )
		{
			A[j + N*i] -= A[j + N*(i-2)];
		}
	}

	// Solve the resulting upper triangular system by backward substitution.
	for ( size_t i = N; i-- > 0; )
	{
		for ( size_t j = i+1; j < N; ++j )
			coeff[i] -= A[i + N*j]*coeff[j];
		coeff[i] /= A[i + N*i];
	}
}

}

namespace vlasovius
{

namespace kernels
{

namespace wendland_impl
{

// Computes the coefficients of Wendlands functions in the Chebyshev
// basis. result[0] will contain the coefficient of the highest degree,
// result[ (dim/2) + 3*k + 1 ] will contain the coefficient of the constant
// polynomial. It thus assumes that the result pointer points to an array of at
// least (dim/2) + 3*k + 2 elements. This "reverse" ordering is used, because in
// Clenshaw's algorithm for evaluating the polynomial, the coefficients are
// accessed in this order.
void compute_coefficients( size_t dim, size_t k, double *result, double *result_int )
{
	// Compute coeffcients in exact arithmetic.
	std::vector<rational> c = monomial_coeffs(dim,k);

	// Coefficients of the integral.
	std::vector<rational> c_int(c.size()+1);
	for ( size_t i = 0; i < c.size(); ++i )
		c_int[i+1] = rational(1,i+1)*c[i];

	monomial2chebyshev(c);
	monomial2chebyshev(c_int);

	// Reverse order and convert to double.
	for ( size_t i = c.size(); i-- > 0; )
		*result++ = static_cast<double>(c[i]);

	for ( size_t i = c_int.size(); i-- > 0; )
		*result_int++ = static_cast<double>(c_int[i]);
}

}

}

}
