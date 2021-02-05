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
#ifndef VLASOVIUS_MATH_RUMP_SUMMATION_H
#define VLASOVIUS_MATH_RUMP_SUMMATION_H

#include <cmath>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <type_traits>

namespace vlasovius
{

namespace misc
{

/*
 * \brief Rump's summation algorithm FastAccSum
 * \param begin An iterator pointing to the beginning of the range to be summed
 * \param end   An iterator pointing to the end       of the range to be summed
 * \return A faithful roundig of the exact sum of the range.
 * \warning Modifies the input sequence.
 * \warning Do not run this with fast-math optimisations. They break the algorithm.
 * \see "Ultimately Fast Accurate Summation", https://dx.doi.org/10.1137/080738490
 *
 * This algorithm requires iterators to one of the built-in floating point types:
 * float, double, or long double.
 *
 * Faithful rounding means the following. If the exact result of the sum is itself
 * a floating point number, then this number is returned. Otherwise, one of the two
 * neighbouring, closest floating point numbers is returned, though not necessesarily
 * the closer one. Thus, if s is the exact result of the sum, this algorithm returns:
 *
 * s                              -- if s is     a floating point number
 * predecessor(s) or successor(s) -- if s is not a floating point number
 *
 * Note that a pleasant property is that the sign of the result is always correct.
 */
template <typename iterator>
typename std::iterator_traits<iterator>::value_type
rump_summation( iterator begin, iterator end )
{
    using float_type = typename std::iterator_traits<iterator>::value_type;
    static_assert( std::is_floating_point<float_type>::value,
                   "Rump's summation algorithm is only implemented for float, double, and long double." );
    static_assert( std::numeric_limits<float_type>::is_iec559,
                   "Rump's summation algoritm is only implemented for IEEE-754/IEC-559 floating point numbers." );
    static_assert( std::numeric_limits<float_type>::has_denorm == std::denorm_present,
                   "Rump's summation algorithm requires support for denormalised numbers." );
    static_assert( std::numeric_limits<float_type>::round_style == std::round_to_nearest,
                   "Rump's summation algorithm requires round_to_nearest rounding." );
    
    constexpr float_type nan = std::numeric_limits<float_type>::signaling_NaN();
    constexpr float_type eps = std::numeric_limits<float_type>::epsilon()/2.0;
    constexpr float_type eta = std::numeric_limits<float_type>::denorm_min();

    size_t     n     { 0 };
    float_type T     { 0 };
    bool       error { false };
    for ( iterator i = begin; i != end; ++i )
    {
        float_type f  { *i };
        error = error || std::isnan(f) || std::isinf(f);
        T += std::abs(f);
        ++n;
    }
    T /= (1 - n*eps);

    if ( error  ) return nan;
    if ( n == 0 ) return 0;
    if ( n == 1 ) return *begin;
    if ( ((2*n+4)*n+6)*eps > 1 )
        throw std::domain_error { "Too many terms for Rump's summation algorithm." };

    if ( T <= eta/eps )
    {
        float_type result = 0;
        for ( iterator i = begin; i != end; ++i )
            result += *i;
        return result;
    }

    float_type t_dash { 0 }, phi;
    float_type sigma0, sigma, sigma_dash, q, tau, t, u;
    do
    {
        sigma = sigma_dash = sigma0 = (2*T)/(1-(3*n+1)*eps);
        for ( iterator i = begin; i != end; ++i )
        {
            sigma_dash = sigma + *i;
            q          = sigma_dash - sigma;
            *i         = *i - q;
            sigma      = sigma_dash;
        }

        tau    = sigma_dash - sigma0;
        t      = t_dash;
        t_dash = t + tau;

        if ( t_dash == 0 )
            return rump_summation(begin,end);

        q   = sigma0/(2*eps);
        u   = std::abs(  q/(1-eps) - q );
        phi = ((2*n*(n+2)*eps)*u)/(float_type(1)-5*eps);
        T   = std::min( ((float_type(1.5)+float_type(4)*eps)*(n*eps))*sigma0, (2*n*eps)*u );
    }
    while ( std::abs(t_dash) < phi && 4*T > eta/eps );

    float_type tau2 = (t-t_dash) + tau;

    float_type result = 0;
    for ( iterator i = begin; i != end; ++i )
        result += *i;
    result = t_dash + ( tau2 + result );
    return result;
}

}

}

#endif
