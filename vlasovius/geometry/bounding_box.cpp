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
#include <vlasovius/geometry/bounding_box.h>

#include <algorithm> // For std::min(), std::max(), etc.

namespace vlasovius
{

namespace geometry
{

namespace bounding_box
{

bool is_empty( const arma::rowvec &b )
{
	return is_empty( b.n_cols/2, b.memptr(), 1 );
}

bool is_empty( size_t dim, const double *b, size_t stride )
{
	bool result { false };
	for ( size_t d = 0; d < dim; ++d )
	{
		result |= (b[d*stride] > b[(dim+d)*stride]);
	}
	return result;
}

double measure( const arma::rowvec &a )
{
	return measure( a.n_cols/2, a.memptr(), 1 );
}

double measure( size_t dim, const double *a, size_t stride )
{
	double meas { 1 };
	for ( size_t d = 0; d < dim; ++d )
	{
		meas *= std::max( 0., a[(d+dim)*stride] - a[d*stride] );
	}
	return meas;
}

bool intersect( const arma::rowvec &a, const arma::rowvec &b )
{
	return intersect( a.n_cols/2, a.memptr(), 1, b.memptr(), 1 );
}

bool intersect( size_t dim, const double *a, size_t a_stride,
		                    const double *b, size_t b_stride )
{
	bool result { true };
	for ( size_t d = 0; d < dim; ++d )
	{
		result &= (a[d*a_stride] <= b[(d+dim)*b_stride ]);
		result &= (b[d*b_stride] <= a[(d+dim)*a_stride ]);
	}
	return result;
}

arma::rowvec intersection( const arma::rowvec &a, const arma::rowvec &b )
{
	arma::rowvec c(a.n_cols);
	intersection( a.n_cols/2, a.memptr(), 1, b.memptr(), 1, c.memptr(), 1 );
	return c;
}

void intersection( size_t dim, const double *a, size_t a_stride,
		                       const double *b, size_t b_stride,
                                     double *c, size_t c_stride )
{
	for ( size_t d = 0; d < dim; ++d )
	{
		const double a_min { a[(d    )*a_stride] };
		const double a_max { a[(d+dim)*a_stride] };
		const double b_min { b[(d    )*b_stride] };
		const double b_max { b[(d+dim)*b_stride] };

		c[(d    )*c_stride] = std::max( a_min, b_min );
		c[(d+dim)*c_stride] = std::min( a_max, b_max );
	}
}

bool is_subset( const arma::rowvec &a, const arma::rowvec &b )
{
	return is_subset( a.n_cols/2, a.memptr(), 1, b.memptr(), 1 );
}

bool is_subset( size_t dim, const double *a, size_t a_stride,
                            const double *b, size_t b_stride )
{
	bool result { true };
	for ( size_t d = 0; d < dim; ++d )
	{
		result &= a[(d    )*a_stride] >= b[(d    )*b_stride];
		result &= a[(d+dim)*a_stride] <= b[(d+dim)*b_stride];
	}
	return result;
}

bool contains_point( const arma::rowvec &a, const arma::rowvec &b )
{
	return contains_point( a.n_cols/2, a.memptr(), 1, b.memptr(), 1 );
}

bool contains_point( size_t dim, const double *a, size_t a_stride,
		                         const double *b, size_t b_stride )
{
	bool result { true };
	for ( size_t d = 0; d < dim; ++d )
	{
		result &= a[(d    )*a_stride] <= b[d*b_stride];
		result &= a[(d+dim)*a_stride] >= b[d*b_stride];
	}
	return result;
}

}

}

}
