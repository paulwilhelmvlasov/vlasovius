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
#ifndef VLASOVIUS_GEOMETRY_BOUNDING_BOX_H
#define VLASOVIUS_GEOMETRY_BOUNDING_BOX_H

#include <armadillo>

namespace vlasovius
{

namespace geometry
{

namespace bounding_box
{

// Under the term bounding_box we understand an axis-aligned D-dimensional
// cuboid. They are represented as a 2*D-dimensional row-vector: the first D
// numbers represent the minimum coordinates, the last three the maximum
// coordinates.
//
// Let us for example consider the case D = 3. Then unit cube [0,1]³ is
// represented as (0,0,0,1,1,1).
//
// This header declares functions which solve common tasks involving such boxes:
// computing intersection, checking if one box is a subset of another,
// or if a box contains a point or not.
//
// Two interfaces are provided:
// 1. Simple interface: points and boxes are passed as arma::rowvec
//
//    This interface is easy to use, but often incurs unnecessary overhead
//    for copying and memory allocation.
//
// 2. BLAS-like interface:
//
//   Here the user passes raw memory pointers, the dimension D, and the
//   stride between subsequent entries. The added flexibility makes code
//   harder to read, but it avoids unnecessary copies.
//
// It is assumed that the user supplies vectors of appropiate sizes, no
// boundary checks are performed.


// Checks if the given bounding_box represents the empty set.
//
// Examples for D=2:
// (0,0,1, 1) -- false.  The unit cube is not empty.
// (0,0,0, 0) -- false.  This contains (only) the point (0,0).
// (0,0,0,-1) -- true.   Second coordinate min == 0 > max == -1.
bool is_empty( const arma::rowvec &b );
bool is_empty( size_t dim, const double *b, size_t stride );

// Computes the D-dimensional Lebesgue measure, i.e., the (hyper-)volume.
double measure( const arma::rowvec &a );
double measure( size_t dim, const double *a, size_t stride );

// Checks if the intersection of two bounding_boxes is non-empty.
bool intersect( const arma::rowvec &a, const arma::rowvec &b );
bool intersect( size_t dim, const double *a, size_t a_stride,
		                    const double *b, size_t b_stride );

// Computes the intersection of two bounding_boxes
// BLAS-version: target box c *is* allowed to coincide with a or b.
arma::rowvec intersection( const arma::rowvec &a, const arma::rowvec &b );
void         intersection( size_t dim, const double *a, size_t a_stride,
		                               const double *b, size_t b_stride,
									         double *c, size_t c_stride );

// Computes if the first box is a subset of the other.
bool is_subset( const arma::rowvec &a, const arma::rowvec &b );
bool is_subset( size_t dim, const double *a, size_t a_stride,
                            const double *b, size_t b_stride );

// Checks whether box a contains *point* b.
// a is a 2xD-dimensional array, b is a 1xD-dimensional.
bool contains_point( const arma::rowvec &a, const arma::rowvec &b );
bool contains_point( size_t dim, const double *a, size_t a_stride,
		                         const double *b, size_t b_stride );

}

}

}

#endif
