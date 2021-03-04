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
#include <vlasovius/geometry/kd_tree.h>

#include <vector>
#include <numeric>   // For std::iota
#include <algorithm> // For std::nth_element

#include <vlasovius/geometry/bounding_box.h>

namespace
{

void split( const arma::mat &X, arma::uword *begin, arma::uword *end, size_t l )
{
	// We will use linear search for small boxes.
	// Magic number:  1 cache line = 64 bytes = 8 doubles.
	// We can stop splitting then.
	if ( (end-begin) > 8 )
	{
		const arma::uword split_dim { l % X.n_cols };
		auto cmp = [&X,split_dim]( arma::uword i, arma::uword j ) noexcept -> bool
		{
			return X(i,split_dim) < X(j,split_dim);
		};

		arma::uword *const median { begin + (end-begin)/2 };
		std::nth_element( begin, median, end, cmp );

		split( X, begin,    median, l+1 );
		split( X, median+1, end,    l+1 );
	}
}

void query( const arma::mat &X,
		    arma::uword l, arma::uword begin, arma::uword end,
			arma::rowvec &b, const arma::rowvec &q, std::vector<arma::uword> &result )
{
	using ::vlasovius::geometry::bounding_box::is_subset;
	using ::vlasovius::geometry::bounding_box::intersect;
	using ::vlasovius::geometry::bounding_box::contains_point;

	// We will use linear search for small boxes.
	// Magic number:  1 cache line = 64 bytes = 8 doubles.
	if ( (end-begin) > 8 )
	{
		if ( is_subset(b,q) )
		{
			const size_t old_size { result.size() };
			const size_t new_size { old_size + (end-begin) };
			result.resize( new_size );
			std::iota( result.data() + old_size, result.data() + new_size, begin );
		}
		else
		{
			const arma::uword median    { begin + (end-begin)/2 };
			const arma::uword split_dim { l % X.n_cols };
			const double      split_val { X(median,split_dim) };

			double tmp;

			// Search left branch if necessary.
			tmp = b(X.n_cols + split_dim); b(X.n_cols + split_dim) = split_val;
			if ( intersect(b,q) ) query(X,l+1,begin,median,b,q,result);
			b(X.n_cols + split_dim) = tmp;

			// Add median if necessary.
			if ( contains_point( X.n_cols, q.memptr(), 1, &X(median,0), X.n_rows ) )
				result.push_back(median);

			// Search right branch if necessary.
			tmp = b(split_dim); b(split_dim) = split_val;
			if ( intersect(b,q) ) query(X,l+1,median+1,end,b,q,result);
			b(split_dim) = tmp;
		}
	}
	else
	{
		for ( arma::uword i = begin; i < end; ++i )
		{
			if ( contains_point( X.n_cols, q.memptr(), 1, &X(i,0), X.n_rows ) )
			{
				result.push_back(i);
			}
		}
	}
}

void traverse_cover( const arma::mat &X,
					 arma::uword target_level, arma::uword begin, arma::uword end,
		             arma::uword box_level,	arma::uword box_number, arma::rowvec &b,
					 arma::mat &cover )
{
	if ( box_level == target_level )
	{
		size_t offset { target_level ? ((size_t(1)<<target_level)-size_t(1)) : size_t(0) };
		cover.row( box_number - offset ) = b;
	}
	else
	{
		const arma::uword median    { begin + (end-begin)/2 };
		const arma::uword split_dim { box_level % X.n_cols };
		const double      split_val { X(median,split_dim) };

		double tmp;

		// Search left branch if necessary.
		tmp = b(X.n_cols + split_dim); b(X.n_cols + split_dim) = split_val;
		traverse_cover(X,target_level,begin,median,box_level+1,2*box_number+1,b,cover);
		b(X.n_cols + split_dim) = tmp;

		// Search right branch if necessary.
		tmp = b(split_dim); b(split_dim) = split_val;
		traverse_cover(X,target_level,median+1,end,box_level+1,2*box_number+2,b,cover);
		b(split_dim) = tmp;
	}
}

}

namespace vlasovius
{

namespace geometry
{

kd_tree::kd_tree( const arma::mat &p_X ):
		idx( p_X.n_rows )
{
	if ( p_X.empty() )
		return;

	arma::uword *begin { idx.memptr() };
	arma::uword *end   { idx.memptr() + p_X.n_rows };
	std::iota( begin, end, 0 );
	split( p_X, begin, end, 0 );
	X = p_X.rows(idx);

}

arma::mat kd_tree::point_query( const arma::rowvec &q ) const
{
	if ( X.empty() || q.empty() )
		return arma::mat(0,0);

	const arma::uword dim { X.n_cols };
	if ( q.n_cols != 2*dim )
		throw std::logic_error { "vlasovius::misc::kd_tree::query(): "
		                         "dimension mismatch of query box and stored points." };

	// Bounding box of the root node is the whole space.
	arma::rowvec b( 2*dim );
	for ( arma::uword d = 0; d < dim; ++d )
	{
		b(d    ) = std::numeric_limits<double>::lowest();
		b(d+dim) = std::numeric_limits<double>::max();
	}

	std::vector<arma::uword> result_positions;
	query( X, 0, 0, X.n_rows, b, q, result_positions );

	arma::mat result( result_positions.size(), dim );
	for ( size_t d = 0; d < dim; ++d )
	{
		for ( size_t i = 0; i < result_positions.size(); ++i )
		{
			result(i,d) = X( result_positions[i], d );
		}
	}
	return result;
}

arma::uvec kd_tree::index_query( const arma::rowvec &q ) const
{
	if ( X.empty() || q.empty() )
		return arma::uvec {};

	const arma::uword dim { X.n_cols };
	if ( q.n_cols != 2*dim )
		throw std::logic_error { "vlasovius::misc::kd_tree::query(): "
		                         "dimension mismatch of query box and stored points." };

	// Bounding box of the root node is the whole space.
	arma::rowvec b( 2*dim );
	for ( arma::uword d = 0; d < dim; ++d )
	{
		b(d    ) = std::numeric_limits<double>::lowest();
		b(d+dim) = std::numeric_limits<double>::max();
	}

	std::vector<arma::uword> result_positions;
	query( X, 0, 0, X.n_rows, b, q, result_positions );

	arma::uvec result( result_positions.size() );
	for ( size_t i = 0; i < result_positions.size(); ++i )
	{
		result(i) = idx( result_positions[i] );
	}
	return result;
}

arma::mat kd_tree::covering_boxes( size_t min_per_box ) const
{
	if ( X.empty() )
	{
		throw std::logic_error { "vlasovius::geometry::kd_tree::covering_boxes: "
		                         "Cannot create meaningful over of nothing." };
	}

	const arma::uword n   { X.n_rows };
	const arma::uword dim { X.n_cols };
	arma::rowvec b( 2*dim );
	for ( arma::uword d = 0; d < dim; ++d )
	{
		b(d    ) = std::numeric_limits<double>::lowest();
		b(d+dim) = std::numeric_limits<double>::max();
	}

	// Except for the leaves, we have a complete binary tree.
	// Thus, we may simply output all boxes on a fixed level l,
	// as long as l is not the leaf-level. In the tree creation
	// routine we stop splitting below 8 points per box, marking
	// the beginning of our leaf-level.
	min_per_box = std::max( min_per_box, size_t(8) );

	size_t l;
	for ( l = 0; (n >> (l+1)) >= min_per_box; ++l ) ;

	arma::mat result( size_t(1) << l, 2*dim );
	traverse_cover(X,l,0,X.n_rows,0,0,b,result);
	return result;
}

}

}
