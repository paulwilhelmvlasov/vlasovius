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
#include <vlasovius/interpolators/pou_interpolator.h>

#include <vlasovius/geometry/kd_tree.h>
#include <vlasovius/geometry/bounding_box.h>


#include <vlasovius/kernels/wendland.h>

namespace vlasovius
{

namespace interpolators
{

template <typename kernel>
periodic_pou_interpolator<kernel>::periodic_pou_interpolator( kernel K, const arma::mat &X, const arma::mat &f,
		                                    arma::rowvec bounding_box, double enlarge,
											size_t min_per_box, double tikhonov_mu, size_t threads ):
nrhs { f.n_cols }
{
	using ::vlasovius::geometry::bounding_box::intersection;
	size_t dim { X.n_cols };
	geometry::kd_tree tree(X);
	cover = tree.covering_boxes(min_per_box);

	// Written for only the first dimension beeing periodic. Change this if you want to
	// use it for higher dimensions also!
	// Also I assume now that the x-dimension is [0, L].
	L = bounding_box(dim);

	local_interpolants.resize(cover.n_rows);

	#pragma omp parallel for if (threads > 1), num_threads(threads)
	for ( size_t i = 0; i < cover.n_rows; ++i )
	{
		// First get the intersection points in [0, L] with the box.
		// Then compute the intersections in [-L, 0] and [L, 2L].
		// Add these to the direct interpolant.

		arma::rowvec bounds = intersection( bounding_box, cover.row(i) );
		for ( size_t d = 0; d < dim; ++d )
		{
			double l = (bounds(d+dim) - bounds(d))*enlarge/2.;
			double c = bounds(d) + (bounds(d+dim)-bounds(d))/2.;
			bounds(d    ) = c - l;
			bounds(d+dim) = c + l;
		}
		cover.row(i) = bounds;

		arma::uvec idx = tree.index_query(bounds);

		arma::rowvec left_bounds = bounds;
		left_bounds(0) 	 -= L;
		left_bounds(dim) -= L;
		arma::uvec left_idx = tree.index_query(left_bounds);
		arma::rowvec right_bounds = bounds;
		right_bounds(0)   += L;
		right_bounds(dim) += L;
		arma::uvec right_idx = tree.index_query(right_bounds);

		arma::uword n_idx = idx.n_rows;
		arma::uword n_left_idx = left_idx.n_rows;
		arma::uword n_right_idx = right_idx.n_rows;

		arma::mat pt_mat = arma::join_vert(X.rows(idx), X.rows(left_idx), X.rows(right_idx));
		if(n_left_idx > 0)
		{
			pt_mat.col(0).subvec(n_idx, n_idx + n_left_idx - 1)
					+= L * arma::vec(n_left_idx, arma::fill::ones);
		}
		if(n_right_idx > 0)
		{
			pt_mat.col(0).subvec(n_idx + n_left_idx, n_idx + n_left_idx  + n_right_idx - 1)
					-= L * arma::vec(n_right_idx, arma::fill::ones);
		}

		arma::mat rhs =  arma::join_vert(f.rows(idx), f.rows(left_idx), f.rows(right_idx));
		local_interpolants[i] = direct_interpolator<kernel>( K, pt_mat, rhs, tikhonov_mu );
	}
}

template <typename kernel>
arma::mat periodic_pou_interpolator<kernel>::operator()( const arma::mat &Y, size_t threads ) const
{
	static vlasovius::kernels::wendland<2,1> W;

	size_t dim { Y.n_cols };
	vlasovius::geometry::kd_tree tree { Y };

	arma::mat result   ( Y.n_rows, nrhs, arma::fill::zeros );
	arma::vec weightsum( Y.n_rows,       arma::fill::zeros );
	#pragma omp parallel if (threads > 1), num_threads(threads)
	{
		arma::mat my_result   ( Y.n_rows, nrhs, arma::fill::zeros );
		arma::vec my_weightsum( Y.n_rows,       arma::fill::zeros );
		arma::rowvec inv_sigma(dim), center(dim);
		arma::rowvec left_inv_sigma(dim), left_center(dim);
		arma::rowvec right_inv_sigma(dim), right_center(dim);

		#pragma omp for schedule(dynamic)
		for ( size_t i = 0; i < cover.n_rows; ++i )
		{
			arma::rowvec box    	= cover.row(i);
			arma::uvec   idx    	= tree.index_query( box );

			arma::rowvec left_box   = box;
			left_box(0)   -= L;
			left_box(dim) -= L;
			arma::uvec   left_idx   = tree.index_query( left_box );

			arma::rowvec right_box  = box;
			right_box(0)   += L;
			right_box(dim) += L;
			arma::uvec   right_idx  = tree.index_query( right_box );

			arma::uword n_idx = idx.n_rows;
			arma::uword n_left_idx = left_idx.n_rows;
			arma::uword n_right_idx = right_idx.n_rows;

			if ( n_idx == 0 && n_left_idx == 0 && n_right_idx == 0)
				continue;


			arma::mat Y_eval( Y.rows( arma::join_vert(idx, left_idx, right_idx) ) );
			if(left_idx.n_rows > 0)
			{
				Y_eval.col(0).subvec(n_idx, n_idx + n_left_idx - 1)
						+= L * arma::vec(n_left_idx, arma::fill::ones);
			}
			if(right_idx.n_rows > 0)
			{
				Y_eval.col(0).subvec(n_idx + n_left_idx, n_idx + n_left_idx  + n_right_idx - 1)
						-= L * arma::vec(n_right_idx, arma::fill::ones);
			}

			arma::mat values = local_interpolants[i]( Y_eval );

			for ( size_t d = 0; d < dim; ++d )
			{
				inv_sigma(d) = 1./((box(d+dim)-box(d))/2);
				   center(d) =     (box(d+dim)+box(d))/2;

				left_inv_sigma(d) = inv_sigma(d);
				   left_center(d) =     (left_box(d+dim)+left_box(d))/2;

				right_inv_sigma(d) = inv_sigma(d);
				   right_center(d) =     (right_box(d+dim)+right_box(d))/2;
			}

			arma::vec weights( n_idx + n_left_idx + n_right_idx );
			for ( size_t j = 0; j < idx.size(); ++j )
			{
				weights(j) = W( std::abs(Y(idx(j),0)-center(0))*inv_sigma(0) );
				for ( size_t d = 1; d < dim; ++d )
					weights(j) *= W( std::abs(Y(idx(j),d)-center(d))*inv_sigma(d) );
			}
			for( size_t j = n_idx; j < n_idx + n_left_idx; ++j)
			{
				weights(j) = W( std::abs(Y(left_idx(j - n_idx),0)-left_center(0))*left_inv_sigma(0) );
				for ( size_t d = 1; d < dim; ++d )
					weights(j) *= W( std::abs(Y(left_idx(j - n_idx),d)
							- left_center(d)) * left_inv_sigma(d) );
			}
			for( size_t j = n_idx + n_left_idx; j < n_idx + n_left_idx + n_right_idx; ++j)
			{
				weights(j) = W( std::abs(Y(right_idx(j - n_idx - n_left_idx),0)
						- right_center(0)) * right_inv_sigma(0) );
				for ( size_t d = 1; d < dim; ++d )
					weights(j) *= W( std::abs(Y(right_idx(j - n_idx - n_left_idx),d)
							- right_center(d)) * right_inv_sigma(d) );
			}

			my_result.rows(arma::join_vert(idx, left_idx, right_idx))    += weights % values;
			my_weightsum.rows(arma::join_vert(idx, left_idx, right_idx)) += weights;
		}

		#pragma omp critical
		{
			weightsum += my_weightsum;
			result    += my_result;
		}
	}

	for ( size_t i = 0; i < result.n_rows; ++i )
		result.row(i) *= 1./weightsum(i);

	return result;
}

}

}
