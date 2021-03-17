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
#include <vlasovius/interpolators/piecewise_interpolator.h>

#include <vlasovius/geometry/kd_tree.h>

namespace vlasovius
{

namespace interpolators
{

template <typename kernel>
piecewise_interpolator<kernel>::piecewise_interpolator( kernel K, const arma::mat &X, const arma::mat &f,
											            size_t min_per_box, double tikhonov_mu, size_t threads ):
nrhs { f.n_cols }
{
	geometry::kd_tree tree(X);
	cover = tree.covering_boxes(min_per_box);
	local_interpolants.resize(cover.n_rows);

	#pragma omp parallel for schedule(dynamic) if(threads>1), num_threads(threads)
	for ( size_t i = 0; i < cover.n_rows; ++i )
	{
		arma::uvec idx = tree.index_query(cover.row(i));
		local_interpolants[i] = direct_interpolator<kernel>( K, X.rows(idx), f.rows(idx), tikhonov_mu );
	}
}

template <typename kernel>
arma::mat piecewise_interpolator<kernel>::operator()( const arma::mat &Y, size_t threads ) const
{
	vlasovius::geometry::kd_tree tree { Y };

	arma::mat result( Y.n_rows, nrhs, arma::fill::zeros );
	arma::vec count ( Y.n_rows, arma::fill::zeros );
	#pragma omp parallel if(threads>1),num_threads(threads)
	{
		arma::mat my_result( Y.n_rows, nrhs, arma::fill::zeros );
		arma::vec my_count ( Y.n_rows, nrhs, arma::fill::zeros );

		#pragma omp for schedule(dynamic)
		for ( size_t i = 0; i < cover.n_rows; ++i )
		{
			arma::uvec idx { tree.index_query( cover.row(i) ) };
			if ( idx.size() == 0 )
				continue;

			arma::mat values = local_interpolants[i]( Y.rows(idx) );

			my_result.rows(idx) += values;
			my_count .rows(idx) += 1.0;
		}

		#pragma omp critical
		{
			result += my_result;
			count  += my_count;
		}
	}

	// Points lying exactly on the boundaries of a box might
	// get added two twice. So we need to divide by the number
	// additions.
	for ( size_t i = 0; i < result.n_rows; ++i )
		result.row(i) *= 1.0/count(i);

	return result;
}

}

}
