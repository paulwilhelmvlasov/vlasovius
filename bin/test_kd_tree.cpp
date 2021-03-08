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

#include <iostream>

#include <vlasovius/geometry/kd_tree.h>
#include <vlasovius/geometry/bounding_box.h>
#include <vlasovius/misc/stopwatch.h>

int main()
{
	size_t N = 1e6; size_t min_per_box = 400;
	arma::mat points(N, 2, arma::fill::randu);

	vlasovius::misc::stopwatch watch;
	vlasovius::geometry::kd_tree baum(points);
	double zeit { watch.elapsed() };
	std::cout << "Speedy speed: " << zeit << std::endl;

	watch.reset();
	arma::mat cover = baum.covering_boxes(min_per_box);
	zeit = watch.elapsed();
	std::cout << "Time for computing covering: " << zeit << std::endl;
	std::cout << "Size of covering: " << cover.n_rows << std::endl;

	double max_zeit = 0, total_zeit = 0;
	for ( size_t i = 0; i < cover.n_rows; ++i )
	{
		arma::rowvec box = cover.row(i);
		watch.reset();
		arma::uvec idx = baum.index_query( box );
		zeit = watch.elapsed();
		max_zeit = std::max(zeit,max_zeit);
		total_zeit += zeit;

		if ( idx.size() < min_per_box && ! (N > min_per_box) )
		{
			std::cout << "Error! Number of results in box is smaller than min_per_box!";
			return -2;
		}
		std::vector<arma::uword> myidx;
		for ( size_t j = 0; j < N; ++j )
		{
			arma::rowvec p = points.row(j);
			if ( vlasovius::geometry::bounding_box::contains_point(box,p) )
			{
				myidx.push_back(j);
			}
		}

		if ( myidx.size() != idx.size() )
		{
			std::cout << "Error! Results of kd_tree and exhaustive search differ.";
			return -1;
		}

		std::sort( myidx.begin(), myidx.end() );
		std::sort(   idx.begin(),   idx.end() );

		for ( size_t j = 0; j < myidx.size(); ++j )
		{
			if ( idx[j] != myidx[j] )
			{
				std::cout << "Error! Results of kd_tree and exhaustive search differ.";
				return -1;
			}
		}
	}
	std::cout << "Maximum query time: " << max_zeit << std::endl;
	std::cout << "Average query time: " << total_zeit/cover.n_rows << std::endl;

	return 0;
}




