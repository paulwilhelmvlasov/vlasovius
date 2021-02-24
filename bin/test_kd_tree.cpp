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

#include <vlasovius/trees/kd_tree.h>
#include <vlasovius/misc/stopwatch.h>

int main()
{
	size_t N = 1e2;
	arma::mat points(N, 2, arma::fill::randu);
	arma::vec rhs(N, arma::fill::randu);

	std::cout << points << std::endl;

	vlasovius::misc::stopwatch watch;
	vlasovius::trees::kd_tree baum(points, rhs, 10, 20);
	double zeit = watch.elapsed();
	std::cout << "Speedy speed: " << zeit << std::endl;
	std::cout << "Number nodes: " << baum.get_number_nodes() << std::endl;
	std::cout << "Number leafs: " << baum.getNumberLeafs() << std::endl;

	arma::mat test_points(N, 2, arma::fill::randu);
	test_points *= 1.1;

	for(size_t i = 0; i < N; i++)
	{
		int j = baum.whichLeafContains(test_points.row(i));
		std::cout << "Leaf " << j << " contains " << test_points.row(i) << std::endl;
	}

	for(size_t i = 0; i < baum.getNumberLeafs(); i++)
	{
		std::cout << "Leaf " << i << std::endl;
		std::cout << baum.getLeaf(i).box.center << std::endl;
		std::cout << baum.getLeaf(i).box.sidelength << std::endl;
	}

	return 0;
}




