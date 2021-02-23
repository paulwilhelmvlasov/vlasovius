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
	size_t N = 1e6;
	arma::mat points(N, 2, arma::fill::randu);
	arma::vec rhs(N, arma::fill::randu);
	/*
	points = {
	{0.2, 1}, {0.4, 0.6}, {0.3, 0.4}, {0.6, 0.7}, {0.8, 0.3}
	};

	std::cout << points << std::endl;
	 */

	vlasovius::misc::stopwatch watch;
	vlasovius::trees::kd_tree baum(points, rhs, 500, 1000);
	double zeit = watch.elapsed();
	std::cout << "Speedy speed: " << zeit << std::endl;
	std::cout << "Number nodes: " << baum.get_number_nodes() << std::endl;
	std::cout << "Number leafs: " << baum.getNumberLeafs() << std::endl;

	return 0;
}




