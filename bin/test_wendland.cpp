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

#include <fstream>
#include <iostream>

#include <armadillo>
#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/kernels/wendland.h>

int main( int argc, char* argv[] )
{
	using vlasovius::misc::stopwatch;


    size_t N = 3000;
    constexpr size_t dim = 2, k = 4;
    vlasovius::kernels::wendland<dim,k> W;

    arma::vec x( N+1 ), v(N+1);
	for ( size_t i = 0; i <= N; ++i )
		x(i) = -1.5 + (3.0/N)*i;
	v = x;

	stopwatch clock;
	v = W( std::move(v) );
	double elapsed = clock.elapsed();
	std::cout << "Time for evaluating: " << elapsed << ".\n";

	std::ofstream file( "test_wendland.txt" );
	for ( size_t i = 0; i <= N; ++i )
		file << x(i) << " " << v(i) << std::endl;
	return 0;
}
