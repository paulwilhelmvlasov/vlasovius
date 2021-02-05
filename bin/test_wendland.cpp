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
    using vlasovius::kernels::scalar_wendland;
    using vlasovius::kernels::avx_wendland;
    using vlasovius::misc::stopwatch;

    size_t N = 1000;
    constexpr size_t dim = 2, k = 4;

    arma::vec xx( N+1 );
	for ( size_t i = 0; i <= N; ++i )
		xx(i) = -1.5 + (3.0/N)*i;


	stopwatch clock;
	arma::vec v1 = scalar_wendland<dim,k>(xx);
	double elapsed = clock.elapsed();
	std::cout << "Time for scalar version: " << elapsed << ".\n";


	clock.reset();
	arma::vec v2 = avx_wendland<dim,k>( xx );
	elapsed = clock.elapsed();
	std::cout << "Time for AVX accelerated version: " << elapsed << ".\n";

	std::cout << "Maximum difference between the two versions: " << norm( v1 - v2, "inf" ) << ".\n";

	std::ofstream file( "test_wendland.txt" );
	for ( size_t i = 0; i < N; ++i )
		file << xx(i) << " " << v1(i) << " " << v2(i) << std::endl;
	return 0;
}
