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

#include "directInterpolator.h"

template<typename kernel>
vlasovius::interpolators::directInterpolator<kernel>::directInterpolator
(const std::vector<point>& X, const arma::vec& f) : interPolPoints(X)
{
	if (X.empty()) {
		throw std::runtime_error("No interpolation-points passed.");
	}

	if (X.size() != f.size()) {
		throw std::runtime_error("Number points does not match value vector length.");
	}

	arma::mat rkhsMat = computeRKHSMatrix(X);
	computeCoeffVec(rkhsMat, f);
}

template<typename kernel>
double vlasovius::interpolators::directInterpolator<kernel>::operator()(const point& x) const
{
	arma::vec v(N);

	#pragma omp parallel for
	for (long i = 0; i < N; i++) {
		v(i) = kernel(interPolPoints[i], x);
	}

	return v.dot(coeff);
}

template<typename kernel>
arma::mat vlasovius::interpolators::directInterpolator<kernel>::computeRKHSMatrix
(const std::vector<point>& X)
{
	N = X.size();
	arma::mat matrix(N, N);

	double norm = kernel(X[0], X[0]);

	#pragma omp parallel for
	for (long i = 0; i < N; i++) {
		matrix(i, i) = norm;
		for (long j = i + 1; j < N; j++) {
			matrix(i, j) = kernel(X[i], X[j]);
			matrix(j, i) = matrix(i, j);
		}
	}

	return matrix;
}

template<typename kernel>
void vlasovius::interpolators::directInterpolator<kernel>::computeCoeffVec
(const arma::mat& rkhsMat, const arma::vec& f)
{
	coeff = solve(rkhsMat, f); 
}

