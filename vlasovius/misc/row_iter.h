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

#ifndef VLASOVIUS_MISC_ROW_ITER_H_
#define VLASOVIUS_MISC_ROW_ITER_H_

#include <armadillo>

namespace vlasovius
{

	namespace misc
	{
		class row_iter
		{
		public:
			row_iter(arma::mat &A, arma::mat uword) : i(0), A(A) {}

		public:
			auto operator*() {
				return A.row(i);
			}

			auto operator->() {
				return A.row(i);
			}

			row_iter operator++() {
				i++;
				return *this;
			}

			row_iter operator--() {
				i--;
				return *this;
			}

			row_iter operator+=(const row_iter& rhs) {
				i += rhs.i;
				return *this;
			}

			row_iter operator-=(const row_iter& rhs) {
				i -= rhs.i;
				return *this;
			}

			row_iter operator+(row_iter left, const row_iter& right){
				left += right;
				return left;
			}

			row_iter operator-(row_iter left, const row_iter& right){
							left -= right;
							return left;
			}

			auto operator[](arma::uword j){
				return A.row(j);
			}

			bool operator<(const row_iter& lhs, const row_iter& rhs){
				return (lhs.i < rhs.i);
			}

			bool operator>(const row_iter& lhs, const row_iter& rhs){
				return rhs < lhs;
			}

			bool operator<=(const row_iter& lhs, const row_iter& rhs){
				return !(lhs > rhs);
			}

			bool operator>=(const row_iter& lhs, const row_iter& rhs){
				return !(lhs < rhs);
			}

			row_iter operator+(row_iter it, arma::uword n) {
				it.i += n;
				return it;
			}

			row_iter operator-(row_iter it, arma::uword n) {
				it.i -= n;
				return it;
			}

			row_iter operator+(arma::uword n, row_iter it) {
				it.i += n;
				return it;
			}

			row_iter operator-(arma::uword n, row_iter it) {
				it.i -= n;
				return it;
			}



		private:
			arma::uword i;
			arma::mat &A;
		};

	}

}



#endif /* VLASOVIUS_MISC_ROW_ITER_H_ */
