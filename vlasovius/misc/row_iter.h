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

#include <iterator>

#include <armadillo>

namespace vlasovius
{

	namespace misc
	{
		class row_iter : public std::iterator
		<std::random_access_iterator_tag, arma::vec>
		{
		public:
			row_iter(arma::mat &A, arma::uword i)
			: i(i), A(A) {}

		public:

			arma::uword getIndex() const
			{
				return i;
			}

			// Pointer-operators:
			auto operator*() {
				return A.row(i);
			}

			auto operator->() {
				return A.row(i);
			}

			// Increment/Decrement operators:
			row_iter operator++(int) {
				i++;
				return *this;
			}

			row_iter& operator++(){
				i++;
				return *this;
			}

			row_iter operator--(int) {
				i--;
				return *this;
			}

			row_iter& operator--(){
				i--;
				return *this;
			}

			// Addition/Subtraction operators:
			row_iter operator+=(const row_iter& rhs) {
				i += rhs.i;
				return *this;
			}

			row_iter operator-=(const row_iter& rhs) {
				i -= rhs.i;
				return *this;
			}

			row_iter operator+(const row_iter& right){
				i += right.i;
				return *this;
			}

			row_iter operator-(const row_iter& right){
				i -= right.i;
				return *this;
			}

			row_iter operator+(arma::uword n) {
				i += n;
				return *this;
			}

			row_iter operator-(arma::uword n) {
				i -= n;
				return *this;
			}

			// Direct access:
			auto operator[](arma::uword j){
				return A.row(j);
			}

			// Comparison-operators:
			bool operator<(const row_iter& rhs){
				return (i < rhs.i);
			}

			bool operator>(const row_iter& rhs){
				return rhs.i < i;
			}

			bool operator<=(const row_iter& rhs){
				return !(i > rhs.i);
			}

			bool operator>=(const row_iter& rhs){
				return !(i < rhs.i);
			}


			bool operator== (const row_iter& iter) {
				return (i == iter.i);
			}

			bool operator!= (const row_iter& iter) {
				return !(*this == iter);
			}


		private:
			arma::uword i;
			arma::mat &A;
		};

	}

}



#endif /* VLASOVIUS_MISC_ROW_ITER_H_ */
