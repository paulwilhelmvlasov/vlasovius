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


#ifndef VLASOVIUS_TREES_KD_TREE_H
#define VLASOVIUS_TREES_KD_TREE_H

#include <algorithm>
#include <deque>
#include <iostream>
#include <map>

#include <armadillo>

#include <vlasovius/misc/stopwatch.h>

namespace vlasovius
{
	namespace trees
	{
		struct bounding_box
		{
			arma::rowvec center;
			arma::rowvec sidelength; // Distance from center to border
			// in each direction.

			bool contains(const arma::rowvec& p) const;
		};

		bool subset(const bounding_box& first, const bounding_box& second);
		bool intersect(const bounding_box& first, const bounding_box& second);

		struct node
		{
			int parent 		= -1;

			int firstChild  = -1;
			int secondChild = -1;

			arma::uword indexFirstElem;
			arma::uword indexLastElem;
			// This points to the index after the last element!

			bounding_box box;

			bool isLeaf() const;
		};

		bounding_box computeBox(arma::mat& points);

		class kd_tree
		{
		public:
			kd_tree(arma::mat& points, arma::vec& rhs, size_t minPerBox, size_t maxPerBox);

		public:
			size_t getNumberLeafs() const;
			size_t get_number_nodes() const;
			node getNode(size_t i) const;
			node getLeaf(size_t i) const;
			int whichLeafContains(const arma::rowvec& p) const;
			int whichLeafContains(size_t i) const;

			std::vector<size_t> get_indices_leafs() const;

		private:

			// Sorting after build:
			void sortPoints(std::vector<arma::uword>& sortedIndices,
					arma::mat& points, arma::vec& rhs);

			// Tree-build methods:
			void buildTree(std::vector<arma::uword>& sortedIndices,
					arma::mat& points, arma::vec& rhs, size_t currentNodeIndex,
					size_t minPerBox, size_t maxPerBox);
			size_t splittingDimension(size_t currentNodeIndex);
			void split(std::vector<arma::uword>& sortedIndices,
					arma::mat& points, arma::vec& rhs, size_t currentNodeIndex,
					size_t dimSplit);

		private:
			arma::uword dim = 0;
			size_t n_leafs = 0;

			std::deque<node> nodes; // deque to avoid reallocating the underlying array several times
			// as the struct node might potentially be a large datatype.
			std::vector<size_t> indices_leafs;
			std::map<size_t, size_t> node_index_leaf_index;
		};

	}
}

#endif /* VLASOVIUS_TREES_KD_TREE_H_ */
