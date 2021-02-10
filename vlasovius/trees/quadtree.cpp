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

#include "quadtree.h"

bool vlasovius::trees::bounding_box_2d::pointInsideAABB(const arma::vec2& x, double tol)
{
	return norm(x - center, 2) < tol;
}

void vlasovius::trees::quadtree::buildTree(arma::mat& points, 
	size_t minElemPerBox, size_t maxElemPerBox,
	node& currNode, size_t currIndex)
{
	if (currNode.childIndices.size() > maxElem) { 
		// topLeft:
		nodes.push_back(node({ currIndex, 0, 0, 0, 0, {},
			bounding_box_2d({currNode.box.center
				+ 0.5 * currNode.box.radius * arma::vec({-1, 1})}) })); 
		size_t topLeft = nodes.size() - 1;
		currNode.topLeft = topLeft;
		// topRight:
		nodes.push_back(node({ currIndex, 0, 0, 0, 0, {},
			bounding_box_2d({currNode.box.center
				+ 0.5 * currNode.box.radius * arma::vec({1, 1})}) })); 
		size_t topRight = nodes.size() - 1;
		currNode.topRight = topRight;
		// botRight:
		nodes.push_back(node({ currIndex, 0, 0, 0, 0, {},
			bounding_box_2d({currNode.box.center
				+ 0.5 * currNode.box.radius * arma::vec({1, -1})}) })); // topLeft
		size_t botRight = nodes.size() - 1;
		currNode.botRight = botRight;
		// botLeft:
		nodes.push_back(node({ currIndex, 0, 0, 0, 0, {},
			bounding_box_2d({currNode.box.center
				+ 0.5 * currNode.box.radius * arma::vec({-1, -1})}) })); // topLeft
		size_t botLeft = nodes.size() - 1;
		currNode.botLeft = botLeft;

		for (size_t i = 0; i < currNode.childIndices.size(); i++) {
			size_t indexPoint = currNode.childIndices[i];
			if (nodes[topLeft].box.pointInsideAABB(points.col(indexPoint))) {
				nodes[topLeft].childIndices.push_back(indexPoint);
			} else if (nodes[topRight].box.pointInsideAABB(points.col(indexPoint))) {
				nodes[topRight].childIndices.push_back(indexPoint);
			} else if (nodes[botRight].box.pointInsideAABB(points.col(indexPoint))) {
				nodes[botRight].childIndices.push_back(indexPoint);
			} else {
				nodes[botLeft].childIndices.push_back(indexPoint);

				// Keep in mind issues with floating point arithmetic!
			}
		}

		buildTree(points, minElemPerBox, maxElemPerBox,
			nodes[topLeft], topLeft); 
		buildTree(points, minElemPerBox, maxElemPerBox,
				nodes[topRight], topRight);
		buildTree(points, minElemPerBox, maxElemPerBox,
			nodes[botRight], botRight);
		buildTree(points, minElemPerBox, maxElemPerBox,
			nodes[botLeft], botLeft);
	}
}
