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


template <size_t k>
vlasovius::interpolators::pou_inducing_kernel<k>::pou_inducing_kernel( const arma::rowvec& sigma )
{
	size_t d = sigma.size();
	dim_kernels.reserve(d);

	for(size_t i = 0; i < d; i++)
	{
		dim_kernels.push_back(wendland{rbf_wendland {}, sigma(i) });
	}
}

template <size_t k>
arma::mat vlasovius::interpolators::pou_inducing_kernel<k>::operator()( const arma::mat &xv1, const arma::mat &xv2 ) const
{
	arma::mat m = dim_kernels[0]( xv1.col(0), xv2.col(0) ) %
		       dim_kernels[1]( xv1.col(1), xv2.col(1) );

	for(size_t i = 2; i < dim_kernels.size(); i++)
	{
		m = m % dim_kernels[i]( xv1.col(i), xv2.col(i) );
	}

	return m;
}

template <typename kernel>
vlasovius::interpolators::pou_interpolator<kernel>::pou_interpolator( kernel K, arma::mat X,
		arma::vec b, double tikhonov_mu, size_t min_per_box, size_t max_per_box,
		double enlargement_factor)
: K(K), tree(vlasovius::trees::kd_tree(X, b, min_per_box, max_per_box))
{
	if(enlargement_factor < 1.0) {
		throw std::runtime_error("Enlargement factor must be greater or equal 1.");
	}

	sub_sfx.reserve(tree.getNumberLeafs());
	weight_fcts.reserve(tree.getNumberLeafs());

	construct_sub_sfx(X, b, enlargement_factor, tikhonov_mu);
}

template <typename kernel>
void vlasovius::interpolators::pou_interpolator<kernel>::construct_sub_sfx(arma::mat X,
		arma::vec b, double enlargement_factor, double tikhonov_mu)
{
	vlasovius::misc::stopwatch clock;
	// Get the leaf-indices:
	// (Note that usually there are approximately twice as many nodes as leafs
	// so just running through all nodes and checking on leaf-status might have
	// the optimal run-time.)
	size_t n_nodes = tree.get_number_nodes();
	size_t n_leafs = tree.getNumberLeafs();

	indices_leafs = tree.get_indices_leafs();

	// Get the point-sets for each sub-sfx and compute the bounding-box. Note that
	// I have to enlarge the boxes of each leaf such that they slightly overlap.
	// To compute the enlarged boxes, use the tree-structure.
	std::vector<std::deque<arma::uword>> indices_points(n_leafs);
	sub_domains = std::vector<vlasovius::trees::bounding_box>(n_leafs);

	#pragma omp parallel for
	for(size_t i = 0; i < n_leafs; i++) // Can I pragma parallel for this?
	{
		size_t index_node = indices_leafs[i];
		vlasovius::trees::node leaf_nd = tree.getNode(index_node);

		//Init local point deque:
		indices_points[i] = std::deque<arma::uword>(leaf_nd.indexLastElem - leaf_nd.indexFirstElem);

		// Fill index-list with current inhabitants:
		size_t counter = 0;
		for(arma::uword j = leaf_nd.indexFirstElem; j < leaf_nd.indexLastElem; j++)
		{
			indices_points[i][counter] = j;
			counter++;
		}

		// Compute new bounding-box:
		sub_domains[i] = leaf_nd.box;
		sub_domains[i].sidelength *= enlargement_factor;

		// Find now all points intersecting the new bounding-box:
		// Therefore first find the parent-node which completely contains the sub-domain
		// and check child-points on intersection.
		int index_curr_parent = leaf_nd.parent;

		// While root (== 0) is not reached and the sub-domain is not a subset of the current
		// bounding box trace the tree to the top.
		while(!subset(sub_domains[i], tree.getNode(index_curr_parent).box))
		{
			if(index_curr_parent > 0){
				index_curr_parent = tree.getNode(index_curr_parent).parent;
			}else{
				index_curr_parent = 0;
				break;
			}
		}

		// Now check the contained points on intersection. Ignore the already known points (i.e.
		// the ones contained in the original leaf):
		vlasovius::trees::node parent_nd = tree.getNode(index_curr_parent);
		for(arma::uword j = parent_nd.indexFirstElem;
				(j < parent_nd.indexLastElem)
			&& !(leaf_nd.indexFirstElem <= j && j <= leaf_nd.indexLastElem);
				j++ )
		{
			if(sub_domains[i].contains(X.row(j)))
			{
				indices_points[i].push_back(j);
			}
		}

		// Finally compute the direct_interpolator's:
		arma::uword N_sub = indices_points[i].size();
		arma::mat sub_pts(N_sub, X.n_cols);
		arma::vec sub_rhs(N_sub);

		for(size_t j = 0; j < indices_points[i].size(); j++)
		{
			arma::uword curr = indices_points[i][j];
			sub_pts.row(j) = X.row(curr);
			sub_rhs(j) = b(curr);
		}

		direct_interpolator<kernel> sfx (K, sub_pts, sub_rhs, tikhonov_mu );
		pou_inducing_kernel<4>      w   (sub_domains[i].sidelength);

		// Writing to the shared arrays can only happen one at a time.
		#pragma omp critical
		{
			sub_sfx.push_back( std::move(sfx) );
			weight_fcts.push_back( std::move(w) );
		}
	}

	// For faster evaluation compute which sub-domains intersect which leafs:
	domains_intersect_leafs.resize(n_leafs);

	for(size_t i_leaf = 0; i_leaf < n_leafs; i_leaf++)
	{
		vlasovius::trees::bounding_box box_leaf = tree.getNode(indices_leafs[i_leaf]).box;
		for(size_t i_d = 0; i_d < n_leafs; i_d++)
		{
			if(vlasovius::trees::intersect(box_leaf, sub_domains[i_d]))
			{
				domains_intersect_leafs[i_leaf].push_back(i_d);
			}
		}
	}

	double elapsed { clock.elapsed() };
	std::cout << "Time for constructing pou after tree is build: " << elapsed << ".\n";
}

template <typename kernel>
arma::vec vlasovius::interpolators::pou_interpolator<kernel>::operator()( const arma::mat &Y ) const
{
	std::vector<int> w(Y.n_rows);
	size_t n_submat = indices_leafs.size() + 1;
	std::vector<size_t> sizes_sub_matrices(n_submat, 0);
	// Compute in which leaf each evaluation point lies and the needed
	// sub-matrix size for each leaf:
	for(size_t i = 0; i < w.size(); i++)
	{
		w[i] = tree.whichLeafContains(Y.row(i));
		if(w[i] < 0){
			sizes_sub_matrices[n_submat - 1]++;
			w[i] = n_submat - 1;
		} else {
			sizes_sub_matrices[w[i]]++;
		}
	}

	// Reserve space for the sub-matrices:
	std::vector<arma::mat> sub_matrix(n_submat);
	std::vector<std::vector<arma::uword>> orig_indices(n_submat);
	for(size_t i = 0; i < n_submat; i++)
	{
		sub_matrix[i] = arma::mat(sizes_sub_matrices[i], Y.n_cols);
		orig_indices[i].resize(sizes_sub_matrices[i]);
	}

	// Fill sub-matrices and remember the index of the point in
	// the original matrix to construct the correctly ordered
	// return vector:
	std::vector<size_t> curr_row_index(n_submat, 0);
	for(size_t i = 0; i < Y.n_rows; i++)
	{
		size_t index_submat = w[i];
		arma::uword row_index = curr_row_index[index_submat];
		sub_matrix[index_submat].row(row_index) = Y.row(i);
		orig_indices[index_submat][row_index] = i;
		curr_row_index[index_submat]++;
	}

	// Now evaluate for each sub-matrix:
	std::vector<arma::vec> sub_r(n_submat);
	sub_r[n_submat - 1] = arma::vec(sizes_sub_matrices[n_submat - 1], arma::fill::zeros);
	for(size_t i = 0; i < n_submat - 1; i++){
		// The formula is now:
		// f(x) = sum_{i in containingBoxes} f[i](x) * w[i](x) / (sum_{i in containingBoxes} w[i](x) )
		if(sizes_sub_matrices[i] > 0)
		{
			arma::vec denominator(sizes_sub_matrices[i], arma::fill::zeros);
			arma::vec nominator(sizes_sub_matrices[i], arma::fill::zeros);

			for(size_t j: domains_intersect_leafs[i])
			{
				nominator   += sub_sfx[j](sub_matrix[i]) % weight_fcts[j](sub_domains[j].center, sub_matrix[i]);
				denominator += weight_fcts[j](sub_domains[j].center, sub_matrix[i]);
			}

			sub_r[i] = nominator / denominator;
		}
	}


	// Construct return-vector:
	arma::vec r(Y.n_rows, arma::fill::zeros);

	for(size_t i_mat = 0; i_mat < orig_indices.size(); i_mat++)
	{
		for(size_t i_row = 0; i_row < orig_indices[i_mat].size(); i_row++)
		{
			r(orig_indices[i_mat][i_row]) = sub_r[i_mat](i_row);
		}
	}

	return r;
}
