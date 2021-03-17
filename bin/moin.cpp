#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <armadillo>

#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/misc/xv_kernel.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/interpolators/pou_interpolator.h>
#include <vlasovius/interpolators/periodic_pou_interpolator.h>

int main() 
{

	constexpr size_t dim { 30 * 72 }, k { 4 };
	constexpr size_t N { 10'000 };
	constexpr double tikhonov_mu { 1e-17 };
	constexpr size_t min_per_box = 200;
	constexpr double enlarge = 1.2;


	using wendland_t     = vlasovius::kernels::wendland<dim,k>;
	using kernel_t       = vlasovius::kernels::rbf_kernel<wendland_t>;
	using interpolator_t = vlasovius::interpolators::pou_interpolator<kernel_t>;
	//using interpolator_t = vlasovius::interpolators::direct_interpolator<kernel_t>;

	arma::mat X( N, dim, arma::fill::randu );
	arma::vec f(N, arma::fill::randu);

	arma::rowvec bounding_box { 0, 0, 1, 1};
	kernel_t K { wendland_t {}, 1.0 };

	vlasovius::misc::stopwatch clock;
	interpolator_t sfx { K, X, f, bounding_box, enlarge, min_per_box, tikhonov_mu };
	//interpolator_t sfx { K, X, f, tikhonov_mu, 6 };
	double elapsed { clock.elapsed() };
	std::cout << "Time for computing RBF-Approximation: " << elapsed << ".\n";
	clock.reset();
	double error = norm(f-sfx(X),"inf");
	elapsed = clock.elapsed();
	std::cout << "Maximal interpolation error: " << error << ".\n";
	std::cout << "Time for evaluating interpolation error: " << elapsed << ".\n";

	return 0;
}
