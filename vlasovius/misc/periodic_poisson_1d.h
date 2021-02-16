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
#ifndef VLASOVIUS_MISC_PERIODIC_POISSON_1D_H
#define VLASOVIUS_MISC_PERIODIC_POISSON_1D_H

#include <array>
#include <armadillo>

namespace vlasovius
{

namespace misc
{

namespace poisson_gedoens
{

template <int order>
class periodic_poisson_1d
{
public:
    // N: Number of cells.
    periodic_poisson_1d( double a, double b, int N );

    arma::vec quadrature_nodes();
    void update_rho( const arma::vec &rho_at_quadrature_nodes );

    double E( double x );

private:
    int                    periodic_number( int idx );
    std::array<int,order> cell_dof_numbers( int cellno );

private:
    double a, b;
    int    N;

    arma::mat A;
    arma::vec coeffs;
};



namespace splines
{

template <size_t order>
constexpr size_t num() noexcept { return order; }

template <size_t order>
using     vals = std::array<double,num<order>()>;

template <size_t order, typename T>
using   coeffs = std::array<T,num<order>()>;

template <size_t order> inline
vals<order> N( double x ) noexcept;

template <size_t order, size_t derivative = 1>
vals<order> dN( double x ) noexcept;

template <size_t order>
void Nimpl( double x, double *result )
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    constexpr int m { order };

    result[m-1]=1;
    for ( int k = 1; k < m; ++k )
    {
        for ( int i = 0; i < m-1; ++i )
        {
            double knot = i - (m-1);
            result[i] = ((x-knot)*result[i] + (knot+k+1-x)*result[i+1])/k;
        }
        result[m-1] = (x/k)*result[m-1];
    }
}

template <size_t order, size_t derivative>
struct dN_helper
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    static void dN( double x, double *data ) noexcept
    {
        if ( derivative >= order ) return;

        dN_helper<order-1,derivative-1>::dN(x,data);

        data[order-1] = data[order-2];
        for ( size_t i = order - 2; i > 0; --i )
            data[i] = data[i-1] - data[i];
        data[0] = -data[0];
    }
};

template <size_t order>
struct dN_helper<order,0>
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    static void dN( double x, double *data ) noexcept
    {
        Nimpl<order>(x,data);
    }
};


template <size_t order> inline
vals<order> N( double x ) noexcept
{
    vals<order> result {};
    Nimpl<order>(x,result.data());
    return result;
}

template <size_t order, size_t derivative = 1> inline
vals<order> dN( double x ) noexcept
{
    vals<order> result {};
    dN_helper<order,derivative>::dN(x,result.data());
    return result;
}


}


struct quadnode { double x, w; };

template <int order>
std::array<quadnode,order> get_quad_rule();

template <> inline
std::array<quadnode,1> get_quad_rule<1>()
{
    return
    {{
        quadnode { 0.5, 1 }
    }};
}

template <> inline
std::array<quadnode,2> get_quad_rule<2>()
{
    return
    {{
        quadnode { 0.21132486540518711774542560974902, 0.5 },
        quadnode { 0.78867513459481288225457439025098, 0.5 }
    }};
}

template <> inline
std::array<quadnode,3> get_quad_rule<3>()
{
    return
    {{
        quadnode { 0.5,0.44444444444444444444444444444444 },
        quadnode { 0.11270166537925831148207346002176, 0.27777777777777777777777777777778 },
        quadnode { 0.88729833462074168851792653997824, 0.27777777777777777777777777777778 }
    }};
}

template <> inline 
std::array<quadnode,4> get_quad_rule<4>()
{
    return
    {{
        quadnode { 0.33000947820757186759866712044838,  0.326072577431273071313468025389 },
        quadnode { 0.66999052179242813240133287955162,  0.326072577431273071313468025389 },
        quadnode { 0.069431844202973712388026755553595, 0.173927422568726928686531974611 },
        quadnode { 0.9305681557970262876119732444464,   0.173927422568726928686531974611 }
    }};
}

template <> inline
std::array<quadnode,5> get_quad_rule<5>()
{
    return
    {{
        quadnode { 0.5,0.28444444444444444444444444444444 },
        quadnode { 0.2307653449471584544818427896499,   0.23931433524968323402064575741782 },
        quadnode { 0.7692346550528415455181572103501,   0.23931433524968323402064575741782 },
        quadnode { 0.046910077030668003601186560850304, 0.11846344252809454375713202035996 },
        quadnode { 0.9530899229693319963988134391497,   0.11846344252809454375713202035996 }
    }};
}

template <> inline
std::array<quadnode,6> get_quad_rule<6>()
{
    return
    {{
        quadnode { 0.83060469323313225683069979750995,  0.18038078652406930378491675691886  },
        quadnode { 0.16939530676686774316930020249005,  0.18038078652406930378491675691886  },
        quadnode { 0.38069040695840154568474913915964,  0.23395696728634552369493517199478  },
        quadnode { 0.61930959304159845431525086084036,  0.23395696728634552369493517199478  },
        quadnode { 0.033765242898423986093849222753003, 0.085662246189585172520148071086366 },
        quadnode { 0.966234757101576013906150777247,    0.085662246189585172520148071086366 }
    }};
}

template <> inline
std::array<quadnode,7> get_quad_rule<7>()
{
    return
    {{
        quadnode { 0.5, 0.2089795918367346938775510204081 },
        quadnode { 0.70292257568869858345330320603848,  0.19091502525255947247518488774449  },
        quadnode { 0.29707742431130141654669679396152,  0.19091502525255947247518488774449  },
        quadnode { 0.12923440720030278006806761335961,  0.13985269574463833395073388571189  },
        quadnode { 0.87076559279969721993193238664039,  0.13985269574463833395073388571189  },
        quadnode { 0.025446043828620737736905157976074, 0.064742483084434846635305716339541 },
        quadnode { 0.97455395617137926226309484202393,  0.064742483084434846635305716339541 }
    }};
}

template <> inline
std::array<quadnode,8> get_quad_rule<8>()
{
    return
    {{
        quadnode { 0.40828267875217509753026192881991,  0.1813418916891809914825752246386   },
        quadnode { 0.59171732124782490246973807118009,  0.1813418916891809914825752246386   },
        quadnode { 0.23723379504183550709113047540538,  0.1568533229389436436689811009933   },
        quadnode { 0.76276620495816449290886952459462,  0.1568533229389436436689811009933   },
        quadnode { 0.10166676129318663020422303176208,  0.11119051722668723527217799721312  },
        quadnode { 0.89833323870681336979577696823792,  0.11119051722668723527217799721312  },
        quadnode { 0.019855071751231884158219565715264, 0.050614268145188129576265677154981 },
        quadnode { 0.98014492824876811584178043428474,  0.050614268145188129576265677154981 }
    }};
}


template <> inline
std::array<quadnode,9> get_quad_rule<9>()
{
    return
    {{
        quadnode { 0.5,0.16511967750062988158226253464349 },
        quadnode { 0.081984446336682102850285105965133,0.090324080347428702029236015621456 },
        quadnode { 0.91801555366331789714971489403487,0.090324080347428702029236015621456  },
        quadnode { 0.015919880246186955082211898548164,0.040637194180787205985946079055262 },
        quadnode { 0.98408011975381304491778810145184,0.040637194180787205985946079055262  },
        quadnode { 0.33787328829809553548073099267833,0.15617353852000142003431520329222   },
        quadnode { 0.66212671170190446451926900732167,0.15617353852000142003431520329222   },
        quadnode { 0.19331428364970480134564898032926,0.13030534820146773115937143470932   },
        quadnode { 0.80668571635029519865435101967074,0.13030534820146773115937143470932   }
    }};
}

template <> inline
std::array<quadnode,10> get_quad_rule<10>()
{
    return
    {{
        quadnode { 0.42556283050918439455758699943514,0.14776211235737643508694649732567   },
        quadnode { 0.57443716949081560544241300056486,0.14776211235737643508694649732567   },
        quadnode { 0.28330230293537640460036702841711,0.13463335965499817754561346078473   },
        quadnode { 0.71669769706462359539963297158289,0.13463335965499817754561346078473   },
        quadnode { 0.16029521585048779688283631744256,0.10954318125799102199776746711408   },
        quadnode { 0.83970478414951220311716368255744,0.10954318125799102199776746711408   },
        quadnode { 0.067468316655507744633951655788253,0.074725674575290296572888169828849 },
        quadnode { 0.93253168334449225536604834421175,0.074725674575290296572888169828849  },
        quadnode { 0.013046735741414139961017993957774,0.033335672154344068796784404946666 },
        quadnode { 0.98695326425858586003898200604223,0.033335672154344068796784404946666  }
    }};
}

template <int order>
periodic_poisson_1d<order>::periodic_poisson_1d(double pa, double pb, int pN) :
a { pa }, b { pb }, N { pN },
A      ( N+1, N+1, arma::fill::zeros ),
coeffs ( N+1 )
{
     double h = (b-a)/N;
     arma::mat::fixed<order,order> elemA { arma::fill::zeros };
     for ( size_t i = 0; i < order*order; ++i ) elemA[i] = 0;

     // Compute stiffness matrix for a single element.
     auto rule = get_quad_rule<order>();
     for ( int k = 0; k < order; ++k )
     {
         double x = rule[k].x;
         double w = rule[k].w*h;
         auto dN = splines::dN<order,1>(x);
         for ( int i = 0; i < order; ++i )
         for ( int j = 0; j < order; ++j )
         {
             elemA(i,j) += w*(dN[i]/h)*(dN[j]/h);
         }
     }

     // Assemble global stiffness matrix.
     for ( int k = 0; k < N; ++k )
     {
         std::array<int,order> nums = cell_dof_numbers(k);

         for ( int i = 0; i < order; ++i )
         for ( int j = 0; j < order; ++j )
         {
             A( nums[i], nums[j] ) += elemA[ i + order*j ];
         }
     }

     // Last row and column are for the Lagrange Multiplicator, that
     // enforces zero average of solution. Without it the system is singular.
     for ( int i = 0; i < N; ++i )
     {
         A(i, N) = h;
         A(N, i) = h;
     }

     A = inv(A);
}

template <int order> inline
int periodic_poisson_1d<order>::periodic_number( int idx )
{
    if ( idx < 0 ) return idx - ((idx/N)-1)*N;
    else return idx % N;
}

template <int order>
std::array<int,order> periodic_poisson_1d<order>::cell_dof_numbers( int cellno )
{
    std::array<int,order> result;
    for ( int k = 0; k < order; ++k )
    {
        result[k] = periodic_number( cellno - (order-1) + k );
    }
    return result;
}

template <int order>
arma::vec periodic_poisson_1d<order>::quadrature_nodes()
{
    double h = (b-a)/N;

    arma::vec result(N*order);
    auto rule = get_quad_rule<order>();
    for ( int i = 0; i < N; ++i )
    {
        for ( size_t k = 0; k < order; ++k )
        {
            result[ order*i + k ] = a + (i + rule[k].x)*h;
        }
    }

    return result;
}

template <int order>
void periodic_poisson_1d<order>::update_rho( const arma::vec &rho_at_quadrature_nodes )
{
    double h = (b-a)/N;
    arma::vec::fixed<order> elemb;

    coeffs.zeros();
    auto rule = get_quad_rule<order>();
    for ( int cell = 0; cell < N; ++cell )
    {
    	elemb.zeros();

        for ( int quad = 0; quad < order; ++quad )
        {
            double x   = rule[quad].x;
            double w   = rule[quad].w*h;
            double rho = rho_at_quadrature_nodes[ cell*order + quad ];

            auto N = splines::N<order>(x);
            for ( int i = 0; i < order; ++i )
                elemb[i] += w*N[i]*rho;
        }

        auto nums = cell_dof_numbers(cell);
        for ( int i = 0; i < order; ++i )
            coeffs(nums[i]) += elemb[i];
    }

    coeffs = A * coeffs;
}

template <int order>
double periodic_poisson_1d<order>::E( double x )
{
    x -= a;
    double h = (b-a)/N;
    int cell = std::floor(x/h);
    double xref = (x - cell*h)/h;

    double result = 0;
    auto nums = cell_dof_numbers(periodic_number(cell));
    auto vals = splines::dN<order>(xref);
    for ( int i = 0; i < order; ++i )
        result += coeffs(nums[i])*vals[i]/h;

    // MINUS Grad phi.
    return -result;
}

}

}

}

#endif
