#include "tutte.h"
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/min_quad_with_fixed.h>

void tutte(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & U)
{
    //get boundary vertices
    Eigen::VectorXi BL;
    Eigen::MatrixXd BV;
    igl::boundary_loop(F,BL);
    //map to a unit circle
    igl::map_vertices_to_circle(V, BL, BV);

    //construct L matrix with inverse edge length as weight
    Eigen::SparseMatrix<double> L;
    L.resize(V.rows(), V.rows());
    std::vector<Eigen::Triplet<double>> tripletList;
    for(int i=0; i<F.rows(); ++i){
        int v0 = F.row(i)(0);
        int v1 = F.row(i)(1);
        int v2 = F.row(i)(2);
        double e0l = (V.row(v1)-V.row(v0)).norm();
        double e1l = (V.row(v2)-V.row(v1)).norm();
        double e2l = (V.row(v0)-V.row(v2)).norm();

        //e1 term
        tripletList.emplace_back(v0, v1, 1.0/e0l);
        tripletList.emplace_back(v1, v0, 1.0/e0l);
        //e2 term
        tripletList.emplace_back(v1, v2, 1.0/e1l);
        tripletList.emplace_back(v2, v1, 1.0/e1l);
        //e3 term
        tripletList.emplace_back(v2, v0, 1.0/e2l);
        tripletList.emplace_back(v0, v2, 1.0/e2l);
        //v1 term
        tripletList.emplace_back(v0, v0, -1.0/e0l-1.0/e2l);
        //v2 term
        tripletList.emplace_back(v1, v1, -1.0/e0l-1.0/e1l);
        //v3 term
        tripletList.emplace_back(v2, v2, -1.0/e1l-1.0/e2l);
    }
    L.setFromTriplets(tripletList.begin(), tripletList.end());

    //solve for min u^TLu with u_bl = bv
    U = Eigen::MatrixXd::Zero(V.rows(), 2);
    igl::min_quad_with_fixed(L, Eigen::VectorXd::Zero(V.rows()),
                             BL, BV,
                             Eigen::SparseMatrix<double>(),
                             Eigen::MatrixXd(V.rows(), 2),
                             false, U); 
}

 