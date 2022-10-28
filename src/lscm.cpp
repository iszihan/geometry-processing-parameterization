#include "lscm.h"
#include "vector_area_matrix.h"
#include <igl/cotmatrix.h>
#include <igl/repdiag.h>
#include <igl/eigs.h>

void lscm(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & U)
{
  int nv = V.rows();

  //get cotangent laplacian matrix
  Eigen::SparseMatrix<double> L, M;
  igl::cotmatrix(V, F, L);
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, M);

  //get vector area A matrix
  Eigen::SparseMatrix<double> A;
  vector_area_matrix(F, A);

  //construct Q,B matrix
  Eigen::SparseMatrix<double> Q, B; 
  igl::repdiag(L, 2, Q);
  igl::repdiag(M, 2, B);
  Q = - Q / 2.0 - A;

  //generalized eigenvalue problem min u^TQu subject to u^TBu=1
  Eigen::MatrixXd sU;
  Eigen::VectorXd sS;
  igl::eigs(Q,B,3,igl::EIGS_TYPE_SM,sU,sS);

  //U = sU.col(0).reshape(nv, 2);
  U.resize(nv,2);
  U.col(0) = sU.col(0).topRows(nv);
  U.col(1) = sU.col(0).bottomRows(nv);

  //find canonical rotation
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(U.transpose()*U, Eigen::ComputeFullV|Eigen::ComputeFullU);
  U = U * svd.matrixV();
}
