#include "vector_area_matrix.h"
#include <igl/boundary_loop.h>

void vector_area_matrix(
  const Eigen::MatrixXi & F,
  Eigen::SparseMatrix<double>& A)
{
  //Get boundary loops
  std::vector<std::vector<int>> BL;
  igl::boundary_loop(F,BL);

  //Construct A 
  int nv = F.maxCoeff()+1;
  //Eigen::SparseMatrix<double> A_tilda;
  //A_tilda.resize(nv*2, nv*2);
  A.resize(nv*2, nv*2);
  std::vector<Eigen::Triplet<double>> tripletList;
  for(int il=0; il<BL.size(); ++il){
    std::vector<int> iL = BL.at(il);
    for(int iv=0; iv<iL.size(); ++iv){
      int cur = iL.at(iv);
      int next = iL.at((iv + 1 ) % iL.size());
      //add the normal term, times 1/2
      tripletList.emplace_back(cur, nv + next, -0.25);
      tripletList.emplace_back(next, nv + cur,  0.25);
      //add the transpose term, times 1/2
      tripletList.emplace_back(nv + next, cur, -0.25);
      tripletList.emplace_back(nv + cur, next,  0.25);
    }
  }
  A.setFromTriplets(tripletList.begin(), tripletList.end());
}

