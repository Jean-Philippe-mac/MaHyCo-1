#include <math.h>  // for floor, sqrt

#include <Kokkos_Core.hpp>  // for deep_copy
#include <algorithm>        // for copy
#include <array>            // for array
#include <iostream>         // for operator<<, basic_ostream::ope...
#include <vector>           // for allocator, vector

#include "Init.h"

#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "types/MathFunctions.h"   // for max, norm, dot
#include "types/MultiArray.h"      // for operator<<
#include "utils/Utils.h"           // for indexOf

namespace initlib {

void Initialisations::initVarSEDOV() noexcept {
  Kokkos::parallel_for(
      "initVarSEDOV", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        m_density_n0(cCells) = 1.0;
        m_density_env_n0(cCells)[0] = 1.0;
        m_cell_velocity_n0(cCells) = {0.0, 0.0};
	double dx = cstmesh->X_EDGE_LENGTH;
	double dy = cstmesh->Y_EDGE_LENGTH;
        int cId(cCells);
        bool isCenterCell = false;
        double pInit = 1.e-6;
        double rhoInit = 1.;
        double rmin = options->threshold;  // depot sur 1 maille
        // double rmin = std::sqrt(dx*dx+dy*dy) + options->threshold;  // depot sur 4 maille
        double e1 = pInit / ((eos->gamma[0] - 1.0) * rhoInit);
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pNodes(pId);
            if (MathFunctions::norm(m_node_coord_n0(pNodes)) < rmin)
              isCenterCell = true;
          }
        }
        if (isCenterCell) {
          //double total_energy_deposit = 0.244816 / (dx * dy);
	  //double total_energy_deposit = 0.244816 / (4 * dx * dy);
	  double total_energy_deposit = 5000.*4/Pi;
	  //double total_energy_deposit = 5000./Pi;
          m_internal_energy_n0(cCells) = e1 + total_energy_deposit;
        } else {
          m_internal_energy_n0(cCells) = e1;
        }
        m_internal_energy_env_n0(cCells)[0] = m_internal_energy_n0(cCells);

        m_pressure_env_n0(cCells)[0] = (eos->gamma[0] - 1.0) *
                                       m_density_env_n0(cCells)[0] *
                                       m_internal_energy_env_n0(cCells)[0];
        m_pressure_n0(cCells) = m_pressure_env_n0(cCells)[0];

        m_speed_velocity_env_n0(cCells)[0] =
            std::sqrt(eos->gamma[0] * m_density_env_n0(cCells)[0] /
                      m_pressure_env_n0(cCells)[0]);
        m_speed_velocity_n0(cCells) = m_speed_velocity_env_n0(cCells)[0];
      });
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    m_node_velocity_n0(pNodes) = {0.0, 0.0};
  });
}
void Initialisations::initVarBiSEDOV() noexcept {
  Kokkos::parallel_for(
      "initVarSEDOV", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        m_density_n0(cCells) = 1.0;
        m_density_env_n0(cCells)[0] = 1.0;
        m_cell_velocity_n0(cCells) = {0.0, 0.0};
        int cId(cCells);
        bool isCenterCell = false;
        double pInit = 1.e-6;
        double rhoInit = 1.;
        double rmin = options->threshold;  // depot sur 1 maille
        double e1 = pInit / ((eos->gamma[0] - 1.0) * rhoInit);
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pNodes(pId);
            if (MathFunctions::norm(m_node_coord_n0(pNodes)) < rmin)
              isCenterCell = true;
          }
        }
        if (isCenterCell) {
          double total_energy_deposit = 0.244816;
          double dx = cstmesh->X_EDGE_LENGTH;
          double dy = cstmesh->Y_EDGE_LENGTH;
          m_internal_energy_n0(cCells) = e1 + 5000.*4/Pi;
	    // + total_energy_deposit / (dx * dy);
        } else {
          m_internal_energy_n0(cCells) = e1;
        }
        m_internal_energy_env_n0(cCells)[0] = m_internal_energy_n0(cCells);
      });
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    m_node_velocity_n0(pNodes) = {0.0, 0.0};
  });
}

}  // namespace initlib
