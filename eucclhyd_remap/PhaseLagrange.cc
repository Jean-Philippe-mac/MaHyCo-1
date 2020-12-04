#include <stdlib.h>  // for exit

#include <Kokkos_Core.hpp>  // for KOKKOS_LAMBDA
#include <algorithm>        // for equal, copy
#include <array>            // for array, operator!=
#include <iostream>         // for operator<<, basic_ostream::ope...
#include <limits>           // for numeric_limits
#include <thread>
#include <vector>  // for vector, allocator

#include "../includes/VariablesLagRemap.h"
#include "Eucclhyd.h"              // for Eucclhyd, Eucclhyd::...
#include "Utiles-Impl.h"           // for Eucclhyd::tensProduct
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "types/MathFunctions.h"   // for max, min, dot, matVectProduct
#include "types/MultiArray.h"      // for operator<<
#include "utils/Utils.h"           // for indexOf
using namespace variableslagremaplib;

/**
 * Job computeCornerNormal called @1.0 in simulate method.
 * In variables: m_node_coord
 * Out variables: m_lminus, m_lpc, m_lplus, m_nminus, m_nplus
 */
void Eucclhyd::computeCornerNormal() noexcept {
  Kokkos::parallel_for(
      "computeCornerNormal", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        size_t cId(cCells);
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pMinus1Id(nodesOfCellC[(pNodesOfCellC - 1 + nbNodesOfCell) %
                                       nbNodesOfCell]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
            int pMinus1Nodes(pMinus1Id);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            RealArray1D<dim> xp = m_node_coord(pNodes);
            RealArray1D<dim> xpPlus = m_node_coord(pPlus1Nodes);
            RealArray1D<dim> xpMinus = m_node_coord(pMinus1Nodes);
            RealArray1D<dim> npc_plus;
            npc_plus[0] = 0.5 * (xpPlus[1] - xp[1]);
            npc_plus[1] = 0.5 * (xp[0] - xpPlus[0]);
            double lpc_plus = MathFunctions::norm(npc_plus);
            npc_plus = npc_plus / lpc_plus;
            m_nplus(pNodes, cCellsOfNodeP) = npc_plus;
            m_lplus(pNodes, cCellsOfNodeP) = lpc_plus;
            RealArray1D<dim> npc_minus;
            npc_minus[0] = 0.5 * (xp[1] - xpMinus[1]);
            npc_minus[1] = 0.5 * (xpMinus[0] - xp[0]);
            double lpc_minus = MathFunctions::norm(npc_minus);
            npc_minus = npc_minus / lpc_minus;
            m_nminus(pNodes, cCellsOfNodeP) = npc_minus;
            m_lminus(pNodes, cCellsOfNodeP) = lpc_minus;
            m_lpc(pNodes, cCellsOfNodeP) =
                (lpc_plus * npc_plus) + (lpc_minus * npc_minus);
          }
        }
      });
}
/**
 * Job computeEOS called in executeTimeLoopN method.
 */
void Eucclhyd::computeEOS() {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    for (int imat = 0; imat < options->nbmat; ++imat) {
      double pression;  // m_pressure_env(cCells)[imat]
      double density = m_density_env_n(cCells)[imat];
      double energy = m_internal_energy_env_n(cCells)[imat];
      double gamma = eos->gamma[imat];
      double tension_limit = eos->tension_limit[imat];
      double sound_speed;  // = m_speed_velocity_env_nplus1(cCells)[imat];
      RealArray1D<2> sortie_eos;  // pression puis sound_speed
      if (eos->Nom[imat] == eos->PerfectGas)
        sortie_eos = eos->computeEOSGP(gamma, density, energy);
      if (eos->Nom[imat] == eos->Void)
        sortie_eos = eos->computeEOSVoid(density, energy);
      if (eos->Nom[imat] == eos->StiffenedGas)
        sortie_eos =
            eos->computeEOSSTIFG(gamma, tension_limit, density, energy);
      if (eos->Nom[imat] == eos->Fictif)
        sortie_eos = eos->computeEOSFictif(gamma, density, energy);
      if (eos->Nom[imat] == eos->SolidLinear)
        sortie_eos = eos->computeEOSSL(density, energy);

      m_pressure_env(cCells)[imat] = sortie_eos[0];
      m_speed_velocity_env(cCells)[imat] = sortie_eos[1];
    }
  });
}
/**
 * Job computeEOS called in executeTimeLoopN method.
 */
void Eucclhyd::computePressionMoyenne() noexcept {
  for (int cCells = 0; cCells < nbCells; cCells++) {
    int nbmat = options->nbmat;
    m_pressure(cCells) = 0.;
    for (int imat = 0; imat < nbmat; ++imat) {
      m_pressure(cCells) +=
          m_fracvol_env(cCells)[imat] * m_pressure_env(cCells)[imat];
      m_speed_velocity(cCells) = MathFunctions::max(
          m_speed_velocity(cCells), m_speed_velocity_env(cCells)[imat]);
    }
    // NONREG GP A SUPPRIMER
    if (m_density_n(cCells) > 0.) {
      m_speed_velocity(cCells) = MathFunctions::sqrt(
          eos->gamma[0] * m_pressure(cCells) / m_density_n(cCells));
    }
  }
}
/**
 * Job computeGradients called @1.0 in executeTimeLoopN method.
 * In variables: m_node_force_n, m_node_velocity_n, m_lpc, spaceOrder, v
 * Out variables: m_velocity_gradient, m_pressure_gradient,
 * m_pressure_gradient_env
 */
void Eucclhyd::computeGradients() noexcept {
  Kokkos::parallel_for(
      "computeDissipationMatrix", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
        int pId(pNodes);
        {
          int nbmat = options->nbmat;
          for (int imat = 0; imat < nbmat; imat++)
            m_node_fracvol(pNodes)[imat] = 0.;
          auto cellsOfNodeP(mesh->getCellsOfNode(pId));
          for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
               cCellsOfNodeP++) {
            int cId(cellsOfNodeP[cCellsOfNodeP]);
            int cCells(cId);
            for (int imat = 0; imat < nbmat; imat++)
              m_node_fracvol(pNodes)[imat] +=
                  m_fracvol_env(cCells)[imat] * 0.25;
          }
        }
      });
  Kokkos::parallel_for(
      "computeGradients", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        size_t cId(cCells);
        RealArray1D<dim> reductionF1 = zeroVect;
        RealArray1D<dim> reductionF2 = zeroVect;
        RealArray1D<dim> reductionF3 = zeroVect;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
            int pNodes(pId);
            reductionF1 = reductionF1 + (m_node_fracvol(pNodes)[0] *
                                         m_lpc(pNodes, cCellsOfNodeP));
            reductionF2 = reductionF2 + (m_node_fracvol(pNodes)[1] *
                                         m_lpc(pNodes, cCellsOfNodeP));
            reductionF3 = reductionF3 + (m_node_fracvol(pNodes)[2] *
                                         m_lpc(pNodes, cCellsOfNodeP));
          }
        }
        particules->m_particlecell_fracvol_gradient_env(cCells, 0) =
            reductionF1 / m_euler_volume(cCells);
        particules->m_particlecell_fracvol_gradient_env(cCells, 1) =
            reductionF2 / m_euler_volume(cCells);
        particules->m_particlecell_fracvol_gradient_env(cCells, 2) =
            reductionF3 / m_euler_volume(cCells);
      });
  if (options->spaceOrder == 2)
    Kokkos::parallel_for(
        "computeGradients", nbCells, KOKKOS_LAMBDA(const int& cCells) {
          size_t cId(cCells);
          RealArray2D<dim, dim> reduction14 = zeroMat;
          {
            auto nodesOfCellC(mesh->getNodesOfCell(cId));
            for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
                 pNodesOfCellC++) {
              int pId(nodesOfCellC[pNodesOfCellC]);
              int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
              int pNodes(pId);
              reduction14 =
                  reduction14 + (tensProduct(m_node_velocity_n(pNodes),
                                             m_lpc(pNodes, cCellsOfNodeP)));
            }
          }
          m_velocity_gradient(cCells) = reduction14 / m_euler_volume(cCells);
          RealArray1D<dim> reduction15 = zeroVect;
          RealArray1D<dim> reduction15a = zeroVect;
          RealArray1D<dim> reduction15b = zeroVect;
          RealArray1D<dim> reduction15c = zeroVect;
          {
            auto nodesOfCellC(mesh->getNodesOfCell(cId));
            for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
                 pNodesOfCellC++) {
              int pId(nodesOfCellC[pNodesOfCellC]);
              int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
              int pNodes(pId);
              reduction15 = reduction15 + m_node_force_n(pNodes, cCellsOfNodeP);
              reduction15a =
                  reduction15a + m_node_force_env_n(pNodes, cCellsOfNodeP, 0);
              reduction15b =
                  reduction15b + m_node_force_env_n(pNodes, cCellsOfNodeP, 1);
              reduction15c =
                  reduction15c + m_node_force_env_n(pNodes, cCellsOfNodeP, 2);
            }
          }
          m_pressure_gradient(cCells) = reduction15 / m_euler_volume(cCells);
          m_pressure_gradient_env(cCells, 0) =
              reduction15a / m_euler_volume(cCells);
          m_pressure_gradient_env(cCells, 1) =
              reduction15b / m_euler_volume(cCells);
          m_pressure_gradient_env(cCells, 2) =
              reduction15c / m_euler_volume(cCells);
        });
}

/**
 * Job computeMass called @1.0 in executeTimeLoopN method.
 * In variables: m_density_n, v
 * Out variables: m
 */
void Eucclhyd::computeMass() noexcept {
  int nbmat = options->nbmat;
  Kokkos::parallel_for(
      "computeMass", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        int nbmat = options->nbmat;
        m_cell_mass(cCells) = m_density_n(cCells) * m_euler_volume(cCells);
        for (int imat = 0; imat < nbmat; imat++)
          m_cell_mass_env(cCells)[imat] =
              m_mass_fraction_env(cCells)[imat] * m_cell_mass(cCells);
      });
}
/**
 * Job computeDissipationMatrix called @2.0 in executeTimeLoopN method.
 * In variables: c, m_lminus, m_lplus, m_nminus, m_nplus, m_density_n
 * Out variables: M
 */
void Eucclhyd::computeDissipationMatrix() noexcept {
  Kokkos::parallel_for(
      "computeDissipationMatrix", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
        int pId(pNodes);
        {
          auto cellsOfNodeP(mesh->getCellsOfNode(pId));
          for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
               cCellsOfNodeP++) {
            int cId(cellsOfNodeP[cCellsOfNodeP]);
            int cCells(cId);
            RealArray2D<dim, dim> cornerMatrix =
                (m_lplus(pNodes, cCellsOfNodeP) *
                 tensProduct(m_nplus(pNodes, cCellsOfNodeP),
                             m_nplus(pNodes, cCellsOfNodeP))) +
                (m_lminus(pNodes, cCellsOfNodeP) *
                 tensProduct(m_nminus(pNodes, cCellsOfNodeP),
                             m_nminus(pNodes, cCellsOfNodeP)));

            m_dissipation_matrix(pNodes, cCellsOfNodeP) =
                m_density_n(cCells) * m_speed_velocity(cCells) * cornerMatrix;

            m_dissipation_matrix_env(pNodes, cCellsOfNodeP, 0) =
                m_density_env_n(cCells)[0] * m_speed_velocity(cCells) *
                cornerMatrix;
            m_dissipation_matrix_env(pNodes, cCellsOfNodeP, 1) =
                m_density_env_n(cCells)[1] * m_speed_velocity(cCells) *
                cornerMatrix;
            m_dissipation_matrix_env(pNodes, cCellsOfNodeP, 2) =
                m_density_env_n(cCells)[2] * m_speed_velocity(cCells) *
                cornerMatrix;
          }
        }
      });
}
/**
 * Job computem_cell_deltat called @2.0 in executeTimeLoopN method.
 * In variables: m_cell_velocity_n, c, m_cell_perimeter, v
 * Out variables: m_cell_deltat
 */
void Eucclhyd::computem_cell_deltat() noexcept {
  Kokkos::parallel_for(
      "computem_cell_deltat", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        if (options->AvecProjection == 1) {
          // cfl euler
          m_cell_deltat(cCells) =
              m_euler_volume(cCells) /
              (m_cell_perimeter(cCells) *
               (MathFunctions::norm(m_cell_velocity_n(cCells)) +
                m_speed_velocity(cCells)));
        } else {
          // cfl lagrange
          m_cell_deltat(cCells) =
              m_euler_volume(cCells) /
              (m_cell_perimeter(cCells) * m_speed_velocity(cCells));
        }
      });
}
/**
 * Job extrapolateValue called @2.0 in executeTimeLoopN method.
 * In variables: m_cell_velocity_n, , m_cell_coord, m_velocity_gradient,
 * m_pressure_gradient, p, spaceOrder Out variables: m_cell_velocity_extrap,
 * m_pressure_extrap, m_pressure_env_extrap
 */
void Eucclhyd::extrapolateValue() noexcept {
  if (options->spaceOrder == 1) {
    Kokkos::parallel_for(
        "extrapolateValue", nbCells, KOKKOS_LAMBDA(const int& cCells) {
          int cId(cCells);
          {
            auto nodesOfCellC(mesh->getNodesOfCell(cId));
            for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
                 pNodesOfCellC++) {
              m_cell_velocity_extrap(cCells, pNodesOfCellC) =
                  m_cell_velocity_n(cCells);
              m_pressure_extrap(cCells, pNodesOfCellC) = m_pressure(cCells);
              int nbmat = options->nbmat;
              for (int imat = 0; imat < nbmat; imat++)
                m_pressure_env_extrap(cCells, pNodesOfCellC)[imat] =
                    m_pressure_env(cCells)[imat];
            }
          }
        });
  } else {
    Kokkos::parallel_for(
        "extrapolateValue", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
          size_t pId(pNodes);
          double reduction16 = numeric_limits<double>::max();
          double reduction16a = numeric_limits<double>::max();
          double reduction16b = numeric_limits<double>::max();
          double reduction16c = numeric_limits<double>::max();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction16 = MathFunctions::min(reduction16, m_pressure(dCells));
              reduction16a =
                  MathFunctions::min(reduction16a, m_pressure_env(dCells)[0]);
              reduction16b =
                  MathFunctions::min(reduction16b, m_pressure_env(dCells)[1]);
              reduction16c =
                  MathFunctions::min(reduction16c, m_pressure_env(dCells)[2]);
            }
          }
          double minP = reduction16;
          double minP1 = reduction16a;
          double minP2 = reduction16b;
          double minP3 = reduction16c;
          double reduction17 = numeric_limits<double>::min();
          double reduction17a = numeric_limits<double>::min();
          double reduction17b = numeric_limits<double>::min();
          double reduction17c = numeric_limits<double>::min();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction17 = MathFunctions::max(reduction17, m_pressure(dCells));
              reduction17a =
                  MathFunctions::max(reduction17a, m_pressure_env(dCells)[0]);
              reduction17b =
                  MathFunctions::max(reduction17b, m_pressure_env(dCells)[1]);
              reduction17c =
                  MathFunctions::max(reduction17c, m_pressure_env(dCells)[2]);
            }
          }
          double maxP = reduction17;
          double maxP1 = reduction17a;
          double maxP2 = reduction17b;
          double maxP3 = reduction17c;
          double reduction18 = numeric_limits<double>::max();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction18 =
                  MathFunctions::min(reduction18, m_cell_velocity_n(dCells)[0]);
            }
          }
          double minVx = reduction18;
          double reduction19 = numeric_limits<double>::min();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction19 =
                  MathFunctions::max(reduction19, m_cell_velocity_n(dCells)[0]);
            }
          }
          double maxVx = reduction19;
          double reduction20 = numeric_limits<double>::max();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction20 =
                  MathFunctions::min(reduction20, m_cell_velocity_n(dCells)[1]);
            }
          }
          double minVy = reduction20;
          double reduction21 = numeric_limits<double>::min();
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int dCellsOfNodeP = 0; dCellsOfNodeP < cellsOfNodeP.size();
                 dCellsOfNodeP++) {
              int dId(cellsOfNodeP[dCellsOfNodeP]);
              int dCells(dId);
              reduction21 =
                  MathFunctions::max(reduction21, m_cell_velocity_n(dCells)[1]);
            }
          }
          double maxVy = reduction21;
          {
            auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
                 cCellsOfNodeP++) {
              int cId(cellsOfNodeP[cCellsOfNodeP]);
              int cCells(cId);
              int pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));

              // double ptmp = m_pressure(cCells) +
              // MathFunctions::dot(m_pressure_gradient(cCells),
              // ArrayOperations::minus(m_node_coord(pNodes),
              // m_cell_coord(cCells))); m_pressure_extrap(cCells,pNodesOfCellC)
              // = MathFunctions::max(MathFunctions::min(maxP, ptmp), minP);

              // pour chaque matériau,
              double ptmp1 = m_pressure_env(cCells)[0] +
                             dot(m_pressure_gradient_env(cCells, 0),
                                 (m_node_coord(pNodes) - m_cell_coord(cCells)));
              m_pressure_env_extrap(cCells, pNodesOfCellC)[0] =
                  MathFunctions::max(MathFunctions::min(maxP1, ptmp1), minP1);
              double ptmp2 = m_pressure_env(cCells)[1] +
                             dot(m_pressure_gradient_env(cCells, 1),
                                 (m_node_coord(pNodes) - m_cell_coord(cCells)));
              m_pressure_env_extrap(cCells, pNodesOfCellC)[1] =
                  MathFunctions::max(MathFunctions::min(maxP2, ptmp2), minP2);
              double ptmp3 = m_pressure_env(cCells)[2] +
                             dot(m_pressure_gradient_env(cCells, 2),
                                 (m_node_coord(pNodes) - m_cell_coord(cCells)));
              m_pressure_env_extrap(cCells, pNodesOfCellC)[2] =
                  MathFunctions::max(MathFunctions::min(maxP3, ptmp3), minP3);

              m_pressure_extrap(cCells, pNodesOfCellC) = 0.;
              int nbmat = options->nbmat;
              // et on recalcule la moyenne
              for (int imat = 0; imat < nbmat; imat++)
                m_pressure_extrap(cCells, pNodesOfCellC) +=
                    m_fracvol_env(cCells)[imat] *
                    m_pressure_env_extrap(cCells, pNodesOfCellC)[imat];

              RealArray1D<dim> Vtmp =
                  m_cell_velocity_n(cCells) +
                  MathFunctions::matVectProduct(
                      m_velocity_gradient(cCells),
                      (m_node_coord(pNodes) - m_cell_coord(cCells)));
              m_cell_velocity_extrap(cCells, pNodesOfCellC)[0] =
                  std::max(MathFunctions::min(maxVx, Vtmp[0]), minVx);
              m_cell_velocity_extrap(cCells, pNodesOfCellC)[1] =
                  std::max(MathFunctions::min(maxVy, Vtmp[1]), minVy);
            }
          }
        });
  }
}
/**
 * Job computeG called @3.0 in executeTimeLoopN method.
 * In variables: M, m_cell_velocity_extrap, m_lpc, m_pressure_extrap
 * Out variables: G
 */
void Eucclhyd::computeG() noexcept {
  Kokkos::parallel_for("computeG", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
    size_t pId(pNodes);
    RealArray1D<dim> reduction1 = zeroVect;
    {
      auto cellsOfNodeP(mesh->getCellsOfNode(pId));
      for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
           cCellsOfNodeP++) {
        int cId(cellsOfNodeP[cCellsOfNodeP]);
        int cCells(cId);
        int pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
        reduction1 =
            reduction1 + (MathFunctions::matVectProduct(
                              m_dissipation_matrix(pNodes, cCellsOfNodeP),
                              m_cell_velocity_extrap(cCells, pNodesOfCellC)) +
                          (m_pressure_extrap(cCells, pNodesOfCellC) *
                           m_lpc(pNodes, cCellsOfNodeP)));
      }
    }
    m_node_G(pNodes) = reduction1;
  });
}

/**
 * Job computeNodeDissipationMatrixAndG called @3.0 in executeTimeLoopN method.
 * In variables: M
 * Out variables: m_node_dissipation
 */
void Eucclhyd::computeNodeDissipationMatrixAndG() noexcept {
  Kokkos::parallel_for(
      "computeNodeDissipationMatrixAndG", nbNodes,
      KOKKOS_LAMBDA(const int& pNodes) {
        int pId(pNodes);
        RealArray2D<dim, dim> reduction0 = zeroMat;
        {
          auto cellsOfNodeP(mesh->getCellsOfNode(pId));
          for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
               cCellsOfNodeP++) {
            reduction0 =
                reduction0 + (m_dissipation_matrix(pNodes, cCellsOfNodeP));
          }
        }
        m_node_dissipation(pNodes) = reduction0;
      });
}
/**
 * Job computeNodeVelocity called @4.0 in executeTimeLoopN method.
 * In variables: G, m_node_dissipation
 * Out variables: m_node_velocity_nplus1
 */
void Eucclhyd::computeNodeVelocity() noexcept {
  auto innerNodes(mesh->getInnerNodes());
  int nbInnerNodes(mesh->getNbInnerNodes());
  Kokkos::parallel_for(
      "computeNodeVelocity", nbInnerNodes,
      KOKKOS_LAMBDA(const int& pInnerNodes) {
        int pId(innerNodes[pInnerNodes]);
        int pNodes(pId);
        m_node_velocity_nplus1(pNodes) = MathFunctions::matVectProduct(
            inverse(m_node_dissipation(pNodes)), m_node_G(pNodes));
      });
}
/**
 * Job computeFaceVelocity called @5.0 in executeTimeLoopN method.
 * In variables: m_node_velocity_nplus1, faceNormal
 * Out variables: faceNormalVelocity
 */
void Eucclhyd::computeFaceVelocity() noexcept {
  auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "computeFaceVelocity", nbFaces, KOKKOS_LAMBDA(const int& fFaces) {
        int fId(faces[fFaces]);
        RealArray1D<dim> reduction5 = zeroVect;
        {
          auto nodesOfFaceF(mesh->getNodesOfFace(fId));
          for (int pNodesOfFaceF = 0; pNodesOfFaceF < nodesOfFaceF.size();
               pNodesOfFaceF++) {
            int pId(nodesOfFaceF[pNodesOfFaceF]);
            int pNodes(pId);
            reduction5 = reduction5 + (m_node_velocity_nplus1(pNodes));
	    // normalement cela devrait etre
	    // reduction5 = reduction5 + .5 * (m_node_velocity_nplus1(pNodes) + m_node_velocity_n(pNodes));
          }
        }
        varlp->faceNormalVelocity(fFaces) =
            dot((0.5 * reduction5), varlp->faceNormal(fFaces));
      });
}

/**
 * Job computeLagrangePosition called @5.0 in executeTimeLoopN method.
 * In variables: m_node_velocity_nplus1, m_node_coord, deltat_n
 * Out variables: XLagrange
 */
void Eucclhyd::computeLagrangePosition() noexcept {
  Kokkos::parallel_for("computeLagrangePosition", nbNodes,
                       KOKKOS_LAMBDA(const int& pNodes) {
                         varlp->XLagrange(pNodes) =
                             m_node_coord(pNodes) +
                             m_node_velocity_nplus1(pNodes) * gt->deltat_n;
                       });
  auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "computeLagrangePosition", nbFaces, KOKKOS_LAMBDA(const int& fFaces) {
        int fId(faces[fFaces]);
        int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
        int n1Id(n1FirstNodeOfFaceF);
        int n1Nodes(n1Id);
        int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
        int n2Id(n2SecondNodeOfFaceF);
        int n2Nodes(n2Id);
        RealArray1D<dim> X_face =
            0.5 * (varlp->XLagrange(n1Nodes) + varlp->XLagrange(n2Nodes));
        RealArray1D<dim> face_vec =
            varlp->XLagrange(n2Nodes) - varlp->XLagrange(n1Nodes);
        varlp->XfLagrange(fFaces) = X_face;
        varlp->faceLengthLagrange(fFaces) = MathFunctions::norm(face_vec);
      });
  if (options->AvecProjection == 0) {
    Kokkos::parallel_for(
        "computeLagrangePosition", nbCells, KOKKOS_LAMBDA(const int& cCells) {
          int cId(cCells);
          double reduction13 = 0.0;
          {
            auto nodesOfCellC(mesh->getNodesOfCell(cId));
            for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
                 pNodesOfCellC++) {
              int pId(nodesOfCellC[pNodesOfCellC]);
              int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                        nbNodesOfCell]);
              int pNodes(pId);
              int pPlus1Nodes(pPlus1Id);
              reduction13 = reduction13 +
                            (MathFunctions::norm(m_node_coord(pNodes) -
                                                 m_node_coord(pPlus1Nodes)));
            }
          }
          m_cell_perimeter(cCells) = reduction13;
        });
  }
}

/**
 * Job computeSubCellForce called @5.0 in executeTimeLoopN method.
 * In variables: M, m_cell_velocity_extrap, m_node_velocity_nplus1, m_lpc,
 * m_pressure_extrap Out variables: m_node_force_nplus1
 */
void Eucclhyd::computeSubCellForce() noexcept {
  Kokkos::parallel_for(
      "computeSubCellForce", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
        size_t pId(pNodes);
        {
          auto cellsOfNodeP(mesh->getCellsOfNode(pId));
          for (int cCellsOfNodeP = 0; cCellsOfNodeP < cellsOfNodeP.size();
               cCellsOfNodeP++) {
            int cId(cellsOfNodeP[cCellsOfNodeP]);
            int cCells(cId);
            int pNodesOfCellC(utils::indexOf(mesh->getNodesOfCell(cId), pId));
            m_node_force_nplus1(pNodes, cCellsOfNodeP) =
                (-m_pressure_extrap(cCells, pNodesOfCellC) *
                 m_lpc(pNodes, cCellsOfNodeP)) +
                MathFunctions::matVectProduct(
                    m_dissipation_matrix(pNodes, cCellsOfNodeP),
                    (m_node_velocity_nplus1(pNodes) -
                     m_cell_velocity_extrap(cCells, pNodesOfCellC)));

            m_node_force_env_nplus1(pNodes, cCellsOfNodeP, 0) =
                (-m_pressure_env_extrap(cCells, pNodesOfCellC)[0] *
                 m_lpc(pNodes, cCellsOfNodeP)) +
                MathFunctions::matVectProduct(
                    m_dissipation_matrix_env(pNodes, cCellsOfNodeP, 0),
                    m_node_velocity_nplus1(pNodes) -
                        m_cell_velocity_extrap(cCells, pNodesOfCellC));

            m_node_force_env_nplus1(pNodes, cCellsOfNodeP, 1) =
                (-m_pressure_env_extrap(cCells, pNodesOfCellC)[1] *
                 m_lpc(pNodes, cCellsOfNodeP)) +
                MathFunctions::matVectProduct(
                    m_dissipation_matrix_env(pNodes, cCellsOfNodeP, 1),
                    m_node_velocity_nplus1(pNodes) -
                        m_cell_velocity_extrap(cCells, pNodesOfCellC));

            m_node_force_env_nplus1(pNodes, cCellsOfNodeP, 2) =
                (-m_pressure_env_extrap(cCells, pNodesOfCellC)[2] *
                 m_lpc(pNodes, cCellsOfNodeP)) +
                MathFunctions::matVectProduct(
                    m_dissipation_matrix_env(pNodes, cCellsOfNodeP, 2),
                    m_node_velocity_nplus1(pNodes) -
                        m_cell_velocity_extrap(cCells, pNodesOfCellC));
          }
        }
      });
}
/**
 * Job computeLagrangeVolumeAndCenterOfGravity called @6.0 in executeTimeLoopN
 * method. In variables: XLagrange Out variables: XcLagrange, vLagrange
 */
void Eucclhyd::computeLagrangeVolumeAndCenterOfGravity() noexcept {
  Kokkos::parallel_for(
      "computeLagrangeVolumeAndCenterOfGravity", nbCells,
      KOKKOS_LAMBDA(const int& cCells) {
        int cId(cCells);
        double reduction6 = 0.0;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            reduction6 =
                reduction6 + (crossProduct2d(varlp->XLagrange(pNodes),
                                             varlp->XLagrange(pPlus1Nodes)));
          }
        }
        double vol = 0.5 * reduction6;
        varlp->vLagrange(cCells) = vol;
        int nbmat = options->nbmat;
        for (int imat = 0; imat < nbmat; imat++)
          m_lagrange_volume(cCells)[imat] = m_fracvol_env(cCells)[imat] * vol;
        RealArray1D<dim> reduction7 = zeroVect;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pPlus1Id(nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) %
                                      nbNodesOfCell]);
            int pNodes(pId);
            int pPlus1Nodes(pPlus1Id);
            reduction7 =
                reduction7 +
                (crossProduct2d(varlp->XLagrange(pNodes),
                                varlp->XLagrange(pPlus1Nodes)) *
                 (varlp->XLagrange(pNodes) + varlp->XLagrange(pPlus1Nodes)));
          }
        }
        varlp->XcLagrange(cCells) = (1.0 / (6.0 * vol) * reduction7);
      });
}
/**
 * Job computeFacedeltaxLagrange called @7.0 in executeTimeLoopN method.
 * In variables: XcLagrange, faceNormal
 * Out variables: deltaxLagrange
 */
void Eucclhyd::computeFacedeltaxLagrange() noexcept {
  auto faces(mesh->getInnerFaces());
  int nbInnerFaces(mesh->getNbInnerFaces());
  // auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "computeFacedeltaxLagrange", nbInnerFaces,
      KOKKOS_LAMBDA(const int& fFaces) {
        size_t fId(faces[fFaces]);
        int cfFrontCellF(mesh->getFrontCell(fId));
        int cfId(cfFrontCellF);
        int cfCells(cfId);
        int cbBackCellF(mesh->getBackCell(fId));
        int cbId(cbBackCellF);
        int cbCells(cbId);
        varlp->deltaxLagrange(fId) =
            dot((varlp->XcLagrange(cfCells) - varlp->XcLagrange(cbCells)),
                varlp->faceNormal(fId));
      });
}

/**
 * Job updateCellCenteredLagrangeVariables called @7.0 in executeTimeLoopN
 * method. In variables: m_node_force_nplus1, m_cell_velocity_n,
 * m_node_velocity_nplus1, deltat_n, m_internal_energy_n, m_lpc, m, m_density_n,
 * vLagrange Out variables: ULagrange
 */
void Eucclhyd::updateCellCenteredLagrangeVariables() noexcept {
  Kokkos::parallel_for(
      "updateCellCenteredLagrangeVariables", nbCells,
      KOKKOS_LAMBDA(const int& cCells) {
        size_t cId(cCells);
        double reduction2 = 0.0;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
            int pNodes(pId);
            reduction2 = reduction2 + (dot(m_lpc(pNodes, cCellsOfNodeP),
                                           m_node_velocity_nplus1(pNodes)));
          }
        }
        double rhoLagrange =
            1 / (1 / m_density_n(cCells) +
                 gt->deltat_n / m_cell_mass(cCells) * reduction2);

        RealArray1D<dim> reduction3 = zeroVect;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
            int pNodes(pId);
            reduction3 =
                reduction3 + m_node_force_nplus1(pNodes, cCellsOfNodeP);
          }
        }
        particules->m_particlecell_pressure_gradient(cCells) =
            reduction3 / m_euler_volume(cCells);
        RealArray1D<dim> cell_velocity_L =
            m_cell_velocity_n(cCells) +
            reduction3 * gt->deltat_n / m_cell_mass(cCells);

        double reduction4 = 0.0;
        RealArray1D<nbmatmax> preduction4 = zeroVectmat;
        {
          auto nodesOfCellC(mesh->getNodesOfCell(cId));
          for (int pNodesOfCellC = 0; pNodesOfCellC < nodesOfCellC.size();
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int cCellsOfNodeP(utils::indexOf(mesh->getCellsOfNode(pId), cId));
            int pNodes(pId);
            reduction4 =
                reduction4 +
                (dot(m_node_force_nplus1(pNodes, cCellsOfNodeP),
                     (m_node_velocity_nplus1(pNodes) -
                      (0.5 * (m_cell_velocity_n(cCells) + cell_velocity_L)))));
            preduction4[0] =
                preduction4[0] +
                (dot(m_node_force_env_nplus1(pNodes, cCellsOfNodeP, 0),
                     (m_node_velocity_nplus1(pNodes) -
                      (0.5 * (m_cell_velocity_n(cCells) + cell_velocity_L)))));
            preduction4[1] =
                preduction4[1] +
                (dot(m_node_force_env_nplus1(pNodes, cCellsOfNodeP, 1),
                     (m_node_velocity_nplus1(pNodes) -
                      (0.5 * (m_cell_velocity_n(cCells) + cell_velocity_L)))));
            preduction4[2] =
                preduction4[2] +
                (dot(m_node_force_env_nplus1(pNodes, cCellsOfNodeP, 2),
                     (m_node_velocity_nplus1(pNodes) -
                      (0.5 * (m_cell_velocity_n(cCells) + cell_velocity_L)))));
          }
        }

        int nbmat = options->nbmat;
        double eLagrange = m_internal_energy_n(cCells) +
                           gt->deltat_n / m_cell_mass(cCells) * reduction4;

        RealArray1D<nbmatmax> peLagrange;
        RealArray1D<nbmatmax> peLagrangec;
        for (int imat = 0; imat < nbmat; imat++) {
          peLagrange[imat] = 0.;
          peLagrangec[imat] = 0.;
          if (m_fracvol_env(cCells)[imat] > options->threshold &&
              m_cell_mass_env(cCells)[imat] != 0.)
            peLagrange[imat] = m_internal_energy_env_n(cCells)[imat] +
                               m_fracvol_env(cCells)[imat] * gt->deltat_n /
                                   m_cell_mass_env(cCells)[imat] *
                                   preduction4[imat];
        }
        for (int imat = 0; imat < nbmat; imat++) {
          varlp->ULagrange(cCells)[imat] = m_lagrange_volume(cCells)[imat];

          varlp->ULagrange(cCells)[nbmat + imat] =
              m_mass_fraction_env(cCells)[imat] * varlp->vLagrange(cCells) *
              rhoLagrange;

          varlp->ULagrange(cCells)[2 * nbmat + imat] =
              m_mass_fraction_env(cCells)[imat] * varlp->vLagrange(cCells) *
              rhoLagrange * peLagrange[imat];
        }

        varlp->ULagrange(cCells)[3 * nbmat] =
            varlp->vLagrange(cCells) * rhoLagrange * cell_velocity_L[0];
        varlp->ULagrange(cCells)[3 * nbmat + 1] =
            varlp->vLagrange(cCells) * rhoLagrange * cell_velocity_L[1];
        // projection de l'energie cinétique
        if (options->projectionConservative == 1)
          varlp->ULagrange(cCells)[3 * nbmat + 2] =
              0.5 * varlp->vLagrange(cCells) * rhoLagrange *
              (cell_velocity_L[0] * cell_velocity_L[0] +
               cell_velocity_L[1] * cell_velocity_L[1]);

        if (options->AvecProjection == 0) {
          // Calcul des valeurs en n+1 si on ne fait pas de projection
          // m_node_velocity_nplus1
          m_cell_velocity_nplus1(cCells) = cell_velocity_L;
          // densites et energies
          m_density_nplus1(cCells) = 0.;
          m_internal_energy_nplus1(cCells) = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            // densités
            m_density_nplus1(cCells) +=
                m_mass_fraction_env(cCells)[imat] * rhoLagrange;
            if (m_fracvol_env(cCells)[imat] > options->threshold) {
              m_density_env_nplus1(cCells)[imat] =
                  m_mass_fraction_env(cCells)[imat] * rhoLagrange /
                  m_fracvol_env(cCells)[imat];
            } else {
              m_density_env_nplus1(cCells)[imat] = 0.;
            }
            // energies
            m_internal_energy_nplus1(cCells) +=
                m_mass_fraction_env(cCells)[imat] * peLagrange[imat];
            if (m_fracvol_env(cCells)[imat] > options->threshold) {
              m_internal_energy_env_nplus1(cCells)[imat] = peLagrange[imat];
            } else {
              m_internal_energy_env_nplus1(cCells)[imat] = 0.;
            }
          }
          // variables pour les sorties du code
          m_fracvol_env1(cCells) = m_fracvol_env(cCells)[0];
          m_fracvol_env2(cCells) = m_fracvol_env(cCells)[1];
          m_fracvol_env3(cCells) = m_fracvol_env(cCells)[2];
          // sorties paraview limitées
          if (m_cell_velocity_nplus1(cCells)[0] > 0.)
            m_x_cell_velocity(cCells) = MathFunctions::max(
                m_cell_velocity_nplus1(cCells)[0], options->threshold);
          if (m_cell_velocity_nplus1(cCells)[0] < 0.)
            m_x_cell_velocity(cCells) = MathFunctions::min(
                m_cell_velocity_nplus1(cCells)[0], -options->threshold);

          if (m_cell_velocity_nplus1(cCells)[1] > 0.)
            m_y_cell_velocity(cCells) = MathFunctions::max(
                m_cell_velocity_nplus1(cCells)[1], options->threshold);
          if (m_cell_velocity_nplus1(cCells)[1] < 0.)
            m_y_cell_velocity(cCells) = MathFunctions::min(
                m_cell_velocity_nplus1(cCells)[1], -options->threshold);
          // pression
          m_pressure_env1(cCells) = m_pressure_env(cCells)[0];
          m_pressure_env2(cCells) = m_pressure_env(cCells)[1];
          m_pressure_env3(cCells) = m_pressure_env(cCells)[2];
        }

        if (limiteurs->projectionAvecPlateauPente == 1) {
          // option ou on ne regarde pas la variation de rho, V et e
          // phi = (f1, f2, rho1, rho2,  e1, e2, Vx, Vy,
          // ce qui permet d'ecrire le flm_x_velocity telque
          // Flm_x_velocity = (dv1 = f1dv, dv2=f2*dv, dm1=rho1*df1,
          // dm2=rho2*df2d(m1e1) = e1*dm1,  d(m2e2) = e2*dm2, d(mV) =
          // V*(dm1+dm2), dans computeFlm_x_velocityPP

          double somme_volume = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            somme_volume += varlp->ULagrange(cCells)[imat];
          }
          // Phi volume
          double somme_masse = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            varlp->Phi(cCells)[imat] =
                varlp->ULagrange(cCells)[imat] / somme_volume;

            // Phi masse
            if (varlp->ULagrange(cCells)[imat] != 0.)
              varlp->Phi(cCells)[nbmat + imat] =
                  varlp->ULagrange(cCells)[nbmat + imat] /
                  varlp->ULagrange(cCells)[imat];
            else
              varlp->Phi(cCells)[nbmat + imat] = 0.;
            somme_masse += varlp->ULagrange(cCells)[nbmat + imat];
          }
          // Phi Vitesse
          varlp->Phi(cCells)[3 * nbmat] =
              varlp->ULagrange(cCells)[3 * nbmat] / somme_masse;
          varlp->Phi(cCells)[3 * nbmat + 1] =
              varlp->ULagrange(cCells)[3 * nbmat + 1] / somme_masse;
          // Phi energie
          for (int imat = 0; imat < nbmat; imat++) {
            if (varlp->ULagrange(cCells)[nbmat + imat] != 0.)
              varlp->Phi(cCells)[2 * nbmat + imat] =
                  varlp->ULagrange(cCells)[2 * nbmat + imat] /
                  varlp->ULagrange(cCells)[nbmat + imat];
            else
              varlp->Phi(cCells)[2 * nbmat + imat] = 0.;
          }
          // Phi energie cinétique
          if (options->projectionConservative == 1)
            varlp->Phi(cCells)[3 * nbmat + 2] =
                varlp->ULagrange(cCells)[3 * nbmat + 2] / somme_masse;

        } else {
          varlp->Phi(cCells) =
              varlp->ULagrange(cCells) / varlp->vLagrange(cCells);
        }

#ifdef TEST
        if ((cCells == dbgcell3 || cCells == dbgcell2 || cCells == dbgcell1) &&
            test_debug == 1) {
          std::cout << " Apres Phase Lagrange cell   " << cCells << "Phi"
                    << varlp->Phi(cCells) << std::endl;
          std::cout << " cell   " << cCells << "varlp->ULagrange "
                    << varlp->ULagrange(cCells) << std::endl;
        }
        if (varlp->ULagrange(cCells) != varlp->ULagrange(cCells)) {
          std::cout << " cell   " << cCells << " varlp->Ulagrange "
                    << varlp->ULagrange(cCells) << std::endl;
          std::cout << " cell   " << cCells << " f1 "
                    << m_fracvol_env(cCells)[0] << " f2 "
                    << m_fracvol_env(cCells)[1] << " f3 "
                    << m_fracvol_env(cCells)[2] << std::endl;
          std::cout << " cell   " << cCells << " c1 "
                    << m_mass_fraction_env(cCells)[0] << " c2 "
                    << m_mass_fraction_env(cCells)[1] << " c3 "
                    << m_mass_fraction_env(cCells)[2] << std::endl;
          std::cout << " cell   " << cCells << " m1 "
                    << m_cell_mass_env(cCells)[0] << " m2 "
                    << m_cell_mass_env(cCells)[1] << " m3 "
                    << m_cell_mass_env(cCells)[2] << std::endl;
          std::cout << " cell   " << cCells << " peLagrange[0] "
                    << peLagrange[0] << " preduction4[0] " << preduction4[0]
                    << " m_internal_energy_env_n[0] "
                    << m_internal_energy_env_n(cCells)[0] << std::endl;
          std::cout << " cell   " << cCells << " peLagrange[1] "
                    << peLagrange[1] << " preduction4[1] " << preduction4[1]
                    << " m_internal_energy_env_n[1] "
                    << m_internal_energy_env_n(cCells)[1] << std::endl;
          std::cout << " cell   " << cCells << " peLagrange[2] "
                    << peLagrange[2] << " preduction4[2] " << preduction4[2]
                    << " m_internal_energy_env_n[2] "
                    << m_internal_energy_env_n(cCells)[2] << std::endl;
          std::cout << " densites  1 " << m_density_env_n(cCells)[0] << " 2 "
                    << m_density_env_n(cCells)[1] << " 3 "
                    << m_density_env_n(cCells)[2] << std::endl;
          exit(1);
        }
#endif

        // m_total_energy_L(cCells) = (rhoLagrange * vLagrange(cCells)) *
        // (eLagrange + 0.5 * (cell_velocity_L[0] * cell_velocity_L[0] +
        // cell_velocity_L[1] * cell_velocity_L[1]));
        m_global_masse_L(cCells) = 0.;
        m_total_energy_L(cCells) =
            (rhoLagrange * varlp->vLagrange(cCells)) *
            (m_mass_fraction_env(cCells)[0] * peLagrange[0] +
             m_mass_fraction_env(cCells)[1] * peLagrange[1] +
             m_mass_fraction_env(cCells)[2] * peLagrange[2] +
             0.5 * (cell_velocity_L[0] * cell_velocity_L[0] +
                    cell_velocity_L[1] * cell_velocity_L[1]));
        for (int imat = 0; imat < nbmat; imat++) {
          m_global_masse_L(cCells) += m_mass_fraction_env(cCells)[imat] *
                                      (rhoLagrange * varlp->vLagrange(cCells));
        }
      });
  double reductionE(0.), reductionM(0.);
  {
    Kokkos::Sum<double> reducerE(reductionE);
    Kokkos::parallel_reduce("reductionE", nbCells,
                            KOKKOS_LAMBDA(const int& cCells, double& x) {
                              reducerE.join(x, m_total_energy_L(cCells));
                            },
                            reducerE);
    Kokkos::Sum<double> reducerM(reductionM);
    Kokkos::parallel_reduce("reductionM", nbCells,
                            KOKKOS_LAMBDA(const int& cCells, double& x) {
                              reducerM.join(x, m_global_masse_L(cCells));
                            },
                            reducerM);
  }
  m_global_total_energy_L = reductionE;
  m_total_masse_L = reductionM;
}
void Eucclhyd::switchalpharho_rho() noexcept {
  Kokkos::parallel_for(
      "updateParticleCoefficient", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        m_density_n(cCells) /=
            particules->m_cell_particle_volume_fraction(cCells);
        for (int imat = 0; imat < nbmatmax; imat++)
          m_density_env_n(cCells)[imat] /=
              particules->m_cell_particle_volume_fraction(cCells);
      });
}

void Eucclhyd::switchrho_alpharho() noexcept {
  Kokkos::parallel_for(
      "updateParticleCoefficient", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        m_density_n(cCells) *=
            particules->m_cell_particle_volume_fraction(cCells);
        for (int imat = 0; imat < nbmatmax; imat++)
          m_density_env_n(cCells)[imat] *=
              particules->m_cell_particle_volume_fraction(cCells);
      });
}
void Eucclhyd::PreparecellvariablesForParticles() noexcept {
  Kokkos::parallel_for(
      "copycellvariables", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        for (int imat = 0; imat < nbmatmax; imat++) {
          particules->m_particlecell_fracvol_env(cCells)[imat] =
              m_fracvol_env(cCells)[imat];
          particules->m_particlecell_density_env_n(cCells)[imat] =
              m_density_env_n(cCells)[imat];
        }
        particules->m_particlecell_density_n(cCells) = m_density_n(cCells);
        particules->m_particlecell_euler_volume(cCells) =
            m_euler_volume(cCells);
        particules->m_particlecell_velocity_n(cCells) =
            m_cell_velocity_n(cCells);
        particules->m_particlecell_velocity_nplus1(cCells) =
            m_cell_velocity_nplus1(cCells);
        particules->m_particlecell_mass(cCells) = m_cell_mass(cCells);
      });
  std::cout << " fin prepare " << std::endl;
}
