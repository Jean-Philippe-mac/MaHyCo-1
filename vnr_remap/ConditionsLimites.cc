#include <Kokkos_Core.hpp>

#include "Vnr.h"                   // for VnrRemap
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "utils/Utils.h"           // for Indexof
// placer apres car a besoin des précédents
#include "../includes/Freefunctions.h"

/**
 *******************************************************************************
 * \file updateNodeBoundaryConditions()
 * \brief Mise à jour des conditions aux limites des valeurs aux noeuds du
 *maillage
 *
 * \param  ConditionsLimites  cdl
 * \return m_node_velocity_nplus1
 *******************************************************************************
 */
void Vnr::updateNodeBoundaryConditions() noexcept {
  const double dt(0.5 * (gt->deltat_nplus1 + gt->deltat_n));
  const auto bottomNodes(mesh->getBottomNodes());
  const size_t nbBottomNodes(bottomNodes.size());
  Kokkos::parallel_for(nbBottomNodes, KOKKOS_LAMBDA(
                                          const size_t& pBottomNodes) {
    const Id pId(bottomNodes[pBottomNodes]);
    const size_t pNodes(pId);

    if (cdl->bottomBC == cdl->symmetry) {
      RealArray1D<2> reduction1({0.0, 0.0});
      {
        const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
        const size_t nbCellsOfNodeP(cellsOfNodeP.size());
        for (size_t cCellsOfNodeP = 0; cCellsOfNodeP < nbCellsOfNodeP;
             cCellsOfNodeP++) {
          const Id cId(cellsOfNodeP[cCellsOfNodeP]);
          const size_t cCells(cId);
          const size_t pNodesOfCellC(
              utils::indexOf(mesh->getNodesOfCell(cId), pId));
          reduction1 = sumR1(
              reduction1,
              (m_pressure_n(cCells) + m_pseudo_viscosity_n(cCells)) *
                  (m_cqs_n(cCells, pNodesOfCellC) +
                   symmetricVector(m_cqs_n(cCells, pNodesOfCellC), {1.0, 0.0})));
        }
      }
      m_node_velocity_nplus1(pNodes) =
          m_node_velocity_n(pNodes) + dt / (m_node_mass(pNodes)) * reduction1;
    } else if (cdl->bottomBC == cdl->imposedVelocity) {
      m_node_velocity_nplus1(pNodes) = cdl->bottomBCValue;
    }
  });
  const auto topNodes(mesh->getTopNodes());
  const size_t nbTopNodes(topNodes.size());
  Kokkos::parallel_for(nbTopNodes, KOKKOS_LAMBDA(const size_t& pTopNodes) {
    const Id pId(topNodes[pTopNodes]);
    const size_t pNodes(pId);

    if (cdl->topBC == cdl->symmetry) {
      RealArray1D<2> reduction2({0.0, 0.0});
      {
        const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
        const size_t nbCellsOfNodeP(cellsOfNodeP.size());
        for (size_t cCellsOfNodeP = 0; cCellsOfNodeP < nbCellsOfNodeP;
             cCellsOfNodeP++) {
          const Id cId(cellsOfNodeP[cCellsOfNodeP]);
          const size_t cCells(cId);
          const size_t pNodesOfCellC(
              utils::indexOf(mesh->getNodesOfCell(cId), pId));
          reduction2 = sumR1(
              reduction2,
              (m_pressure_n(cCells) + m_pseudo_viscosity_n(cCells)) *
                  (m_cqs_n(cCells, pNodesOfCellC) +
                   symmetricVector(m_cqs_n(cCells, pNodesOfCellC), {1.0, 0.0})));
        }
      }
      m_node_velocity_nplus1(pNodes) =
          m_node_velocity_n(pNodes) + dt / (m_node_mass(pNodes)) * reduction2;
    } else if (cdl->topBC == cdl->imposedVelocity) {
      m_node_velocity_nplus1(pNodes) = cdl->topBCValue;
    }
  });
  const auto leftNodes(mesh->getLeftNodes());
  const size_t nbLeftNodes(leftNodes.size());
  Kokkos::parallel_for(nbLeftNodes, KOKKOS_LAMBDA(const size_t& pLeftNodes) {
    const Id pId(leftNodes[pLeftNodes]);
    const size_t pNodes(pId);
    if (cdl->leftBC == cdl->symmetry) {
      RealArray1D<2> reduction3({0.0, 0.0});
      {
        const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
        const size_t nbCellsOfNodeP(cellsOfNodeP.size());
        for (size_t cCellsOfNodeP = 0; cCellsOfNodeP < nbCellsOfNodeP;
             cCellsOfNodeP++) {
          const Id cId(cellsOfNodeP[cCellsOfNodeP]);
          const size_t cCells(cId);
          const size_t pNodesOfCellC(
              utils::indexOf(mesh->getNodesOfCell(cId), pId));
          reduction3 = sumR1(
              reduction3,
              (m_pressure_n(cCells) + m_pseudo_viscosity_n(cCells)) *
                  (m_cqs_n(cCells, pNodesOfCellC) +
                   symmetricVector(m_cqs_n(cCells, pNodesOfCellC), {0.0, 1.0})));
        }
      }
      m_node_velocity_nplus1(pNodes) =
          m_node_velocity_n(pNodes) + dt / (m_node_mass(pNodes)) * reduction3;
    } else if (cdl->leftBC == cdl->imposedVelocity) {
      m_node_velocity_nplus1(pNodes) = cdl->leftBCValue;
    }
  });
  const auto rightNodes(mesh->getRightNodes());
  const size_t nbRightNodes(rightNodes.size());
  Kokkos::parallel_for(nbRightNodes, KOKKOS_LAMBDA(const size_t& pRightNodes) {
    const Id pId(rightNodes[pRightNodes]);
    const size_t pNodes(pId);
    if (cdl->rightBC == cdl->symmetry) {
      RealArray1D<2> reduction4({0.0, 0.0});
      {
        const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
        const size_t nbCellsOfNodeP(cellsOfNodeP.size());
        for (size_t cCellsOfNodeP = 0; cCellsOfNodeP < nbCellsOfNodeP;
             cCellsOfNodeP++) {
          const Id cId(cellsOfNodeP[cCellsOfNodeP]);
          const size_t cCells(cId);
          const size_t pNodesOfCellC(
              utils::indexOf(mesh->getNodesOfCell(cId), pId));
          reduction4 = sumR1(
              reduction4,
              (m_pressure_n(cCells) + m_pseudo_viscosity_n(cCells)) *
                  (m_cqs_n(cCells, pNodesOfCellC) +
                   symmetricVector(m_cqs_n(cCells, pNodesOfCellC), {0.0, 1.0})));
        }
      }
      m_node_velocity_nplus1(pNodes) =
          m_node_velocity_n(pNodes) + dt / (m_node_mass(pNodes)) * reduction4;
    } else if (cdl->rightBC == cdl->imposedVelocity) {
      m_node_velocity_nplus1(pNodes) = cdl->rightBCValue;
    }
  });
  const auto topLeftNode(mesh->getTopLeftNode());
  const size_t nbTopLeftNode(mesh->getNbTopLeftNode());
  Kokkos::parallel_for(
      "computeBoundaryNodeVelocities", nbTopLeftNode,
      KOKKOS_LAMBDA(const int& pTopLeftNode) {
        int pId(topLeftNode[pTopLeftNode]);
        int pNodes(pId);
        if (cdl->topBC == cdl->symmetry && cdl->leftBC == cdl->symmetry)
          m_node_velocity_nplus1(pNodes) = zeroVect;
        else if (cdl->topBC == cdl->imposedVelocity &&
                 cdl->leftBC == cdl->imposedVelocity)
          m_node_velocity_nplus1(pNodes) = cdl->leftBCValue;
      });
  const auto topRightNode(mesh->getTopRightNode());
  const size_t nbTopRightNode(mesh->getNbTopRightNode());
  Kokkos::parallel_for(
      "computeBoundaryNodeVelocities", nbTopRightNode,
      KOKKOS_LAMBDA(const int& pTopRightNode) {
        int pId(topRightNode[pTopRightNode]);
        int pNodes(pId);
        if (cdl->topBC == cdl->symmetry && cdl->rightBC == cdl->symmetry)
          m_node_velocity_nplus1(pNodes) = zeroVect;
        else if (cdl->topBC == cdl->imposedVelocity &&
                 cdl->rightBC == cdl->imposedVelocity)
          m_node_velocity_nplus1(pNodes) = cdl->rightBCValue;
      });
  const auto bottomLeftNode(mesh->getBottomLeftNode());
  const size_t nbBottomLeftNode(mesh->getNbBottomLeftNode());
  Kokkos::parallel_for(
      "computeBoundaryNodeVelocities", nbBottomLeftNode,
      KOKKOS_LAMBDA(const int& pBottomLeftNode) {
        int pId(bottomLeftNode[pBottomLeftNode]);
        int pNodes(pId);
        if (cdl->bottomBC == cdl->symmetry && cdl->leftBC == cdl->symmetry)
          m_node_velocity_nplus1(pNodes) = zeroVect;
        else if (cdl->bottomBC == cdl->imposedVelocity &&
                 cdl->leftBC == cdl->imposedVelocity)
          m_node_velocity_nplus1(pNodes) = cdl->leftBCValue;
      });
  const auto bottomRightNode(mesh->getBottomRightNode());
  const size_t nbBottomRightNode(mesh->getNbBottomRightNode());
  Kokkos::parallel_for(
      "computeBoundaryNodeVelocities", nbBottomRightNode,
      KOKKOS_LAMBDA(const int& pBottomRightNode) {
        int pId(bottomRightNode[pBottomRightNode]);
        int pNodes(pId);
        if (cdl->bottomBC == cdl->symmetry && cdl->rightBC == cdl->symmetry)
          m_node_velocity_nplus1(pNodes) = zeroVect;
        else if (cdl->bottomBC == cdl->imposedVelocity &&
                 cdl->rightBC == cdl->imposedVelocity)
          m_node_velocity_nplus1(pNodes) = cdl->rightBCValue;
      });
}
/**
 *******************************************************************************
 * \file updateCellBoundaryConditions()
 * \brief Mise à jour des conditions aux limites des valeurs aux mailles du
 *maillage
 *
 * \param  ConditionsLimites  cdl
 * \return m_density_nplus1, m_internal_energy_nplus1,
 *         m_density_env_nplus1, m_internal_energy_env_nplus1
 *******************************************************************************
 */
void Vnr::updateCellBoundaryConditions() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCell) {
    if (cdl->rightCellBC == cdl->periodic) {
      int LeftCell = mesh->getLeftCellfromRight(cCell);
      if (LeftCell != -1) {
        m_density_nplus1(LeftCell) = init->m_density_n0(LeftCell);
        m_internal_energy_nplus1(LeftCell) =
            init->m_internal_energy_n0(LeftCell);
        m_density_nplus1(cCell) = m_density_nplus1(LeftCell);
        m_internal_energy_nplus1(cCell) = m_internal_energy_nplus1(LeftCell);
        int nbmat = options->nbmat;
        for (int imat = 0; imat < nbmat; ++imat) {
          m_density_env_nplus1(LeftCell)[imat] =
              init->m_density_env_n0(LeftCell)[imat];
          m_internal_energy_env_nplus1(LeftCell)[imat] =
              init->m_internal_energy_env_n0(LeftCell)[imat];
          m_density_env_nplus1(cCell)[imat] =
              m_density_env_nplus1(LeftCell)[imat];
          m_internal_energy_env_nplus1(cCell)[imat] =
              m_internal_energy_env_nplus1(LeftCell)[imat];
        }
      }
    }
    if (cdl->topCellBC == cdl->periodic) {
      int BottomCell = mesh->getBottomCellfromTop(cCell);
      if (BottomCell != -1) {
        m_density_nplus1(BottomCell) = init->m_density_n0(BottomCell);
        m_internal_energy_nplus1(BottomCell) =
            init->m_internal_energy_n0(BottomCell);
        m_density_nplus1(cCell) = m_density_nplus1(BottomCell);
        m_internal_energy_nplus1(cCell) = m_internal_energy_nplus1(BottomCell);
        int nbmat = options->nbmat;
        for (int imat = 0; imat < nbmat; ++imat) {
          m_density_env_nplus1(BottomCell)[imat] =
              init->m_density_env_n0(BottomCell)[imat];
          m_internal_energy_env_nplus1(BottomCell)[imat] =
              init->m_internal_energy_env_n0(BottomCell)[imat];
          m_density_env_nplus1(cCell)[imat] =
              m_density_env_nplus1(BottomCell)[imat];
          m_internal_energy_env_nplus1(cCell)[imat] =
              m_internal_energy_env_nplus1(BottomCell)[imat];
        }
      }
    }
  });
}
