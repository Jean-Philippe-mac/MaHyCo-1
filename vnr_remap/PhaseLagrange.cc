#include "Vnr.h"

using namespace nablalib;

#include "../includes/Freefunctions.h"
#include "types/MathFunctions.h"  // for max, min, dot, matVectProduct
#include "utils/Utils.h"          // for Indexof

/**
 *******************************************************************************
 * \file computeCellMass()
 * \brief Calcul de la masse des mailles
 *
 * \param  m_euler_volume_n0, m_density_n0, m_mass_fraction_env
 * \return m_cell_mass, m_cell_mass_env
 *******************************************************************************
 */
void Vnr::computeCellMass() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    int nbmat = options->nbmat;
    m_cell_mass(cCells) =
        init->m_euler_volume_n0(cCells) * init->m_density_n0(cCells);
    for (int imat = 0; imat < nbmat; ++imat) {
      m_cell_mass_env(cCells)[imat] =
          m_mass_fraction_env(cCells)[imat] * m_cell_mass(cCells);
    }
  });
}
/**
 *******************************************************************************
 * \file computeNodeMass()
 * \brief Calcul de la masse nodale
 *
 * \param  m_cell_mass
 * \return m_node_mass
 *******************************************************************************
 */
void Vnr::computeNodeMass() noexcept {
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    const Id pId(pNodes);
    const auto cells_of_node(mesh->getCellsOfNode(pId));
    double reduction0(0.0);
    {
      const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
      const size_t nbCellsOfNodeP(cellsOfNodeP.size());
      for (size_t cCellsOfNodeP = 0; cCellsOfNodeP < nbCellsOfNodeP;
           cCellsOfNodeP++) {
        const Id cId(cellsOfNodeP[cCellsOfNodeP]);
        const size_t cCells(cId);
        reduction0 = sumR0(reduction0, m_cell_mass(cCells));
      }
    }
    m_node_mass(pNodes) = reduction0 / cells_of_node.size();
  });
}
/**
 *******************************************************************************
 * \file computeArtificialViscosity()
 * \brief Calcul de la viscosité artificielle
 *
 * \param  m_node_cellvolume_n, m_tau_density_nplus1, m_speed_velocity_n,
 *         m_divu_nplus1, m_fracvol_env
 * \return m_pseudo_viscosity_nplus1, m_pseudo_viscosity_env_nplus1
 *******************************************************************************
 */
void Vnr::computeArtificialViscosity() noexcept {
  double reductionP(numeric_limits<double>::min());
  Kokkos::Max<double> reducer(reductionP);
  Kokkos::parallel_reduce("reductionP", nbCells,
                          KOKKOS_LAMBDA(const int& cCells, double& x) {
                            reducer.join(x, m_pressure_n(cCells));
                          },
                          reducer);
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    const Id cId(cCells);
    if (m_divu_n(cCells) < 0.0) {
      double reduction0(0.0);
      {
        const auto nodesOfCellC(mesh->getNodesOfCell(cId));
        const size_t nbNodesOfCellC(nodesOfCellC.size());
        for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
             pNodesOfCellC++) {
          reduction0 =
              sumR0(reduction0, m_node_cellvolume_n(cCells, pNodesOfCellC));
        }
      }
      double gamma(
          eos->gamma[0]);  // a changer en prenant une moyenne des gamma ?
      m_pseudo_viscosity_nplus1(cCells) =
          1.0 / m_tau_density_nplus1(cCells) *
          (-0.5 * std::sqrt(reduction0) * m_speed_velocity_n(cCells) *
               m_divu_nplus1(cCells) +
           (gamma + 1) / 2.0 * reduction0 * m_divu_nplus1(cCells) *
               m_divu_nplus1(cCells));
    } else
      m_pseudo_viscosity_nplus1(cCells) = 0.0;
    //
    // limitation par la pression
    // permet de limiter un exces de pseudo lié à des erreurs d'arrondis sur
    // m_tau_density_nplus1
    if (m_pseudo_viscosity_nplus1(cCells) > reductionP) {
      std::cout << cCells << " pseudo " << m_pseudo_viscosity_nplus1(cCells)
                << " pression " << m_pressure_n(cCells) << std::endl;
      m_pseudo_viscosity_nplus1(cCells) = reductionP;
    }
    // pour chaque matériau
    for (int imat = 0; imat < options->nbmat; ++imat)
      m_pseudo_viscosity_env_nplus1(cCells)[imat] =
          m_fracvol_env(cCells)[imat] * m_pseudo_viscosity_nplus1(cCells);
  });
}
/**
 *******************************************************************************
 * \file computeCornerNormal()
 * \brief Calcul des vecteurs de coin
 *
 * \param  m_node_coord_n
 * \return m_cqs
 *******************************************************************************
 */
void Vnr::computeCornerNormal() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    const Id cId(cCells);
    {
      const auto nodesOfCellC(mesh->getNodesOfCell(cId));
      const size_t nbNodesOfCellC(nodesOfCellC.size());
      for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
           pNodesOfCellC++) {
        const Id pId(nodesOfCellC[pNodesOfCellC]);
        const Id pPlus1Id(
            nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) % nbNodesOfCell]);
        const Id pMinus1Id(
            nodesOfCellC[(pNodesOfCellC - 1 + nbNodesOfCell) % nbNodesOfCell]);
        const size_t pNodes(pId);
        const size_t pPlus1Nodes(pPlus1Id);
        const size_t pMinus1Nodes(pMinus1Id);
        m_cqs(cCells, pNodesOfCellC) =
            computeLpcNpc(m_node_coord_n(pNodes), m_node_coord_n(pPlus1Nodes),
                          m_node_coord_n(pMinus1Nodes));
      }
    }
  });
}
/**
 *******************************************************************************
 * \file computeNodeVolume()
 * \brief Calcul du volume de chaque noeud du maillage
 *
 * \param  m_node_cellvolume_n
 * \return m_node_volume
 *******************************************************************************
 */
void Vnr::computeNodeVolume() noexcept {
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    const Id pId(pNodes);
    double reduction0(0.0);
    {
      const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
      const size_t nbCellsOfNodeP(cellsOfNodeP.size());
      for (size_t cCellsOfNodeP = 0; cCellsOfNodeP < nbCellsOfNodeP;
           cCellsOfNodeP++) {
        const Id cId(cellsOfNodeP[cCellsOfNodeP]);
        const size_t cCells(cId);
        const size_t pNodesOfCellC(
            utils::indexOf(mesh->getNodesOfCell(cId), pId));
        reduction0 =
            sumR0(reduction0, m_node_cellvolume_n(cCells, pNodesOfCellC));
      }
    }
    m_node_volume(pNodes) = reduction0;
  });
}
/**
 *******************************************************************************
 * \file updateVelocity()
 * \brief Calcul de la vitesse
 *
 * \param  gt->deltat_nplus1, gt->deltat_n, m_node_velocity_n
 *         m_pressure_n, m_pseudo_viscosity_n, m_cqs
 * \return m_node_velocity_nplus1, m_x_velocity, m_y_velocity
 *******************************************************************************
 */
void Vnr::updateVelocity() noexcept {
  const double dt(0.5 * (gt->deltat_nplus1 + gt->deltat_n));
  {
    const auto innerNodes(mesh->getInnerNodes());
    const size_t nbInnerNodes(mesh->getNbInnerNodes());
    Kokkos::parallel_for(
        nbInnerNodes, KOKKOS_LAMBDA(const size_t& pInnerNodes) {
          const Id pId(innerNodes[pInnerNodes]);
          const size_t pNodes(pId);
          RealArray1D<2> reduction0({0.0, 0.0});
          {
            const auto cellsOfNodeP(mesh->getCellsOfNode(pId));
            const size_t nbCellsOfNodeP(cellsOfNodeP.size());
            for (size_t cCellsOfNodeP = 0; cCellsOfNodeP < nbCellsOfNodeP;
                 cCellsOfNodeP++) {
              const Id cId(cellsOfNodeP[cCellsOfNodeP]);
              const size_t cCells(cId);
              const size_t pNodesOfCellC(
                  utils::indexOf(mesh->getNodesOfCell(cId), pId));
              reduction0 = sumR1(reduction0, (m_pressure_n(cCells) +
                                              m_pseudo_viscosity_n(cCells)) *
                                                 m_cqs(cCells, pNodesOfCellC));
            }
          }
          m_node_velocity_nplus1(pNodes) =
              m_node_velocity_n(pNodes) + dt / m_node_mass(pNodes) * reduction0;
          m_x_velocity(pNodes) = m_node_velocity_nplus1(pNodes)[0];
          m_y_velocity(pNodes) = m_node_velocity_nplus1(pNodes)[1];
        });
  }
}
/**
 *******************************************************************************
 * \file updateVelocityWithoutLagrange()
 * \brief Calcul de la vitesse pour les cas d'advection pure
 *
 * \param  gt->t_nplus1, m_node_velocity_n0, m_node_velocity_n, m_node_coord_n
 * \return m_node_velocity_nplus1
 *******************************************************************************
 */
void Vnr::updateVelocityWithoutLagrange() noexcept {
  deep_copy(m_node_velocity_nplus1, m_node_velocity_n);
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    if (test->Nom == test->RiderVortexTimeReverse ||
        test->Nom == test->MonoRiderVortexTimeReverse ||
        test->Nom == test->RiderDeformationTimeReverse ||
        test->Nom == test->MonoRiderDeformationTimeReverse) {
      m_node_velocity_nplus1(pNodes)[0] =
          init->m_node_velocity_n0(pNodes)[0] * cos(Pi * gt->t_nplus1 / 4.);
      m_node_velocity_nplus1(pNodes)[1] =
          init->m_node_velocity_n0(pNodes)[1] * cos(Pi * gt->t_nplus1 / 4.);
    }
  });
}
/**
 *******************************************************************************
 * \file updatePosition()
 * \brief Calcul de la position
 *
 * \param  gt->t_nplus1, m_node_velocity_nplus1, m_node_coord_n
 * \return m_node_coord_nplus1
 *******************************************************************************
 */
void Vnr::updatePosition() noexcept {
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    m_node_coord_nplus1(pNodes) =
        m_node_coord_n(pNodes) +
        gt->deltat_nplus1 * m_node_velocity_nplus1(pNodes);
  });
}
/**
 *******************************************************************************
 * \file updateCellPos()
 * \brief Calcul de la position du centre de la maille
 *
 * \param  m_node_coord_nplus1
 * \return m_cell_coord_nplus1
 *******************************************************************************
 */
void Vnr::updateCellPos() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    const Id cId(cCells);
    RealArray1D<2> reduction0({0.0, 0.0});
    {
      const auto nodesOfCellC(mesh->getNodesOfCell(cId));
      const size_t nbNodesOfCellC(nodesOfCellC.size());
      for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
           pNodesOfCellC++) {
        const Id pId(nodesOfCellC[pNodesOfCellC]);
        const size_t pNodes(pId);
        reduction0 = sumR1(reduction0, m_node_coord_nplus1(pNodes));
      }
    }
    m_cell_coord_nplus1(cCells) = 0.25 * reduction0;
  });
}
/**
 *******************************************************************************
 * \file computeSubVol()
 * \brief Calcul des sous-volumes aux noeuds de chaque maille
 *
 * \param  m_node_coord_nplus1
 * \return m_node_cellvolume_nplus1
 *******************************************************************************
 */
void Vnr::computeSubVol() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    const Id cId(cCells);
    {
      const auto nodesOfCellC(mesh->getNodesOfCell(cId));
      const size_t nbNodesOfCellC(nodesOfCellC.size());
      for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
           pNodesOfCellC++) {
        const Id pMinus1Id(
            nodesOfCellC[(pNodesOfCellC - 1 + nbNodesOfCell) % nbNodesOfCell]);
        const Id pId(nodesOfCellC[pNodesOfCellC]);
        const Id pPlus1Id(
            nodesOfCellC[(pNodesOfCellC + 1 + nbNodesOfCell) % nbNodesOfCell]);
        const size_t pMinus1Nodes(pMinus1Id);
        const size_t pNodes(pId);
        const size_t pPlus1Nodes(pPlus1Id);
        const RealArray1D<2> x1(m_cell_coord_nplus1(cCells));
        const RealArray1D<2> x2(0.5 * (m_node_coord_nplus1(pMinus1Nodes) +
                                       m_node_coord_nplus1(pNodes)));
        const RealArray1D<2> x3(m_node_coord_nplus1(pNodes));
        const RealArray1D<2> x4(0.5 * (m_node_coord_nplus1(pPlus1Nodes) +
                                       m_node_coord_nplus1(pNodes)));
        m_node_cellvolume_nplus1(cCells, pNodesOfCellC) =
            0.5 * (crossProduct2d(x1, x2) + crossProduct2d(x2, x3) +
                   crossProduct2d(x3, x4) + crossProduct2d(x4, x1));
      }
    }
  });
}
/**
 *******************************************************************************
 * \file updateRho()
 * \brief Calcul du volume lagrange et de la densité
 *
 * \param  m_node_cellvolume_nplus1, m_cell_mass_env, m_fracvol_env
 * \return varlp->vLagrange, m_density_nplus1, m_density_env_nplus1
 *******************************************************************************
 */
void Vnr::updateRho() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    const Id cId(cCells);
    double reduction0(0.0);
    {
      const auto nodesOfCellC(mesh->getNodesOfCell(cId));
      const size_t nbNodesOfCellC(nodesOfCellC.size());
      for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
           pNodesOfCellC++) {
        reduction0 =
            sumR0(reduction0, m_node_cellvolume_nplus1(cCells, pNodesOfCellC));
      }
    }
    varlp->vLagrange(cCells) = reduction0;
    m_density_nplus1(cCells) = 0.;
    for (int imat = 0; imat < options->nbmat; ++imat) {
      if (m_fracvol_env(cCells)[imat] > options->threshold)
        m_density_env_nplus1(cCells)[imat] =
            m_cell_mass_env(cCells)[imat] /
            (m_fracvol_env(cCells)[imat] * reduction0);
      // ou 1/rhon_nplus1 += m_mass_fraction_env(cCells)[imat] /
      // m_density_env_nplus1[imat];
      m_density_nplus1(cCells) +=
          m_fracvol_env(cCells)[imat] * m_density_env_nplus1(cCells)[imat];
    }
  });
}
/**
 *******************************************************************************
 * \file computeTau()
 * \brief Calcul de la variation du volume specifique (variation de 1/densite)
 *        en moyenne et pour chaque materiau
 *
 * \param  m_density_env_nplus1, m_density_env_n, m_density_nplus1, m_density_n
 * \return m_tau_density_nplus1, m_tau_density_env_nplus1
 *******************************************************************************
 */
void Vnr::computeTau() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    m_tau_density_nplus1(cCells) =
        0.5 * (1.0 / m_density_nplus1(cCells) + 1.0 / m_density_n(cCells));
    for (int imat = 0; imat < options->nbmat; ++imat) {
      m_tau_density_env_nplus1(cCells)[imat] = 0.;
      if ((m_density_env_nplus1(cCells)[imat] > options->threshold) &&
          (m_density_env_n(cCells)[imat] > options->threshold))
        m_tau_density_env_nplus1(cCells)[imat] =
            0.5 * (1.0 / m_density_env_nplus1(cCells)[imat] +
                   1.0 / m_density_env_n(cCells)[imat]);
    }
  });
}
/**
 *******************************************************************************
 * \file updateEnergy()
 * \brief Calcul de l'energie interne (seul le cas du gaz parfait est codé)
 *
 * \param  m_density_env_nplus1, m_density_env_n,
 *         m_pseudo_viscosity_env_nplus1, m_pseudo_viscosity_env_n
 *         m_pressure_env_n, m_mass_fraction_env
 *
 * \return m_internal_energy_env_nplus1, m_internal_energy_nplus1
 *******************************************************************************
 */
void Vnr::updateEnergy() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    m_internal_energy_nplus1(cCells) = 0.;
    for (int imat = 0; imat < options->nbmat; ++imat) {
      m_internal_energy_env_nplus1(cCells)[imat] = 0.;
      if ((m_density_env_nplus1(cCells)[imat] > options->threshold) &&
          (m_density_env_n(cCells)[imat] > options->threshold)) {
        // calcul du DV a changer utiliser divU
        double pseudo(0.);
        if ((options->pseudo_centree == 1) &&
            ((m_pseudo_viscosity_env_nplus1(cCells)[imat] +
              m_pseudo_viscosity_env_n(cCells)[imat]) *
                 (1.0 / m_density_env_nplus1(cCells)[imat] -
                  1.0 / m_density_env_n(cCells)[imat]) >
             0.)) {
          pseudo = 0.5 * (m_pseudo_viscosity_env_nplus1(cCells)[imat] +
                          m_pseudo_viscosity_env_n(cCells)[imat]);
        }
        if (options->pseudo_centree ==
            0) {  // test sur la positivité du travail dans le calcul de
                  // m_pseudo_viscosity_nplus1(cCells)
          pseudo = m_pseudo_viscosity_env_nplus1(cCells)[imat];
        }
        const double den(1 + 0.5 * (eos->gamma[imat] - 1.0) *
                                 m_density_env_nplus1(cCells)[imat] *
                                 (1.0 / m_density_env_nplus1(cCells)[imat] -
                                  1.0 / m_density_env_n(cCells)[imat]));
        const double num(m_internal_energy_env_n(cCells)[imat] -
                         (0.5 * m_pressure_env_n(cCells)[imat] + pseudo) *
                             (1.0 / m_density_env_nplus1(cCells)[imat] -
                              1.0 / m_density_env_n(cCells)[imat]));
        m_internal_energy_env_nplus1(cCells)[imat] = num / den;
        m_internal_energy_nplus1(cCells) +=
            m_mass_fraction_env(cCells)[imat] *
            m_internal_energy_env_nplus1(cCells)[imat];
      }
    }
  });
}
/**
 *******************************************************************************
 * \file computeDivU()
 * \brief Calcul de la divergence de la vitesse
 *
 * \param  m_density_nplus1, m_density_n, gt->deltat_nplus1,
 *m_tau_density_nplus1 \return m_divu_nplus1
 *******************************************************************************
 */
void Vnr::computeDivU() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    m_divu_nplus1(cCells) =
        1.0 / gt->deltat_nplus1 *
        (1.0 / m_density_nplus1(cCells) - 1.0 / m_density_n(cCells)) /
        m_tau_density_nplus1(cCells);
    // a changer comme le calcul du DV, utiliser les
    // m_cqs(cCells,pNodesOfCellC)
  });
}
/**
 *******************************************************************************
 * \file computeEOS()
 * \brief Appel aux differentes équations d'état
 *
 * \param  m_density_env_nplus1, m_internal_energy_env_nplus1
 *         eos->gamma, eos->tension_limit
 * \return m_pressure_env_nplus1, m_speed_velocity_env_nplus1
 *******************************************************************************
 */
void Vnr::computeEOS() {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    for (int imat = 0; imat < options->nbmat; ++imat) {
      double pression;  // = m_pressure_env_nplus1(cCells)[imat];
      double density = m_density_env_nplus1(cCells)[imat];
      double energy = m_internal_energy_env_nplus1(cCells)[imat];
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
      //
      m_pressure_env_nplus1(cCells)[imat] = sortie_eos[0];
      m_speed_velocity_env_nplus1(cCells)[imat] = sortie_eos[1];
    }
  });
}
/**
 *******************************************************************************
 * \file computePressionMoyenne()
 * \brief Calcul de la pression moyenne
 *
 * \param  m_fracvol_env, m_pressure_env_nplus1, m_speed_velocity_env_nplus1
 * \return m_pressure_nplus1, m_speed_velocity_nplus1
 *******************************************************************************
 */
void Vnr::computePressionMoyenne() noexcept {
  for (int cCells = 0; cCells < nbCells; cCells++) {
    m_pressure_nplus1(cCells) = 0.;
    for (int imat = 0; imat < options->nbmat; ++imat) {
      m_pressure_nplus1(cCells) +=
          m_fracvol_env(cCells)[imat] * m_pressure_env_nplus1(cCells)[imat];
      m_speed_velocity_nplus1(cCells) =
          MathFunctions::max(m_speed_velocity_nplus1(cCells),
                             m_speed_velocity_env_nplus1(cCells)[imat]);
    }
    // NONREG GP A SUPPRIMER
    if (m_density_nplus1(cCells) > 0.) {
      m_speed_velocity_nplus1(cCells) =
          std::sqrt(eos->gamma[0] * (eos->gamma[0] - 1.0) *
                    m_internal_energy_nplus1(cCells));
    }
    for (int imat = 0; imat < options->nbmat; ++imat)
      if (eos->Nom[imat] == eos->Void)
        m_internal_energy_nplus1(cCells) +=
            m_mass_fraction_env(cCells)[imat] *
            m_internal_energy_env_nplus1(cCells)[imat];
  }
}
