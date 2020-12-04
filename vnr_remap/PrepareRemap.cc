#include "Vnr.h"

using namespace nablalib;

#include "../includes/Freefunctions.h"
#include "types/MathFunctions.h"  // for max, min, dot, matVectProduct
#include "utils/Utils.h"          // for Indexof

/**
 *******************************************************************************
 * \file computeFaceQuantitesForRemap()
 * \brief Calcul de quantites aux faces pour la projection :
 *    DxLagrange, du milieu, de la longueur des faces et de leur vitesse normale
 *
 * \param  varlp->XcLagrange, varlp->XLagrange, varlp->faceNormal
 *         m_node_velocity_nplus1
 * \return varlp->deltaxLagrange, varlp->XfLagrange, varlp->faceLengthLagrange,
 *         varlp->faceNormalVelocity
 *******************************************************************************
 */
void Vnr::computeFaceQuantitesForRemap() noexcept {
  auto Innerfaces(mesh->getInnerFaces());
  int nbInnerFaces(mesh->getNbInnerFaces());
  Kokkos::parallel_for(
      "computeFacedeltaxLagrange", nbInnerFaces,
      KOKKOS_LAMBDA(const int& fFaces) {
        size_t fId(Innerfaces[fFaces]);
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
  auto faces(mesh->getFaces());
  Kokkos::parallel_for(
      "computeFaceQuantities", nbFaces, KOKKOS_LAMBDA(const int& fFaces) {
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

        RealArray1D<dim> reduction5 = zeroVect;
        {
          auto nodesOfFaceF(mesh->getNodesOfFace(fId));
          for (int pNodesOfFaceF = 0; pNodesOfFaceF < nodesOfFaceF.size();
               pNodesOfFaceF++) {
            int pId(nodesOfFaceF[pNodesOfFaceF]);
            int pNodes(pId);
	    if (scheme->schema == scheme->CSTS) {
	      reduction5 = reduction5 + .5 * (m_node_velocity_nplus1(pNodes) + m_node_velocity_n(pNodes));
	    } else
              reduction5 = reduction5 + (m_node_velocity_nplus1(pNodes)); // vitesse VNR en fait en N+1/2
	      
          }
        }

        varlp->faceNormalVelocity(fFaces) =
            dot((0.5 * reduction5), varlp->faceNormal(fFaces));
      });
}
/**
 *******************************************************************************
 * \file computeCellQuantitesForRemap()
 * \brief Calcul du centre des mailles pour la projection
 *
 * \param  varlp->XLagrange
 * \return varlp->XcLagrange
 *******************************************************************************
 */
void Vnr::computeCellQuantitesForRemap() noexcept {
  Kokkos::parallel_for(
      "computeLagrangeVolumeAndCenterOfGravity", nbCells,
      KOKKOS_LAMBDA(const int& cCells) {
        int cId(cCells);
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
        varlp->XcLagrange(cCells) =
            (1.0 / (6.0 * varlp->vLagrange(cCells)) * reduction7);
      });
}
/**
 *******************************************************************************
 * \file computeVariablesForRemap()
 * \brief Remplissage des variables de la projection et de la projection duale
 *         varlp->ULagrange (variables aux mailles)
 *                           de 0 à nbmat-1 : volume partiel,
 *                           de nbmat à 2*nbmat-1 : masse partielle
 *                           de 2*nbmat à 3*nbmat-1 : energie partielle
 *                           de 3*nbmat à 3*nbmat+1 : quantite de mouvement
 *                           3*nbmat+2 : enegie cinetique
 *                           3*nbmat+3 : pseudo-viscosite * volume
 *
 *         varlp->UDualLagrange (variables aux noeuds)
 *                           0 : masse
 *                           1 à 2 : quantite de mouvement
 *                           3 : energie cinetique
 *
 *  Pour l'option projection avec limiteurs pente-borne
 *
 *         varlp->Phi (variables aux mailles)
 *                           de 0 à nbmat-1 : fraction volumique
 *                           de nbmat à 2*nbmat-1 : densite partielle
 *                           de 2*nbmat à 3*nbmat-1 : energie specifique
 *partielle de 3*nbmat à 3*nbmat+1 : vitesse 3*nbmat+2 : enegie cinetique
 *specifique 3*nbmat+3 : pseudo-viscosite
 *
 *         varlp->DualPhi (variables aux noeuds)
 *                           0 : densite moyenne
 *                           1 à 2 : vitesse
 *                           3 : energie cinetique specifique
 * \param m_fracvol_env, varlp->vLagrange, m_mass_fraction_env, m_density_nplus1
 *        m_internal_energy_nplus1,
 * \return varlp->ULagrange, varlp->UDualLagrange, varlp->Phi, varlp->DualPhi
 *******************************************************************************
 */
void Vnr::computeVariablesForRemap() noexcept {
  Kokkos::parallel_for(
      "computeVariablesForRemap", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        int nbmat = options->nbmat;
        for (int imat = 0; imat < nbmat; imat++) {
          // volumes matériels (partiels)
          varlp->ULagrange(cCells)[imat] =
              m_fracvol_env(cCells)[imat] * varlp->vLagrange(cCells);

          // masses matériels (partiels)
          varlp->ULagrange(cCells)[nbmat + imat] =
              m_mass_fraction_env(cCells)[imat] * varlp->vLagrange(cCells) *
              m_density_nplus1(cCells);

          // energies matériels (partiels)
          varlp->ULagrange(cCells)[2 * nbmat + imat] =
              m_mass_fraction_env(cCells)[imat] * varlp->vLagrange(cCells) *
              m_density_nplus1(cCells) * m_internal_energy_nplus1(cCells);
        }
        varlp->ULagrange(cCells)[3 * nbmat + 0] = 0.;
        varlp->ULagrange(cCells)[3 * nbmat + 1] = 0.;
        varlp->ULagrange(cCells)[3 * nbmat + 2] = 0.;

        // Pseudo moyenne pour la quantité de mouvement
        varlp->ULagrange(cCells)[3 * nbmat + 3] =
            m_pseudo_viscosity_nplus1(cCells) * varlp->vLagrange(cCells);

        if (limiteurs->projectionAvecPlateauPente == 1) {
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
          // Phi energie
          for (int imat = 0; imat < nbmat; imat++) {
            if (varlp->ULagrange(cCells)[nbmat + imat] != 0.)
              varlp->Phi(cCells)[2 * nbmat + imat] =
                  varlp->ULagrange(cCells)[2 * nbmat + imat] /
                  varlp->ULagrange(cCells)[nbmat + imat];
            else
              varlp->Phi(cCells)[2 * nbmat + imat] = 0.;
          }
          // les phi sur la vitesse et energie cinétique n'existent pas en VnR
          varlp->Phi(cCells)[3 * nbmat + 0] = 0.;
          varlp->Phi(cCells)[3 * nbmat + 1] = 0.;
          varlp->Phi(cCells)[3 * nbmat + 2] = 0.;
          // Phi Pseudo
          varlp->Phi(cCells)[3 * nbmat + 3] = m_pseudo_viscosity_nplus1(cCells);
        } else {
          varlp->Phi(cCells) =
              varlp->ULagrange(cCells) / varlp->vLagrange(cCells);
        }
        varlp->rLagrange(cCells) = m_density_nplus1(cCells);
      });

  // variables duales
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    // Position fin de phase Lagrange
    varlp->XLagrange(pNodes) = m_node_coord_nplus1(pNodes);
    // quantité de mouvement
    varlp->UDualLagrange(pNodes)[0] =
        m_node_mass(pNodes) * m_node_velocity_nplus1(pNodes)[0];
    varlp->UDualLagrange(pNodes)[1] =
        m_node_mass(pNodes) * m_node_velocity_nplus1(pNodes)[1];
    // projection de l'energie cinétique
    if (options->projectionConservative == 1)
      varlp->UDualLagrange(pNodes)[2] =
          0.5 * m_node_mass(pNodes) * (m_node_velocity_nplus1(pNodes)[0] *
				       m_node_velocity_nplus1(pNodes)[0] +
				       m_node_velocity_nplus1(pNodes)[1] *
				       m_node_velocity_nplus1(pNodes)[1]);
    // masse nodale
    varlp->UDualLagrange(pNodes)[3] = m_node_mass(pNodes);
    
    // *** variables Phi
    varlp->DualPhi(pNodes)[0] = m_node_velocity_nplus1(pNodes)[0];
    varlp->DualPhi(pNodes)[1] = m_node_velocity_nplus1(pNodes)[1];
    // Phi energie cinétique
    if (options->projectionConservative == 1)
      varlp->DualPhi(pNodes)[2] = 0.5*(m_node_velocity_nplus1(pNodes)[0] *
                                       m_node_velocity_nplus1(pNodes)[0] +
				       m_node_velocity_nplus1(pNodes)[1] *
                                       m_node_velocity_nplus1(pNodes)[1]);
    // masse nodale
    varlp->DualPhi(pNodes)[3] = m_node_mass(pNodes);
  });
}
