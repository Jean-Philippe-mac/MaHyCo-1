#include "Vnr.h"

using namespace nablalib;

#include <iomanip>  // for operator<<, setw, setiosflags

#include "../includes/Freefunctions.h"
#include "utils/Utils.h"  // for __RESET__, __BOLD__, __GREEN__

/**
 *******************************************************************************
 * \file computeDeltaT()
 * \brief Calcul du pas de temps
 *
 * \param  m_speed_velocity_n, m_node_velocity_n, gt->deltat_n
 * \return gt->deltat_nplus1
 *******************************************************************************
 */
void Vnr::computeDeltaT() noexcept {
  double reduction0;
  double Aveccfleuler(0.);
  double cfl(0.1);
  if (options->AvecProjection == 1) {
    // cfl euler
    Aveccfleuler = 1;
    cfl = 0.05;  // explication à trouver, permet de passer les cas euler ?
  }
  Kokkos::parallel_reduce(
      nbCells,
      KOKKOS_LAMBDA(const size_t& cCells, double& accu) {
        const Id cId(cCells);
        double uc(0.0);
        double reduction1(0.0);
        {
          const auto nodesOfCellC(mesh->getNodesOfCell(cId));
          const size_t nbNodesOfCellC(nodesOfCellC.size());
          for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
               pNodesOfCellC++) {
            int pId(nodesOfCellC[pNodesOfCellC]);
            int pNodes(pId);
            reduction1 =
                sumR0(reduction1, m_node_cellvolume_n(cCells, pNodesOfCellC));
            uc += std::sqrt(m_node_velocity_n(pNodes)[0] *
                                m_node_velocity_n(pNodes)[0] +
                            m_node_velocity_n(pNodes)[1] *
                                m_node_velocity_n(pNodes)[1]) *
                  0.25;
          }
        }
        // 0.05 a expliquer
        accu =
            minR0(accu, cfl * std::sqrt(reduction1) /
                            (Aveccfleuler * uc + m_speed_velocity_n(cCells)));
      },
      KokkosJoiner<double>(reduction0, numeric_limits<double>::max(), &minR0));
  
  gt->deltat_nplus1 = std::min(reduction0, 1.05 * gt->deltat_n);
}
/**
 *******************************************************************************
 * \file computeTime()
 * \brief Calcul du temps courant
 *
 * \param  gt->t_n, gt->deltat_nplus1
 * \return gt->t_nplus1
 *******************************************************************************
 */
void Vnr::computeTime() noexcept { gt->t_nplus1 = gt->t_n + gt->deltat_nplus1; }
/**
 *******************************************************************************
 * \file computeDeltaTinit()
 * \brief Calcul du pas de temps initial
 *
 * \param  m_speed_velocity_n0
 * \return gt->deltat_init
 *******************************************************************************
 */
void Vnr::computeDeltaTinit() noexcept {
  double reduction0;
  Kokkos::parallel_reduce(
      nbCells,
      KOKKOS_LAMBDA(const size_t& cCells, double& accu) {
        const Id cId(cCells);
        double reduction1(0.0);
        {
          const auto nodesOfCellC(mesh->getNodesOfCell(cId));
          const size_t nbNodesOfCellC(nodesOfCellC.size());
          for (size_t pNodesOfCellC = 0; pNodesOfCellC < nbNodesOfCellC;
               pNodesOfCellC++) {
            reduction1 = sumR0(
                reduction1, init->m_node_cellvolume_n0(cCells, pNodesOfCellC));
          }
        }
        accu = minR0(accu, 0.1 * std::sqrt(reduction1) /
                               init->m_speed_velocity_n0(cCells));
      },
      KokkosJoiner<double>(reduction0, numeric_limits<double>::max(), &minR0));
  gt->deltat_init = reduction0 * 1.0E-6;
}
/**
 *******************************************************************************
 * \file computeVariablesGlobalesInit()
 * \brief Calcul de l'energie totale et la masse initiale du systeme
 *
 * \param  m_cell_velocity_n0, m_density_n0, m_euler_volume
 * \return m_total_energy_0, m_global_masse_0,
 *         m_global_total_energy_0, m_total_masse_0
 *
 *******************************************************************************
 */
void Vnr::computeVariablesGlobalesInit() noexcept {
  m_global_total_energy_0 = 0.;
  Kokkos::parallel_for(
      "init_m_global_total_var_0", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        m_total_energy_0(cCells) =
	  m_cell_mass(cCells) *
            (init->m_internal_energy_n0(cCells) +
             0.5 * (init->m_cell_velocity_n0(cCells)[0] *
                        init->m_cell_velocity_n0(cCells)[0] +
                    init->m_cell_velocity_n0(cCells)[1] *
                        init->m_cell_velocity_n0(cCells)[1]));
        m_total_masse_0(cCells) = m_cell_mass(cCells);
      });
  double reductionE(0.), reductionM(0.);
  {
    Kokkos::Sum<double> reducerE(reductionE);
    Kokkos::parallel_reduce("reductionE", nbCells,
                            KOKKOS_LAMBDA(const int& cCells, double& x) {
                              reducerE.join(x, m_total_energy_0(cCells));
                            },
                            reducerE);
    Kokkos::Sum<double> reducerM(reductionM);
    Kokkos::parallel_reduce("reductionM", nbCells,
                            KOKKOS_LAMBDA(const int& cCells, double& x) {
                              reducerM.join(x, m_total_masse_0(cCells));
                            },
                            reducerM);
  }
  m_global_total_energy_0 = reductionE;
  m_global_total_masse_0 = reductionM;
}
/**
 *******************************************************************************
 * \file computeVariablesSortiesInit()
 * \brief Calcul des variables initiales pour les sorties
 *
 * \param
 * \return m_fracvol_env1, m_fracvol_env2, m_fracvol_env3
 *         m_interface12, m_interface13, m_interface23
 *         m_x_velocity, m_y_velocity
 *******************************************************************************
 */
void Vnr::computeVariablesSortiesInit() noexcept {
  // pour les sorties au temps 0
  Kokkos::parallel_for("sortie", nbCells, KOKKOS_LAMBDA(const int& cCells) {
    // pour les sorties au temps 0:
    m_fracvol_env1(cCells) = init->m_fracvol_env_n0(cCells)[0];
    m_fracvol_env2(cCells) = init->m_fracvol_env_n0(cCells)[1];
    m_fracvol_env3(cCells) = init->m_fracvol_env_n0(cCells)[2];
    m_interface12(cCells) = m_fracvol_env1(cCells) * m_fracvol_env2(cCells);
    m_interface13(cCells) = m_fracvol_env1(cCells) * m_fracvol_env3(cCells);
    m_interface23(cCells) = m_fracvol_env2(cCells) * m_fracvol_env3(cCells);
  });
  Kokkos::parallel_for("sortie", nbNodes, KOKKOS_LAMBDA(const int& pNodes) {
    m_x_velocity(pNodes) = init->m_node_velocity_n0(pNodes)[0];
    m_y_velocity(pNodes) = init->m_node_velocity_n0(pNodes)[1];
  });
}
/**
 *******************************************************************************
 * \file setUpTimeLoopN()
 * \brief Initialisation de la boucle en temps
 *
 * \param  init->...
 * \return m_...
 *******************************************************************************
 */
void Vnr::setUpTimeLoopN() noexcept {
  deep_copy(m_node_coord_n, init->m_node_coord_n0);
  deep_copy(m_cell_coord_n, init->m_cell_coord_n0);
  deep_copy(m_euler_volume, init->m_euler_volume_n0);
  deep_copy(m_lagrange_volume_n, init->m_euler_volume_n0);
  deep_copy(m_density_n, init->m_density_n0);
  deep_copy(m_density_env_n, init->m_density_env_n0);
  deep_copy(m_internal_energy_n, init->m_internal_energy_n0);
  deep_copy(m_internal_energy_env_n, init->m_internal_energy_env_n0);
  deep_copy(m_node_velocity_n, init->m_node_velocity_n0);
  deep_copy(m_mass_fraction_env, init->m_mass_fraction_env_n0);
  deep_copy(m_fracvol_env, init->m_fracvol_env_n0);
  // specifiques à VNR
  deep_copy(m_node_cellvolume_n, init->m_node_cellvolume_n0);
  deep_copy(m_pressure_n, init->m_pressure_n0);
  deep_copy(m_pressure_env_n, init->m_pressure_env_n0);
  deep_copy(m_pseudo_viscosity_n, init->m_pseudo_viscosity_n0);
  deep_copy(m_pseudo_viscosity_env_n, init->m_pseudo_viscosity_env_n0);
  deep_copy(m_tau_density_n, init->m_tau_density_n0);
  deep_copy(m_tau_density_env_n, init->m_tau_density_env_n0);
  deep_copy(m_divu_n, init->m_divu_n0);
  deep_copy(m_speed_velocity_n, init->m_speed_velocity_n0);
  deep_copy(m_speed_velocity_env_n, init->m_speed_velocity_env_n0);
  // pas de temps
  gt->deltat_n = gt->deltat_init;
}
/**
 *******************************************************************************
 * \file executeTimeLoopN()
 * \brief Execution de la boucle en temps
 *
 *******************************************************************************
 */
void Vnr::executeTimeLoopN() noexcept {
  n = 0;
  bool continueLoop = true;
  do {
    global_timer.start();
    cpu_timer.start();
    n++;
    if (n != 1)
      std::cout << "[" << __CYAN__ << __BOLD__ << setw(3) << n
                << __RESET__ "] t = " << __BOLD__
                << setiosflags(std::ios::scientific) << setprecision(8)
                << setw(16) << gt->t_n << __RESET__;
    // Calcul de l'energie totale et la masse du systeme en debut de Lagrange
    computeVariablesGlobalesL0();
    
    if (options->sansLagrange == 0) {
      // calcul des m_cqs_n
      computeCornerNormal();
      // retour a la vitesse en n-1/2 : m_velocity_n
      if (scheme->schema == scheme->CSTS && options->AvecProjection == 1) updateVelocitybackward();
      // calcul du volume de chaque noeud du maillage : m_node_cellvolume_n
      computeNodeVolume();
      // Calcul du pas de temps : gt->deltat_nplus1
      computeDeltaT();
    } else {
      gt->deltat_nplus1 = gt->deltat_n;
    }
    // Calcul du temps courant : gt->t_nplus1
    computeTime();
    // Sorties 
    dumpVariables();

    if (options->sansLagrange == 0) {
      // Calcul de la vitesse en n+1/2 : m_velocity_nplus1
      updateVelocity();
      // Conditions aux limites de vitesses
      updateNodeBoundaryConditions();
    }
    // Calcul des positions en n+1 : m_node_coord_nplus1 
    updatePosition();
    // Calcul du centre des mailles : m_cell_coord_nplus1
    updateCellPos();
    // Calcul des sous-volumes aux noeuds : m_node_cellvolume_nplus1
    computeSubVol();
    // Calcul du volume lagrange et de la densité : m_density_nplus1
    updateRho();

    if (options->sansLagrange == 0) {
      // Calcul des m_cqs_nplus1
      updateCornerNormal();
      // Calcul de la variation du volume specifique
      computeTau();
      // Calcul de la divergence de la vitesse : m_divu_nplus1
      computeDivU();
      // Calcul de la viscosité artificielle : m_pseudo_viscosity_nplus1
      computeArtificialViscosity();
      // Calcul des corrections CSTS d'energie interne
      if (scheme->schema == scheme->CSTS) {
	// Calcul de l'energie interne
	updateEnergycqs();
	updateEnergyForTotalEnergyConservation();
      } else {
	// Calcul de l'energie interne
	updateEnergy();
      }	
      // Appel aux differentes équations d'état : m_pressure_env_nplus1
      computeEOS();
      // Calcul de la pression moyenne : m_pressure_nplus1
      computePressionMoyenne();
    } else {
      deep_copy(m_internal_energy_nplus1, m_internal_energy_n);
    }
    // Calcul des conditions aux limites dans les mailles
    updateCellBoundaryConditions();
    // Calcul des quantites apres la phase Lagrange
    computeVariablesGlobalesL();
    
    if (options->AvecProjection == 1) {
      // Calcul de la vitesse de n+1/2 a n+1 : m_node_velocity_nplus1
      if (scheme->schema == scheme->CSTS)
	updateVelocityforward();  
      // Remplissage des variables de la projection et de la projection duale
      computeVariablesForRemap();
      // Calcul du centre des mailles pour la projection
      computeCellQuantitesForRemap();
      // Calcul de quantites aux faces pour la projection
      computeFaceQuantitesForRemap();
      // phase de projection : premiere direction
      remap->computeGradPhiFace1();
      remap->computeGradPhi1();
      remap->computeUpwindFaceQuantitiesForProjection1();
      remap->computeUremap1();
      remap->computeDualUremap1();
      // phase de projection : seconde direction
      remap->computeGradPhiFace2();
      remap->computeGradPhi2();
      remap->computeUpwindFaceQuantitiesForProjection2();
      remap->computeUremap2();
      remap->computeDualUremap2();
      // Calcul des variables aux mailles et aux noeuds qui ont ete projetees
      remapVariables();
      if (options->sansLagrange == 1) {
        // les cas d'advection doivent etre à vitesses constantes
        updateVelocityWithoutLagrange();
      }
      // Calcul de la masse nodale : m_node_mass
      computeNodeMass();         // avec la masse des mailles recalculée dans
                                 // remapVariables()
      // Appel aux differentes équations d'état : m_pressure_env_nplus1
      computeEOS();              // rappel EOS apres projection
      // Calcul de la pression moyenne : m_pressure_nplus1
      computePressionMoyenne();  // rappel Pression moyenne apres projection     
      // Calcul des quantites apres la phase de projection
      computeVariablesGlobalesT();
    }
    // Evaluate loop condition with variables at time n
    continueLoop =
        (n + 1 < gt->max_time_iterations && gt->t_nplus1 < gt->final_time);
					 
    if (continueLoop) {
      // Switch variables to prepare next iteration
      std::swap(varlp->x_then_y_nplus1, varlp->x_then_y_n);
      std::swap(gt->t_nplus1, gt->t_n);
      std::swap(gt->deltat_nplus1, gt->deltat_n);
      std::swap(m_density_nplus1, m_density_n);
      std::swap(m_density_env_nplus1, m_density_env_n);
      std::swap(m_pressure_nplus1, m_pressure_n);
      std::swap(m_pressure_env_nplus1, m_pressure_env_n);
      std::swap(m_pseudo_viscosity_nplus1, m_pseudo_viscosity_n);
      std::swap(m_pseudo_viscosity_env_nplus1, m_pseudo_viscosity_env_n);
      std::swap(m_tau_density_nplus1, m_tau_density_n);
      std::swap(m_tau_density_env_nplus1, m_tau_density_env_n);
      std::swap(m_tau_volume_nplus1, m_tau_volume_n);
      std::swap(m_tau_volume_env_nplus1, m_tau_volume_env_n);
      std::swap(m_divu_nplus1, m_divu_n);
      std::swap(m_speed_velocity_nplus1, m_speed_velocity_n);
      std::swap(m_speed_velocity_env_nplus1, m_speed_velocity_env_n);
      std::swap(m_internal_energy_nplus1, m_internal_energy_n);
      std::swap(m_internal_energy_env_nplus1, m_internal_energy_env_n);
      std::swap(m_node_velocity_nplus1, m_node_velocity_n);
      if (options->AvecProjection == 0) {
        std::swap(m_cell_coord_nplus1, m_cell_coord_n);
        std::swap(m_node_cellvolume_nplus1, m_node_cellvolume_n);
        std::swap(m_node_coord_nplus1, m_node_coord_n);
	std::swap(m_lagrange_volume_nplus1, m_lagrange_volume_n);
      }
    }
    std::cout << "  "  << std::endl;
    ofstream fichierE(options->fichier_sortie1D, ios::app);  // ouverture en écriture avec effacement du fichier ouvert
    //ofstream fichierM("Mtotale.txt", ios::app);  // ouverture en écriture avec effacement du fichier ouvert
 
    if(fichierE)
        {
	  fichierE  << gt->t_n << " " << m_global_total_energy_L;	  
	  if (options->AvecProjection == 1)
	    fichierE  << " " << m_global_total_energy_T << std::endl;
	  else
	    fichierE  << std::endl;
	}
    // if(fichierM)
    //     {
    // 	  fichierM << gt->t_n << " " << m_global_total_masse_L;
	  
    // 	  if (options->AvecProjection == 1)
    // 	    fichierM  <<  " " << m_global_total_masse_T << std::endl;
    // 	  else
    // 	    fichierE  << std::endl;
    // 	}
											   
    // std::cout << " DT  = " << gt->deltat_nplus1 << std::endl;
    cpu_timer.stop();
    global_timer.stop();

    // Timers display
    if (!writer.isDisabled())
      std::cout << " {CPU: " << __BLUE__ << cpu_timer.print(true)
                << __RESET__ ", IO: " << __BLUE__ << io_timer.print(true)
                << __RESET__ "} ";
    else
      std::cout << " {CPU: " << __BLUE__ << cpu_timer.print(true)
                << __RESET__ ", IO: " << __RED__ << "none" << __RESET__ << "} ";

    // Progress
    std::cout << utils::progress_bar(n, gt->max_time_iterations, gt->t_n,
                                     gt->final_time, 25);
    std::cout << __BOLD__ << __CYAN__
              << utils::Timer::print(
                     utils::eta(n, gt->max_time_iterations, gt->t_n,
                                gt->final_time, gt->deltat_n, global_timer),
                     true)
              << __RESET__ << "\r";
    std::cout.flush();

    cpu_timer.reset();
    io_timer.reset();
  } while (continueLoop);
  // force a last output at the end
  dumpVariables();
}
/**
 *******************************************************************************
 * \file dumpVariables()
 * \brief Ecriture des sorties
 *
 *******************************************************************************
 */
void Vnr::dumpVariables() noexcept {
  nbCalls++;
  if (!writer.isDisabled() &&
      (gt->t_n >= lastDump + gt->output_time || gt->t_n == 0.)) {
    cpu_timer.stop();
    io_timer.start();
    std::map<string, double*> cellVariables;
    std::map<string, double*> nodeVariables;
    std::map<string, double*> partVariables;
    if (so->pression)
      cellVariables.insert(
          pair<string, double*>("Pressure", m_pressure_n.data()));
    if (so->densite)
      cellVariables.insert(
          pair<string, double*>("Density", m_density_n.data()));
    if (so->energie_interne)
      cellVariables.insert(
          pair<string, double*>("Energy", m_internal_energy_n.data()));
    if (options->nbmat > 1) {
      if (so->fraction_volumique)
        cellVariables.insert(
            pair<string, double*>("fracvol1", m_fracvol_env1.data()));
      if (so->interface)
        cellVariables.insert(
            pair<string, double*>("interface12", m_interface12.data()));
    }
    if (options->nbmat > 2 && options->AvecProjection == 1) {
      if (so->fraction_volumique) {
        cellVariables.insert(
            pair<string, double*>("fracvol2", m_fracvol_env2.data()));
        cellVariables.insert(
            pair<string, double*>("fracvol3", m_fracvol_env3.data()));
      }
      if (so->interface)
        cellVariables.insert(
            pair<string, double*>("interface23", m_interface23.data()));
      cellVariables.insert(
          pair<string, double*>("interface13", m_interface13.data()));
    }
    if (so->vitesse) {
      nodeVariables.insert(
          pair<string, double*>("VitesseX", m_x_velocity.data()));
      nodeVariables.insert(
          pair<string, double*>("VitesseY", m_y_velocity.data()));
    }
    auto quads = mesh->getGeometry()->getQuads();
    writer.writeFile(nbCalls, gt->t_n, nbNodes, m_node_coord_n.data(), nbCells,
                     quads.data(), cellVariables, nodeVariables);
    lastDump = gt->t_n;
    std::cout << " time = " << gt->t_n << " sortie demandée " << std::endl;
    io_timer.stop();
    cpu_timer.start();
  }
}
/**
 *******************************************************************************
 * \file simulate()
 * \brief programme principale de lancement de la simulation
 *
 *******************************************************************************
 */
void Vnr::simulate() {
  std::cout << "\n"
            << __BLUE_BKG__ << __YELLOW__ << __BOLD__ << "\tStarting Vnr ..."
            << __RESET__ << "\n\n";

  std::cout << "[" << __GREEN__ << "MESH" << __RESET__
            << "]      X=" << __BOLD__ << cstmesh->X_EDGE_ELEMS << __RESET__
            << ", Y=" << __BOLD__ << cstmesh->Y_EDGE_ELEMS << __RESET__
            << ", X length=" << __BOLD__ << cstmesh->X_EDGE_LENGTH << __RESET__
            << ", Y length=" << __BOLD__ << cstmesh->Y_EDGE_LENGTH << __RESET__
            << std::endl;

  if (Kokkos::hwloc::available()) {
    std::cout << "[" << __GREEN__ << "TOPOLOGY" << __RESET__
              << "]  NUMA=" << __BOLD__
              << Kokkos::hwloc::get_available_numa_count()

              << __RESET__ << ", Cores/NUMA=" << __BOLD__
              << Kokkos::hwloc::get_available_cores_per_numa() << __RESET__
              << ", Threads/Core=" << __BOLD__
              << Kokkos::hwloc::get_available_threads_per_core() << __RESET__
              << std::endl;
  } else {
    std::cout << "[" << __GREEN__ << "TOPOLOGY" << __RESET__
              << "]  HWLOC unavailable cannot get topological informations"
              << std::endl;
  }

  if (!writer.isDisabled())
    std::cout << "[" << __GREEN__ << "OUTPUT" << __RESET__
              << "]    VTK files stored in " << __BOLD__
              << writer.outputDirectory() << __RESET__ << " directory"
              << std::endl;
  else
    std::cout << "[" << __GREEN__ << "OUTPUT" << __RESET__ << "]    "
              << __BOLD__ << "Disabled" << __RESET__ << std::endl;

  init->initBoundaryConditions();
  init->initCellPos();
  init->initVar();
  init->initSubVol();
  init->initMeshGeometryForFaces();
  remap->FacesOfNode();

  if (options->sansLagrange == 0) {
    init->initPseudo();
    computeDeltaTinit();
  }

  setUpTimeLoopN();
  computeCellMass();
  computeNodeMass();
  computeVariablesGlobalesInit();
  computeVariablesSortiesInit();
  executeTimeLoopN();

  std::cout << __YELLOW__ << "\n\tDone ! Took " << __MAGENTA__ << __BOLD__
            << global_timer.print() << __RESET__ << std::endl;
}
