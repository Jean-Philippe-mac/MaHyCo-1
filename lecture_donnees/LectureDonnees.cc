#include <stdlib.h>  // for std

#include <fstream>   // for ifstream
#include <iostream>  // for operator<<, endl, basic_o...
#include "LectureDonnees.h"
/**
 *******************************************************************************
 * \file LectureDonnees(..)
 * \brief Lecture du fichier de donnees
 *
 * \param   Fichier donnees en argument
 * \return SchemaLagrange* s : schéma du cas
 *         Options* o : options du cas
 *         SortieVariables* so : variables à sortir du cas
 *         ConstantesMaillages* cstmesh : parametres du maillages
 *         GestTemps* gt : gestion du temps ou du pas de temps
 *         Limiteurs* l : limiteurs des projections (pente-borne, superbee...)
 *         EquationDetat* eos : equations d'etat des matériaux
 *         Test* test : nom du cas test
 *******************************************************************************
 */
void LectureDonneesClass::LectureDonnees(
    string Fichier, schemalagrangelib::SchemaLagrangeClass::SchemaLagrange* s,
    optionschemalib::OptionsSchema::Options* o,
    sortielib::Sortie::SortieVariables* so,
    cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh,
    gesttempslib::GestionTempsClass::GestTemps* gt,
    limiteurslib::LimiteursClass::Limiteurs* l, eoslib::EquationDetat* eos,
    castestlib::CasTest::Test* test) {
  // string Fichier=argv[1];
  ifstream mesdonnees(Fichier);  // Ouverture d'un fichier en lecture
  if (mesdonnees) {
    // Tout est prêt pour la lecture.
    string ligne;
    char motcle[30];
    char valeur[30];
    char variables[30];
    int entier;

    while (getline(mesdonnees, ligne))  // Tant qu'on n'est pas à la fin, on lit
    {
      mesdonnees >> motcle;
      std::cout << " lecture de " << motcle << std::endl;
      if (!strcmp(motcle, "CAS")) {
        mesdonnees >> valeur;
        test->Nom = castestToOptions[valeur];
        std::cout << " Cas test " << valeur << " ( " << test->Nom << " ) "
                  << std::endl;
	o->fichier_sortie1D+=valeur;
        o->fichier_sortie1D+="-";
        mesdonnees.ignore();

        // on en deduit le nombre de materiaux du calcul
        if (test->Nom < test->BiUnitTestCase) {
          o->nbmat = 1;
        } else {
          o->nbmat = 2;
        }
        if (test->Nom == test->BiTriplePoint) o->nbmat = 3;

        if (test->Nom == test->AdvectionX || test->Nom == test->AdvectionY ||
            test->Nom == test->BiAdvectionX ||
            test->Nom == test->BiAdvectionY ||
            test->Nom == test->AdvectionVitX ||
            test->Nom == test->AdvectionVitY ||
            test->Nom == test->BiAdvectionVitX ||
            test->Nom == test->BiAdvectionVitY ||
            test->Nom >= test->MonoRiderTx) {
          o->sansLagrange = 1;
          std::cout << " Cas test d'advection pure " << std::endl;
        }

        if (test->Nom == test->Implosion) o->nbmat = 2;
        if (test->Nom == test->BiImplosion) o->nbmat = 3;
        if (test->Nom >= test->MonoRiderTx && test->Nom < test->RiderTx) {
          o->nbmat = 1;
        }
        if (o->nbmat == 1) {
          std::cout << " Cas test mono materiau " << std::endl;
	  o->fichier_sortie1D+="Mono-";
        } else if (o->nbmat == 2) {
          std::cout << " Cas test bi materiau " << std::endl;
	  o->fichier_sortie1D+="Bi-";
	}
      }

      else if (!strcmp(motcle, "NX")) {
        mesdonnees >> cstmesh->X_EDGE_ELEMS;
        std::cout << " Nombre de Mailles en X " << cstmesh->X_EDGE_ELEMS
                  << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "NY")) {
        mesdonnees >> cstmesh->Y_EDGE_ELEMS;
        std::cout << " Nombre de Mailles en Y " << cstmesh->Y_EDGE_ELEMS
                  << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "DELTA_X")) {
        mesdonnees >> cstmesh->X_EDGE_LENGTH;
        std::cout << " DELTA X " << cstmesh->X_EDGE_LENGTH << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "DELTA_Y")) {
        mesdonnees >> cstmesh->Y_EDGE_LENGTH;
        std::cout << " DELTA Y " << cstmesh->Y_EDGE_LENGTH << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "CYLINDRIC")) {
        mesdonnees >> valeur;
        cstmesh->cylindrical_mesh = ouiOUnon[valeur];
        std::cout << "Utilisation d'un maille cylindrique" << std::endl;
      }

      else if (!strcmp(motcle, "R_MIN")) {
        mesdonnees >> cstmesh->minimum_radius;
        std::cout << " RAYON-MINIMUM " << cstmesh->minimum_radius << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "TFIN")) {
        mesdonnees >> gt->final_time;
        std::cout << " Temps-final " << gt->final_time << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "T_SORTIE")) {
        mesdonnees >> gt->output_time;
        std::cout << " Temps entre deux sorties " << gt->output_time
                  << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "DTDEBUT")) {
        mesdonnees >> gt->deltat_init;
        std::cout << " Pas de temps initial " << gt->deltat_init << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "DTMIN")) {
        mesdonnees >> gt->deltat_min;
        std::cout << " Pas de temps minimal " << gt->deltat_min << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "EOS")) {
        for (int imat = 0; imat < o->nbmat; ++imat) {
          mesdonnees >> valeur;
          eos->Nom[imat] = liste_eos[valeur];
          std::cout << " Equation d'etat " << valeur << " ( " << eos->Nom[imat]
                    << " ) " << std::endl;
          mesdonnees.ignore();
        }
      }

      else if (!strcmp(motcle, "SCHEMA_LAGRANGE")) {
        mesdonnees >> valeur;
        s->schema = schema_lagrange[valeur];
        std::cout << " schema Lagrange " << valeur << " ( " << s->schema
                  << " ) " << std::endl;
	o->fichier_sortie1D+=valeur;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "ORDRE_SCHEMA_LAGRANGE")) {
        mesdonnees >> o->spaceOrder;
        std::cout << "Ordre du schéma Lagrange" << o->spaceOrder << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "MODE_EULER")) {
        mesdonnees >> valeur;
        o->AvecProjection = ouiOUnon[valeur];
        std::cout << " Avec Projection " << valeur << " ( " << o->AvecProjection
                  << " ) " << std::endl;
        mesdonnees.ignore();
	if (o->AvecProjection) o->fichier_sortie1D+="-euler";
      }

      else if (!strcmp(motcle, "PROJECTION_CONSERVATIVE")) {
        mesdonnees >> valeur;
        o->projectionConservative = ouiOUnon[valeur];
        std::cout << " Projection conservative " << valeur << " ( "
                  << o->projectionConservative << " ) " << std::endl;
        mesdonnees.ignore();
	if (o->projectionConservative) o->fichier_sortie1D+="-conservatif";
	std::cout << "fichier de sortie : " << o->fichier_sortie1D << std::endl;
      }

      else if (!strcmp(motcle, "LIMITE_FRACTION_VOLUMIQUE")) {
        mesdonnees >> o->threshold;
        std::cout << " Limite de fraction volumique des mailles mixtes  "
                  << o->threshold << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "ORDRE_PROJECTION")) {
        mesdonnees >> o->projectionOrder;
        std::cout << " Ordre de la phase de projection  " << o->projectionOrder
                  << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "LIMITEURS_PROJECTION")) {
        mesdonnees >> valeur;
        l->projectionLimiterId = limiteur[valeur];
        std::cout << " Limiteur  " << valeur << " ( " << l->projectionLimiterId
                  << " ) " << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "PENTE_BORNE_PROJECTION")) {
        mesdonnees >> valeur;
        l->projectionAvecPlateauPente = ouiOUnon[valeur];
        std::cout << " Projection Avec Plateau Pente " << valeur << " ( "
                  << l->projectionAvecPlateauPente << " ) " << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "PENTE_BORNE_MASSE_MAILLE_PURE")) {
        getline(mesdonnees, ligne);  // Projection Avec Plateau Pente Mixte
        mesdonnees >> valeur;
        l->projectionLimiteurMixte = ouiOUnon[valeur];
        std::cout << " Projection Avec Plateau Pente Mixte " << valeur << " ( "
                  << l->projectionLimiteurMixte << " ) " << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "LIMITEURS_PROJECTION_MAILLE_PURE")) {
        mesdonnees >> valeur;
        l->projectionLimiterIdPure = limiteur[valeur];
        std::cout << " Limiteur pour Mailles Pures " << valeur << " ( "
                  << l->projectionLimiterIdPure << " ) " << std::endl;
        if ((l->projectionLimiterId == -1) &&
            (l->projectionLimiteurMixte == 1)) {
          cout << "ERREUR: Limiteur pour mailles pures non défini "
               << " alors que l option projectionLimiteurMixte est demandée"
               << endl;
          exit(1);
        }
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "PENTE_BORNE_COMPLET")) {
        mesdonnees >> valeur;
        l->projectionPlateauPenteComplet = ouiOUnon[valeur];
        std::cout << " Projection Avec Plateau Pente Complet" << valeur << " ( "
                  << l->projectionPlateauPenteComplet << " ) " << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "PSEUDO_CENTREE")) {
        mesdonnees >> valeur;
        o->pseudo_centree = ouiOUnon[valeur];
        std::cout << " Pseudo_centree : " << valeur << " ( "
                  << o->pseudo_centree << " ) " << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "CALCUL_FLUX_MASSE_DUAL")) {
        mesdonnees >> valeur;
        o->methode_flux_masse = A1OUA2OUPB[valeur];
        std::cout << " Methode Projection Dual : " << valeur << " ( "
                  << o->methode_flux_masse << " ) " << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "SCHEMA_PARTICULE")) {
        mesdonnees >> valeur;
        o->AvecParticules = ouiOUnon[valeur];
        std::cout << " Presence de Particules " << valeur << " ( "
                  << o->AvecParticules << " ) " << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "NOMBRE_PARTICULES")) {
        mesdonnees >> cstmesh->Nombre_Particules;
        std::cout << " Nombre de Particules " << cstmesh->Nombre_Particules
                  << std::endl;
        mesdonnees.ignore();
      }

      else if (!strcmp(motcle, "SORTIES")) {
        mesdonnees >> variables;
        while (strcmp(variables, "fin_liste")) {
          if (!strcmp(variables, "pression")) so->pression = true;
          if (!strcmp(variables, "densite")) so->densite = true;
          if (!strcmp(variables, "energie_interne")) so->energie_interne = true;
          if (!strcmp(variables, "fraction_volumique"))
            so->fraction_volumique = true;
          if (!strcmp(variables, "interface")) so->interface = true;
          if (!strcmp(variables, "vitesse")) so->vitesse = true;
          mesdonnees >> variables;
        }
        mesdonnees.ignore();
      } else {
        cout << "ERREUR: Mot-cle " << motcle << " inconnu" << endl;
        exit(1);
      }
    }
  } else {
    cout << "ERREUR: Impossible d'ouvrir le fichier en lecture." << endl;
    exit(1);
  }
}
