
#include <cmath>
#include <limits>

#include <WarpX.H>
#include <WarpXConst.H>
#include <WarpX_f.H>
#ifdef WARPX_USE_PY
#include <WarpX_py.H>
#endif

using namespace amrex;

void
WarpX::Evolve (int numsteps) {
    BL_PROFILE_REGION("WarpX::Evolve()");

#ifdef WARPX_DO_ELECTROSTATIC
    if (do_electrostatic) {
        EvolveES(numsteps);
    } else {
      EvolveEM(numsteps);
    }
#else
    EvolveEM(numsteps);
#endif // WARPX_DO_ELECTROSTATIC
    
}

void
WarpX::EvolveEM (int numsteps)
{
    BL_PROFILE("WarpX::EvolveEM()");

    Real cur_time = t_new[0];
    static int last_plot_file_step = 0;
    static int last_check_file_step = 0;

    int numsteps_max;
    if (numsteps < 0) {  // Note that the default argument is numsteps = -1
        numsteps_max = max_step;
    } else {
        numsteps_max = std::min(istep[0]+numsteps, max_step);
    }

    bool max_time_reached = false;
    Real walltime, walltime_start = amrex::second();
    for (int step = istep[0]; step < numsteps_max && cur_time < stop_time; ++step)
    {

	// Start loop on time steps
        amrex::Print() << "\nSTEP " << step+1 << " starts ...\n";
#ifdef WARPX_USE_PY
        if (warpx_py_beforestep) warpx_py_beforestep();
#endif

        if (costs[0] != nullptr)
        {
#ifdef WARPX_USE_PSATD
            amrex::Abort("LoadBalance for PSATD: TODO");
#endif
            if (step > 0 && (step+1) % load_balance_int == 0)
            {
                LoadBalance();
		// Reset the costs to 0
		for (int lev = 0; lev <= finest_level; ++lev) {
		  costs[lev]->setVal(0.0);
		}
            }

            for (int lev = 0; lev <= finest_level; ++lev) {
	      // Perform running average of the costs
	      // (Giving more importance to most recent costs)
	      (*costs[lev].get()).mult( (1. - 2./load_balance_int) );
            }
        }

        // At the beginning, we have B^{n} and E^{n}.
        // Particles have p^{n} and x^{n}.
        // is_synchronized is true.
        if (is_synchronized) {
            FillBoundaryE();
            FillBoundaryB();
            UpdateAuxilaryData();
            // on first step, push p by -0.5*dt
            for (int lev = 0; lev <= finest_level; ++lev) {
                mypc->PushP(lev, -0.5*dt[0],
                            *Efield_aux[lev][0],*Efield_aux[lev][1],*Efield_aux[lev][2],
                            *Bfield_aux[lev][0],*Bfield_aux[lev][1],*Bfield_aux[lev][2]);
            }
           is_synchronized = false;
        } else {
           // Beyond one step, we have E^{n} and B^{n}.
           // Particles have p^{n-1/2} and x^{n}.
           UpdateAuxilaryData();
       }

        // Push particle from x^{n} to x^{n+1}
        //               from p^{n-1/2} to p^{n+1/2}
        // Deposit current j^{n+1/2}
        // Deposit charge density rho^{n}
#ifdef WARPX_USE_PY
        if (warpx_py_particleinjection) warpx_py_particleinjection();
        if (warpx_py_particlescraper) warpx_py_particlescraper();
        if (warpx_py_beforedeposition) warpx_py_beforedeposition();
#endif
        PushParticlesandDepose(cur_time);
#ifdef WARPX_USE_PY
        if (warpx_py_afterdeposition) warpx_py_afterdeposition();
#endif

        SyncCurrent();

        SyncRho(rho_fp, rho_cp);

        // Push E and B from {n} to {n+1}
        // (And update guard cells immediately afterwards)
#ifdef WARPX_USE_PSATD
        PushPSATD(dt[0]);
        FillBoundaryE();
        FillBoundaryB();
#else
        EvolveF(0.5*dt[0], DtType::FirstHalf);
        EvolveB(0.5*dt[0]); // We now have B^{n+1/2}
        FillBoundaryB();
        EvolveE(dt[0]); // We now have E^{n+1}
        FillBoundaryE();
        EvolveF(0.5*dt[0], DtType::SecondHalf);
        EvolveB(0.5*dt[0]); // We now have B^{n+1}
        if (do_pml) {
            DampPML();
            FillBoundaryE();
        }
        FillBoundaryB();
#endif

#ifdef WARPX_USE_PY
        if (warpx_py_beforeEsolve) warpx_py_beforeEsolve();
#endif
        if (cur_time + dt[0] >= stop_time - 1.e-3*dt[0] || step == numsteps_max-1) {
            // At the end of last step, push p by 0.5*dt to synchronize
            UpdateAuxilaryData();
            for (int lev = 0; lev <= finest_level; ++lev) {
                mypc->PushP(lev, 0.5*dt[0],
                    *Efield_aux[lev][0],*Efield_aux[lev][1],*Efield_aux[lev][2],
                    *Bfield_aux[lev][0],*Bfield_aux[lev][1],*Bfield_aux[lev][2]);
                }
            is_synchronized = true;
        }
#ifdef WARPX_USE_PY
        if (warpx_py_afterEsolve) warpx_py_afterEsolve();
#endif

        for (int lev = 0; lev <= max_level; ++lev) {
            ++istep[lev];
        }

	cur_time += dt[0];

        bool to_make_plot = (plot_int > 0) && ((step+1) % plot_int == 0);

        bool move_j = is_synchronized || to_make_plot;
        // If is_synchronized we need to shift j too so that next step we can evolve E by dt/2.
        // We might need to move j because we are going to make a plotfile.
	MoveWindow(move_j);

        if (max_level == 0) {
            mypc->RedistributeLocal();
        }
        else {
            mypc->Redistribute();
        }

        amrex::Print()<< "STEP " << step+1 << " ends." << " TIME = " << cur_time
                      << " DT = " << dt[0] << "\n";
        walltime = amrex::second() - walltime_start;
        amrex::Print()<< "Walltime = " << walltime
             << " s; Avg. per step = " << walltime/(step+1) << " s\n";

	// sync up time
	for (int i = 0; i <= max_level; ++i) {
	    t_new[i] = cur_time;
	}

        if (do_boosted_frame_diagnostic) {
            std::unique_ptr<MultiFab> cell_centered_data = nullptr;
            if (WarpX::do_boosted_frame_fields) {
                cell_centered_data = GetCellCenteredData();
            }
            myBFD->writeLabFrameData(cell_centered_data.get(), *mypc, geom[0], cur_time, dt[0]);
        }

	if (to_make_plot)
        {
            FillBoundaryE();
            FillBoundaryB();
            UpdateAuxilaryData();

            for (int lev = 0; lev <= finest_level; ++lev) {
                mypc->FieldGather(lev,
                                  *Efield_aux[lev][0],*Efield_aux[lev][1],*Efield_aux[lev][2],
                                  *Bfield_aux[lev][0],*Bfield_aux[lev][1],*Bfield_aux[lev][2]);
            }

	    last_plot_file_step = step+1;
	    WritePlotFile();
	}

	if (check_int > 0 && (step+1) % check_int == 0) {
	    last_check_file_step = step+1;
	    WriteCheckPointFile();
	}

	if (cur_time >= stop_time - 1.e-3*dt[0]) {
	    max_time_reached = true;
	    break;
	}

#ifdef WARPX_USE_PY
        if (warpx_py_afterstep) warpx_py_afterstep();
#endif
	// End loop on time steps
    }

    if (plot_int > 0 && istep[0] > last_plot_file_step && (max_time_reached || istep[0] >= max_step))
    {
        FillBoundaryE();
        FillBoundaryB();
        UpdateAuxilaryData();
        
        for (int lev = 0; lev <= finest_level; ++lev) {
            mypc->FieldGather(lev,
                              *Efield_aux[lev][0],*Efield_aux[lev][1],*Efield_aux[lev][2],
                              *Bfield_aux[lev][0],*Bfield_aux[lev][1],*Bfield_aux[lev][2]);
        }

	WritePlotFile();
    }

    if (check_int > 0 && istep[0] > last_check_file_step && (max_time_reached || istep[0] >= max_step)) {
	WriteCheckPointFile();
    }

    if (do_boosted_frame_diagnostic) {
        myBFD->Flush(geom[0]);
    }
}

void
WarpX::EvolveB (Real dt)
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        EvolveB(lev, dt);
    }
}

void
WarpX::EvolveB (int lev, Real dt)
{
    BL_PROFILE("WarpX::EvolveB()");

    // Parameters of the solver: order and mesh spacing
    const int norder = 2;

    int npatches = (lev == 0) ? 1 : 2;

    for (int ipatch = 0; ipatch < npatches; ++ipatch)
    {
        int patch_level = (ipatch == 0) ? lev : lev-1;
        const std::array<Real,3>& dx = WarpX::CellSize(patch_level);
        const std::array<Real,3> dtsdx {dt/dx[0], dt/dx[1], dt/dx[2]};

        MultiFab *Ex, *Ey, *Ez, *Bx, *By, *Bz;
        if (ipatch == 0)
        {
            Ex = Efield_fp[lev][0].get();
            Ey = Efield_fp[lev][1].get();
            Ez = Efield_fp[lev][2].get();
            Bx = Bfield_fp[lev][0].get();
            By = Bfield_fp[lev][1].get();
            Bz = Bfield_fp[lev][2].get();
        }
        else
        {
            Ex = Efield_cp[lev][0].get();
            Ey = Efield_cp[lev][1].get();
            Ez = Efield_cp[lev][2].get();
            Bx = Bfield_cp[lev][0].get();
            By = Bfield_cp[lev][1].get();
            Bz = Bfield_cp[lev][2].get();
        }

        MultiFab* cost = costs[lev].get();
        const IntVect& rr = (lev < finestLevel()) ? refRatio(lev) : IntVect::TheUnitVector();

        // Loop through the grids, and over the tiles within each grid
#ifdef _OPENMP
#pragma omp parallel
#endif
        for ( MFIter mfi(*Bx,true); mfi.isValid(); ++mfi )
        {
            Real wt = amrex::second();

            const Box& tbx  = mfi.tilebox(Bx_nodal_flag);
            const Box& tby  = mfi.tilebox(By_nodal_flag);
            const Box& tbz  = mfi.tilebox(Bz_nodal_flag);

            // Call picsar routine for each tile
            warpx_push_bvec(
                tbx.loVect(), tbx.hiVect(),
                tby.loVect(), tby.hiVect(),
                tbz.loVect(), tbz.hiVect(),
                BL_TO_FORTRAN_3D((*Ex)[mfi]),
                BL_TO_FORTRAN_3D((*Ey)[mfi]),
                BL_TO_FORTRAN_3D((*Ez)[mfi]),
                BL_TO_FORTRAN_3D((*Bx)[mfi]),
                BL_TO_FORTRAN_3D((*By)[mfi]),
                BL_TO_FORTRAN_3D((*Bz)[mfi]),
	        &dtsdx[0], &dtsdx[1], &dtsdx[2],
	        &WarpX::maxwell_fdtd_solver_id);

            if (cost) {
                Box cbx = mfi.tilebox(IntVect{AMREX_D_DECL(0,0,0)});
                if (ipatch == 1) cbx.refine(rr);
                wt = (amrex::second() - wt) / cbx.d_numPts();
                (*cost)[mfi].plus(wt, cbx);
            }
        }
    }

    if (do_pml && pml[lev]->ok())
    {
        for (int ipatch = 0; ipatch < npatches; ++ipatch)
        {
            const auto& pml_B = (ipatch==0) ? pml[lev]->GetB_fp() : pml[lev]->GetB_cp();
            const auto& pml_E = (ipatch==0) ? pml[lev]->GetE_fp() : pml[lev]->GetE_cp();
	    int patch_level = (ipatch == 0) ? lev : lev-1;
	    const std::array<Real,3>& dx = WarpX::CellSize(patch_level);
	    const std::array<Real,3> dtsdx {dt/dx[0], dt/dx[1], dt/dx[2]};
#ifdef _OPENMP
#pragma omp parallel
#endif
            for ( MFIter mfi(*pml_B[0],true); mfi.isValid(); ++mfi )
            {
                const Box& tbx  = mfi.tilebox(Bx_nodal_flag);
                const Box& tby  = mfi.tilebox(By_nodal_flag);
                const Box& tbz  = mfi.tilebox(Bz_nodal_flag);

                WRPX_PUSH_PML_BVEC(
                    tbx.loVect(), tbx.hiVect(),
                    tby.loVect(), tby.hiVect(),
                    tbz.loVect(), tbz.hiVect(),
                    BL_TO_FORTRAN_3D((*pml_E[0])[mfi]),
                    BL_TO_FORTRAN_3D((*pml_E[1])[mfi]),
                    BL_TO_FORTRAN_3D((*pml_E[2])[mfi]),
                    BL_TO_FORTRAN_3D((*pml_B[0])[mfi]),
                    BL_TO_FORTRAN_3D((*pml_B[1])[mfi]),
                    BL_TO_FORTRAN_3D((*pml_B[2])[mfi]),
		    &dtsdx[0], &dtsdx[1], &dtsdx[2],
		    &WarpX::maxwell_fdtd_solver_id);
            }
        }
    }
}

void
WarpX::EvolveE (Real dt)
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        EvolveE(lev, dt);
    }
}

void
WarpX::EvolveE (int lev, Real dt)
{
    BL_PROFILE("WarpX::EvolveE()");

    // Parameters of the solver: order and mesh spacing
    const int norder = 2;
    static constexpr Real c2 = PhysConst::c*PhysConst::c;
    const Real mu_c2_dt = (PhysConst::mu0*PhysConst::c*PhysConst::c) * dt;
    const Real c2dt = (PhysConst::c*PhysConst::c) * dt;

    int npatches = (lev == 0) ? 1 : 2;

    for (int ipatch = 0; ipatch < npatches; ++ipatch)
    {
        int patch_level = (ipatch == 0) ? lev : lev-1;
        const std::array<Real,3>& dx = WarpX::CellSize(patch_level);
        const std::array<Real,3> dtsdx_c2 {c2dt/dx[0], c2dt/dx[1], c2dt/dx[2]};

        MultiFab *Ex, *Ey, *Ez, *Bx, *By, *Bz, *jx, *jy, *jz, *F;
        if (ipatch == 0)
        {
            Ex = Efield_fp[lev][0].get();
            Ey = Efield_fp[lev][1].get();
            Ez = Efield_fp[lev][2].get();
            Bx = Bfield_fp[lev][0].get();
            By = Bfield_fp[lev][1].get();
            Bz = Bfield_fp[lev][2].get();
            jx = current_fp[lev][0].get();
            jy = current_fp[lev][1].get();
            jz = current_fp[lev][2].get();
            F  = F_fp[lev].get();
        }
        else
        {
            Ex = Efield_cp[lev][0].get();
            Ey = Efield_cp[lev][1].get();
            Ez = Efield_cp[lev][2].get();
            Bx = Bfield_cp[lev][0].get();
            By = Bfield_cp[lev][1].get();
            Bz = Bfield_cp[lev][2].get();
            jx = current_cp[lev][0].get();
            jy = current_cp[lev][1].get();
            jz = current_cp[lev][2].get();
            F  = F_cp[lev].get();
        }

        MultiFab* cost = costs[lev].get();
        const IntVect& rr = (lev < finestLevel()) ? refRatio(lev) : IntVect::TheUnitVector();

        // Loop through the grids, and over the tiles within each grid
#ifdef _OPENMP
#pragma omp parallel
#endif
        for ( MFIter mfi(*Ex,true); mfi.isValid(); ++mfi )
        {
            Real wt = amrex::second();

            const Box& tex  = mfi.tilebox(Ex_nodal_flag);
            const Box& tey  = mfi.tilebox(Ey_nodal_flag);
            const Box& tez  = mfi.tilebox(Ez_nodal_flag);

            // Call picsar routine for each tile
            warpx_push_evec(
                tex.loVect(), tex.hiVect(),
                tey.loVect(), tey.hiVect(),
                tez.loVect(), tez.hiVect(),
                BL_TO_FORTRAN_3D((*Ex)[mfi]),
                BL_TO_FORTRAN_3D((*Ey)[mfi]),
                BL_TO_FORTRAN_3D((*Ez)[mfi]),
                BL_TO_FORTRAN_3D((*Bx)[mfi]),
                BL_TO_FORTRAN_3D((*By)[mfi]),
                BL_TO_FORTRAN_3D((*Bz)[mfi]),
                BL_TO_FORTRAN_3D((*jx)[mfi]),
                BL_TO_FORTRAN_3D((*jy)[mfi]),
                BL_TO_FORTRAN_3D((*jz)[mfi]),
                &mu_c2_dt,
                &dtsdx_c2[0], &dtsdx_c2[1], &dtsdx_c2[2]);

            if (F) {
                WRPX_CLEAN_EVEC(tex.loVect(), tex.hiVect(),
                                tey.loVect(), tey.hiVect(),
                                tez.loVect(), tez.hiVect(),
                                BL_TO_FORTRAN_3D((*Ex)[mfi]),
                                BL_TO_FORTRAN_3D((*Ey)[mfi]),
                                BL_TO_FORTRAN_3D((*Ez)[mfi]),
                                BL_TO_FORTRAN_3D((*F)[mfi]),
                                &dtsdx_c2[0]);
            }

            if (cost) {
                Box cbx = mfi.tilebox(IntVect{AMREX_D_DECL(0,0,0)});
                if (ipatch == 1) cbx.refine(rr);
                wt = (amrex::second() - wt) / cbx.d_numPts();
                (*cost)[mfi].plus(wt, cbx);
            }
        }
    }

    if (do_pml && pml[lev]->ok())
    {
        pml[lev]->ExchangeF(F_fp[lev].get(), F_cp[lev].get());

        for (int ipatch = 0; ipatch < npatches; ++ipatch)
        {
            const auto& pml_B = (ipatch==0) ? pml[lev]->GetB_fp() : pml[lev]->GetB_cp();
            const auto& pml_E = (ipatch==0) ? pml[lev]->GetE_fp() : pml[lev]->GetE_cp();
            const auto& pml_F = (ipatch==0) ? pml[lev]->GetF_fp() : pml[lev]->GetF_cp();
            int patch_level = (ipatch == 0) ? lev : lev-1;
            const std::array<Real,3>& dx = WarpX::CellSize(patch_level);
            const std::array<Real,3> dtsdx_c2 {c2dt/dx[0], c2dt/dx[1], c2dt/dx[2]};
#ifdef _OPENMP
#pragma omp parallel
#endif
            for ( MFIter mfi(*pml_E[0],true); mfi.isValid(); ++mfi )
            {
                const Box& tex  = mfi.tilebox(Ex_nodal_flag);
                const Box& tey  = mfi.tilebox(Ey_nodal_flag);
                const Box& tez  = mfi.tilebox(Ez_nodal_flag);

                WRPX_PUSH_PML_EVEC(
                    tex.loVect(), tex.hiVect(),
                    tey.loVect(), tey.hiVect(),
                    tez.loVect(), tez.hiVect(),
                    BL_TO_FORTRAN_3D((*pml_E[0])[mfi]),
                    BL_TO_FORTRAN_3D((*pml_E[1])[mfi]),
                    BL_TO_FORTRAN_3D((*pml_E[2])[mfi]),
                    BL_TO_FORTRAN_3D((*pml_B[0])[mfi]),
                    BL_TO_FORTRAN_3D((*pml_B[1])[mfi]),
                    BL_TO_FORTRAN_3D((*pml_B[2])[mfi]),
                    &dtsdx_c2[0], &dtsdx_c2[1], &dtsdx_c2[2]);

                if (pml_F)
                {
                    WRPX_PUSH_PML_EVEC_F(
                        tex.loVect(), tex.hiVect(),
                        tey.loVect(), tey.hiVect(),
                        tez.loVect(), tez.hiVect(),
                        BL_TO_FORTRAN_3D((*pml_E[0])[mfi]),
                        BL_TO_FORTRAN_3D((*pml_E[1])[mfi]),
                        BL_TO_FORTRAN_3D((*pml_E[2])[mfi]),
                        BL_TO_FORTRAN_3D((*pml_F   )[mfi]),
                        &dtsdx_c2[0]);
                }
            }
        }
    }
}

void
WarpX::EvolveF (Real dt, DtType dt_type)
{
    if (!do_dive_cleaning) return;

    for (int lev = 0; lev <= finest_level; ++lev) {
        EvolveF(lev, dt, dt_type);
    }
}

void
WarpX::EvolveF (int lev, Real dt, DtType dt_type)
{
    if (!do_dive_cleaning) return;

    BL_PROFILE("WarpX::EvolveF()");

    static constexpr Real c2inv = 1.0/(PhysConst::c*PhysConst::c);
    static constexpr Real mu_c2 = PhysConst::mu0*PhysConst::c*PhysConst::c;

    int npatches = (lev == 0) ? 1 : 2;

    for (int ipatch = 0; ipatch < npatches; ++ipatch)
    {
        int patch_level = (ipatch == 0) ? lev : lev-1;
        const auto& dx = WarpX::CellSize(patch_level);
        const std::array<Real,3> dtsdx {dt/dx[0], dt/dx[1], dt/dx[2]};

        MultiFab *Ex, *Ey, *Ez, *rho, *F;
        if (ipatch == 0)
        {
            Ex = Efield_fp[lev][0].get();
            Ey = Efield_fp[lev][1].get();
            Ez = Efield_fp[lev][2].get();
            rho = rho_fp[lev].get();
            F = F_fp[lev].get();
        }
        else
        {
            Ex = Efield_cp[lev][0].get();
            Ey = Efield_cp[lev][1].get();
            Ez = Efield_cp[lev][2].get();
            rho = rho_cp[lev].get();
            F = F_cp[lev].get();
        }

        const int rhocomp = (dt_type == DtType::FirstHalf) ? 0 : 1;

        MultiFab src(rho->boxArray(), rho->DistributionMap(), 1, 0);
        ComputeDivE(src, 0, {Ex,Ey,Ez}, dx);
        MultiFab::Saxpy(src, -mu_c2, *rho, rhocomp, 0, 1, 0);
        MultiFab::Saxpy(*F, dt, src, 0, 0, 1, 0);

        if (do_pml && pml[lev]->ok())
        {
            const auto& pml_F = (ipatch==0) ? pml[lev]->GetF_fp() : pml[lev]->GetF_cp();
            const auto& pml_E = (ipatch==0) ? pml[lev]->GetE_fp() : pml[lev]->GetE_cp();

#ifdef _OPENMP
#pragma omp parallel
#endif
            for ( MFIter mfi(*pml_F,true); mfi.isValid(); ++mfi )
            {
                const Box& bx = mfi.tilebox();
                WRPX_PUSH_PML_F(bx.loVect(), bx.hiVect(),
                                BL_TO_FORTRAN_ANYD((*pml_F   )[mfi]),
                                BL_TO_FORTRAN_ANYD((*pml_E[0])[mfi]),
                                BL_TO_FORTRAN_ANYD((*pml_E[1])[mfi]),
                                BL_TO_FORTRAN_ANYD((*pml_E[2])[mfi]),
                                &dtsdx[0], &dtsdx[1], &dtsdx[2]);
            }
        }
    }
}

void
WarpX::DampPML ()
{
    if (!do_pml) return;

    for (int lev = 0; lev <= finest_level; ++lev) {
        DampPML(lev);
    }
}

void
WarpX::DampPML (int lev)
{
    if (!do_pml) return;

    BL_PROFILE("WarpX::DampPML()");

    int npatches = (lev == 0) ? 1 : 2;

    for (int ipatch = 0; ipatch < npatches; ++ipatch)
    {
        if (pml[lev]->ok())
        {
            const auto& pml_E = (ipatch==0) ? pml[lev]->GetE_fp() : pml[lev]->GetE_cp();
            const auto& pml_B = (ipatch==0) ? pml[lev]->GetB_fp() : pml[lev]->GetB_cp();
            const auto& pml_F = (ipatch==0) ? pml[lev]->GetF_fp() : pml[lev]->GetF_cp();
            const auto& sigba = (ipatch==0) ? pml[lev]->GetMultiSigmaBox_fp()
                                            : pml[lev]->GetMultiSigmaBox_cp();

#ifdef _OPENMP
#pragma omp parallel
#endif
            for ( MFIter mfi(*pml_E[0],true); mfi.isValid(); ++mfi )
            {
                const Box& tex  = mfi.tilebox(Ex_nodal_flag);
                const Box& tey  = mfi.tilebox(Ey_nodal_flag);
                const Box& tez  = mfi.tilebox(Ez_nodal_flag);
                const Box& tbx  = mfi.tilebox(Bx_nodal_flag);
                const Box& tby  = mfi.tilebox(By_nodal_flag);
                const Box& tbz  = mfi.tilebox(Bz_nodal_flag);

                WRPX_DAMP_PML(tex.loVect(), tex.hiVect(),
                              tey.loVect(), tey.hiVect(),
                              tez.loVect(), tez.hiVect(),
                              tbx.loVect(), tbx.hiVect(),
                              tby.loVect(), tby.hiVect(),
                              tbz.loVect(), tbz.hiVect(),
                              BL_TO_FORTRAN_3D((*pml_E[0])[mfi]),
                              BL_TO_FORTRAN_3D((*pml_E[1])[mfi]),
                              BL_TO_FORTRAN_3D((*pml_E[2])[mfi]),
                              BL_TO_FORTRAN_3D((*pml_B[0])[mfi]),
                              BL_TO_FORTRAN_3D((*pml_B[1])[mfi]),
                              BL_TO_FORTRAN_3D((*pml_B[2])[mfi]),
                              WRPX_PML_TO_FORTRAN(sigba[mfi]));

                if (pml_F) {
                    const Box& tnd  = mfi.nodaltilebox();
                    WRPX_DAMP_PML_F(tnd.loVect(), tnd.hiVect(),
                                    BL_TO_FORTRAN_3D((*pml_F)[mfi]),
                                    WRPX_PML_TO_FORTRAN(sigba[mfi]));
                }
            }
        }
    }
}

void
WarpX::PushParticlesandDepose (Real cur_time)
{
    // Evolve particles to p^{n+1/2} and x^{n+1}
    // Depose current, j^{n+1/2}
    for (int lev = 0; lev <= finest_level; ++lev) {
        PushParticlesandDepose(lev, cur_time);
    }
}

void
WarpX::PushParticlesandDepose (int lev, Real cur_time)
{
    mypc->Evolve(lev,
                 *Efield_aux[lev][0],*Efield_aux[lev][1],*Efield_aux[lev][2],
                 *Bfield_aux[lev][0],*Bfield_aux[lev][1],*Bfield_aux[lev][2],
                 *current_fp[lev][0],*current_fp[lev][1],*current_fp[lev][2],
                 current_buf[lev][0].get(), current_buf[lev][1].get(), current_buf[lev][2].get(),
                 rho_fp[lev].get(), 
                 Efield_cax[lev][0].get(), Efield_cax[lev][1].get(), Efield_cax[lev][2].get(),
                 Bfield_cax[lev][0].get(), Bfield_cax[lev][1].get(), Bfield_cax[lev][2].get(),
                 cur_time, dt[lev]);
}

void
WarpX::ComputeDt ()
{
    const Real* dx = geom[max_level].CellSize();
    Real deltat = 0.;
    
    if (maxwell_fdtd_solver_id == 0) {
      // CFL time step Yee solver
      deltat  = cfl * 1./( std::sqrt(AMREX_D_TERM(  1./(dx[0]*dx[0]),
                                                    + 1./(dx[1]*dx[1]),
                                                    + 1./(dx[2]*dx[2]))) * PhysConst::c );
    } else {
      // CFL time step CKC solver
#if (BL_SPACEDIM == 3)
      const Real delta = std::min(dx[0],std::min(dx[1],dx[2]));
#elif (BL_SPACEDIM == 2)
      const Real delta = std::min(dx[0],dx[1]);
#endif
      deltat = cfl*delta/PhysConst::c;
    }
    dt.resize(0);
    dt.resize(max_level+1,deltat);

    if (do_electrostatic) {
        dt[0] = const_dt;
    }
}