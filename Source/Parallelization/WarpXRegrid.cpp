/* Copyright 2019 Andrew Myers, Ann Almgren, Axel Huebl
 * David Grote, Maxence Thevenet, Michael Rowan
 * Remi Lehe, Weiqun Zhang, levinem
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_BLProfiler.H>

using namespace amrex;

void
WarpX::LoadBalance ()
{
    WARPX_PROFILE_REGION("LoadBalance");
    WARPX_PROFILE("WarpX::LoadBalance()");

    AMREX_ALWAYS_ASSERT(costs[0] != nullptr);
    
#ifdef AMREX_USE_MPI

    if (WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Heuristic)
    {
        WarpX::ComputeCostsHeuristic(costs);
    }

    // By default, do not do a redistribute; this toggles to true if RemakeLevel
    // is called for any level
    bool doLoadBalance = false;

    const int nLevels = finestLevel();
    for (int lev = 0; lev <= nLevels; ++lev)
    {
        // Gather prelims:
        // gather(n_elems_to_send, &recvcount[0], root) // recvcount
        // compute displ from recvcount
        
        // Gather quantities to root:
        // These all have the pattern
        //     gatherv(&my_quantity[0], my_quantity.size(),
        //             &root_recvbuf[0], root_recvcount,
        //             rootproc)
        //     new_index_to_old_index: [i,j,k, p,q, r,s,t,v, ..., a,b,c]    gatherv mfi.index()
        //     new_index_to_rank:      [1,1,1, 2,2, 3,3,3,3, ..., n,n,n]    gatherv MyProc()
        //     new_index_to_costs:     [0,5,3, 2,6, 2,3,4,3, ..., x,y,z]    gatherv costs[lev]()
        //     old_index_to_rank(dm):  pre-gather index --> rank

        amrex::Vector<Real> cost_to_send;
        //amrex::Vector<int> rank_to_send;
        amrex::Vector<int> index_to_send;

        MultiFab* Ex = Efield_fp[lev][0].get();
        for (MFIter mfi(*Ex, false); mfi.isValid(); ++mfi)
        {            
            cost_to_send.push_back((*costs[lev])[mfi.index()]);
            index_to_send.push_back(mfi.index());
        }
                
        amrex::Vector<int> recvcount, disp;
        recvcount.resize(ParallelDescriptor::NProcs(), 0);
        disp.resize(ParallelDescriptor::NProcs(), 0);

        Real n_cost = cost_to_send.size();
        ParallelDescriptor::Gather(&n_cost, 1,
                                   &recvcount[0], 1,
                                   ParallelDescriptor::IOProcessorNumber());
        
        for (int i=1; i<disp.size(); i++)
        {
            disp[i] = disp[i-1] + recvcount[i-1] + 1;
        }
        
        amrex::Vector<Real> new_index_to_cost;
        amrex::Vector<int> new_index_to_rank, new_index_to_old_index;
        new_index_to_cost.resize(costs[lev]->size(), 0.0);
        new_index_to_rank.resize(costs[lev]->size(), 0);
        new_index_to_old_index.resize(costs[lev]->size(), 0);
        
        
        ParallelDescriptor::Gatherv(&cost_to_send[0],
                                    cost_to_send.size(),
                                    &new_index_to_cost[0],
                                    recvcount,
                                    disp,
                                    ParallelDescriptor::IOProcessorNumber());
        
        // ParallelDescriptor::Gatherv(&rank_to_send[0],
        //                             rank_to_send.size(),
        //                             &new_index_to_rank[0],
        //                             recvcount,
        //                             disp,
        //                             ParallelDescriptor::IOProcessorNumber());
        
        ParallelDescriptor::Gatherv(&index_to_send[0],
                                    index_to_send.size(),
                                    &new_index_to_old_index[0],
                                    recvcount,
                                    disp,
                                    ParallelDescriptor::IOProcessorNumber());
        
        // Now work just on the root
        // ^With these, compute the cost in pre-gather index space
        if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber())
        {
            // Invert the new_ind-->old_ind
            Vector<int> old_index_to_new_index;
            old_index_to_new_index.resize(costs[lev]->size(), 0);
            for (int i=0; i<old_index_to_new_index.size(); i++)
            {
                new_index_to_old_index[old_index_to_new_index[i]] = i;
            }
        
            //amrex::Vector<Real> old_index_to_cost;
            //old_index_to_cost.resize(*costs[lev]->size(), 0.0);
            for (int i=0; i<costs[lev]->size(); i++)
            {
                (*costs[lev])[i] = new_index_to_cost[old_index_to_new_index[i]];
            }
        }

        // Decide whether to load balance based on cost
        // To store efficiency (meaning, the  average 'cost' over all ranks, normalized to the
        // max cost) for current distribution mapping
        amrex::Real currentEfficiency = 0.0;
        // To store efficiency for proposed distribution mapping
        amrex::Real proposedEfficiency = 0.0;
        const DistributionMapping newdm;
        
        if (load_balance_efficiency_ratio_threshold > 0.0
            & ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber())
        {
            // Compute efficiency for the current distribution mapping
            const DistributionMapping& currentdm = DistributionMap(lev);
            ComputeDistributionMappingEfficiency(currentdm, *costs[lev],
                                                 currentEfficiency);

            // Arguments for the load balancing
            const amrex::Real nboxes = costs[lev]->size();
            const amrex::Real nprocs = ParallelContext::NProcsSub();
            const int nmax = static_cast<int>(std::ceil(nboxes/nprocs*load_balance_knapsack_factor));

            newdm = (load_balance_with_sfc)
                ? DistributionMapping::makeSFC(*costs[lev], boxArray(lev), proposedEfficiency, false)
                : DistributionMapping::makeKnapSack(*costs[lev], proposedEfficiency, nmax);

            // Root has all the information to decide
            doLoadBalance = (proposedEfficiency > load_balance_efficiency_ratio_threshold*currentEfficiency);
        }

        // Broadcast the cost to all proc
        ParallelDescriptor::Bcast(&doLoadBalance, 1,
                                  ParallelDescriptor::IOProcessorNumber());
        
        if (doLoadBalance)
        {
            // Broadcastv newdm, or just the vector from which to construct
            //RemakeLevel(lev, t_new[lev], boxArray(lev), newdm);
        }
    }
    if (doLoadBalance)
    {
        //mypc->Redistribute();
    }
#endif
}


void
WarpX::RemakeLevel (int lev, Real /*time*/, const BoxArray& ba, const DistributionMapping& dm)
{
    if (ba == boxArray(lev))
    {
        if (ParallelDescriptor::NProcs() == 1) return;

        // Fine patch
        for (int idim=0; idim < 3; ++idim)
        {
            {
                const IntVect& ng = Bfield_fp[lev][idim]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Bfield_fp[lev][idim]->boxArray(),
                                                                  dm, Bfield_fp[lev][idim]->nComp(), ng));
                pmf->Redistribute(*Bfield_fp[lev][idim], 0, 0, Bfield_fp[lev][idim]->nComp(), ng);
                Bfield_fp[lev][idim] = std::move(pmf);
            }
            {
                const IntVect& ng = Efield_fp[lev][idim]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Efield_fp[lev][idim]->boxArray(),
                                                                  dm, Efield_fp[lev][idim]->nComp(), ng));
                pmf->Redistribute(*Efield_fp[lev][idim], 0, 0, Efield_fp[lev][idim]->nComp(), ng);
                Efield_fp[lev][idim] = std::move(pmf);
            }
            {
                const IntVect& ng = current_fp[lev][idim]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(current_fp[lev][idim]->boxArray(),
                                                                  dm, current_fp[lev][idim]->nComp(), ng));
                current_fp[lev][idim] = std::move(pmf);
            }
            if (current_store[lev][idim])
            {
                const IntVect& ng = current_store[lev][idim]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(current_store[lev][idim]->boxArray(),
                                                                  dm, current_store[lev][idim]->nComp(), ng));
                // no need to redistribute
                current_store[lev][idim] = std::move(pmf);
            }
        }

        if (F_fp[lev] != nullptr) {
            const IntVect& ng = F_fp[lev]->nGrowVect();
            auto pmf = std::unique_ptr<MultiFab>(new MultiFab(F_fp[lev]->boxArray(),
                                                              dm, F_fp[lev]->nComp(), ng));
            pmf->Redistribute(*F_fp[lev], 0, 0, F_fp[lev]->nComp(), ng);
            F_fp[lev] = std::move(pmf);
        }

        if (rho_fp[lev] != nullptr) {
            const int nc = rho_fp[lev]->nComp();
            const IntVect& ng = rho_fp[lev]->nGrowVect();
            auto pmf = std::unique_ptr<MultiFab>(new MultiFab(rho_fp[lev]->boxArray(),
                                                              dm, nc, ng));
            rho_fp[lev] = std::move(pmf);
        }

        // Aux patch
        if (lev == 0 && Bfield_aux[0][0]->ixType() == Bfield_fp[0][0]->ixType())
        {
            for (int idim = 0; idim < 3; ++idim) {
                Bfield_aux[lev][idim].reset(new MultiFab(*Bfield_fp[lev][idim], amrex::make_alias, 0, Bfield_aux[lev][idim]->nComp()));
                Efield_aux[lev][idim].reset(new MultiFab(*Efield_fp[lev][idim], amrex::make_alias, 0, Efield_aux[lev][idim]->nComp()));
            }
        } else {
            for (int idim=0; idim < 3; ++idim)
            {
                {
                    const IntVect& ng = Bfield_aux[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Bfield_aux[lev][idim]->boxArray(),
                                                                      dm, Bfield_aux[lev][idim]->nComp(), ng));
                    // pmf->Redistribute(*Bfield_aux[lev][idim], 0, 0, Bfield_aux[lev][idim]->nComp(), ng);
                    Bfield_aux[lev][idim] = std::move(pmf);
                }
                {
                    const IntVect& ng = Efield_aux[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Efield_aux[lev][idim]->boxArray(),
                                                                      dm, Efield_aux[lev][idim]->nComp(), ng));
                    // pmf->Redistribute(*Efield_aux[lev][idim], 0, 0, Efield_aux[lev][idim]->nComp(), ng);
                    Efield_aux[lev][idim] = std::move(pmf);
                }
            }
        }

        // Coarse patch
        if (lev > 0) {
            for (int idim=0; idim < 3; ++idim)
            {
                {
                    const IntVect& ng = Bfield_cp[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Bfield_cp[lev][idim]->boxArray(),
                                                                      dm, Bfield_cp[lev][idim]->nComp(), ng));
                    pmf->Redistribute(*Bfield_cp[lev][idim], 0, 0, Bfield_cp[lev][idim]->nComp(), ng);
                    Bfield_cp[lev][idim] = std::move(pmf);
                }
                {
                    const IntVect& ng = Efield_cp[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Efield_cp[lev][idim]->boxArray(),
                                                                      dm, Efield_cp[lev][idim]->nComp(), ng));
                    pmf->Redistribute(*Efield_cp[lev][idim], 0, 0, Efield_cp[lev][idim]->nComp(), ng);
                    Efield_cp[lev][idim] = std::move(pmf);
                }
                {
                    const IntVect& ng = current_cp[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>( new MultiFab(current_cp[lev][idim]->boxArray(),
                                                                       dm, current_cp[lev][idim]->nComp(), ng));
                    current_cp[lev][idim] = std::move(pmf);
                }
            }

            if (F_cp[lev] != nullptr) {
                const IntVect& ng = F_cp[lev]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(F_cp[lev]->boxArray(),
                                                                  dm, F_cp[lev]->nComp(), ng));
                pmf->Redistribute(*F_cp[lev], 0, 0, F_cp[lev]->nComp(), ng);
                F_cp[lev] = std::move(pmf);
            }

            if (rho_cp[lev] != nullptr) {
                const int nc = rho_cp[lev]->nComp();
                const IntVect& ng = rho_cp[lev]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(rho_cp[lev]->boxArray(),
                                                                  dm, nc, ng));
                rho_cp[lev] = std::move(pmf);
            }
        }

        if (lev > 0 && (n_field_gather_buffer > 0 || n_current_deposition_buffer > 0)) {
            for (int idim=0; idim < 3; ++idim)
            {
                if (Bfield_cax[lev][idim])
                {
                    const IntVect& ng = Bfield_cax[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Bfield_cax[lev][idim]->boxArray(),
                                                                      dm, Bfield_cax[lev][idim]->nComp(), ng));
                    // pmf->ParallelCopy(*Bfield_cax[lev][idim], 0, 0, Bfield_cax[lev][idim]->nComp(), ng, ng);
                    Bfield_cax[lev][idim] = std::move(pmf);
                }
                if (Efield_cax[lev][idim])
                {
                    const IntVect& ng = Efield_cax[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Efield_cax[lev][idim]->boxArray(),
                                                                      dm, Efield_cax[lev][idim]->nComp(), ng));
                    // pmf->ParallelCopy(*Efield_cax[lev][idim], 0, 0, Efield_cax[lev][idim]->nComp(), ng, ng);
                    Efield_cax[lev][idim] = std::move(pmf);
                }
                if (current_buf[lev][idim])
                {
                    const IntVect& ng = current_buf[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(current_buf[lev][idim]->boxArray(),
                                                                      dm, current_buf[lev][idim]->nComp(), ng));
                    // pmf->ParallelCopy(*current_buf[lev][idim], 0, 0, current_buf[lev][idim]->nComp(), ng, ng);
                    current_buf[lev][idim] = std::move(pmf);
                }
            }
            if (charge_buf[lev])
            {
                const IntVect& ng = charge_buf[lev]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(charge_buf[lev]->boxArray(),
                                                                  dm, charge_buf[lev]->nComp(), ng));
                // pmf->ParallelCopy(*charge_buf[lev][idim], 0, 0, charge_buf[lev]->nComp(), ng, ng);
                charge_buf[lev] = std::move(pmf);
            }
            if (current_buffer_masks[lev])
            {
                const IntVect& ng = current_buffer_masks[lev]->nGrowVect();
                auto pmf = std::unique_ptr<iMultiFab>(new iMultiFab(current_buffer_masks[lev]->boxArray(),
                                                                    dm, current_buffer_masks[lev]->nComp(), ng));
                // pmf->ParallelCopy(*current_buffer_masks[lev], 0, 0, current_buffer_masks[lev]->nComp(), ng, ng);
                current_buffer_masks[lev] = std::move(pmf);
            }
            if (gather_buffer_masks[lev])
            {
                const IntVect& ng = gather_buffer_masks[lev]->nGrowVect();
                auto pmf = std::unique_ptr<iMultiFab>(new iMultiFab(gather_buffer_masks[lev]->boxArray(),
                                                                    dm, gather_buffer_masks[lev]->nComp(), ng));
                // pmf->ParallelCopy(*gather_buffer_masks[lev], 0, 0, gather_buffer_masks[lev]->nComp(), ng, ng);
                gather_buffer_masks[lev] = std::move(pmf);
            }
        }

        if (costs[lev] != nullptr)
        {
            costs[lev].reset(new amrex::Vector<Real>);
            const int nboxes = Efield_fp[lev][0].get()->size();
            costs[lev]->resize(nboxes, 0.0);
        }

        SetDistributionMap(lev, dm);

    } else
    {
        amrex::Abort("RemakeLevel: to be implemented");
    }
}

void
WarpX::ComputeCostsHeuristic (amrex::Vector<std::unique_ptr<amrex::Vector<amrex::Real> > >& a_costs)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto & mypc_ref = WarpX::GetInstance().GetPartContainer();
        auto nSpecies = mypc_ref.nSpecies();

        // Species loop
        for (int i_s = 0; i_s < nSpecies; ++i_s)
        {
            auto & myspc = mypc_ref.GetParticleContainer(i_s);

            // Particle loop
            for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti)
            {
                (*a_costs[lev])[pti.index()] += costs_heuristic_particles_wt*pti.numParticles();
            }
        }

        //Cell loop
        MultiFab* Ex = Efield_fp[lev][0].get();
        for (MFIter mfi(*Ex, false); mfi.isValid(); ++mfi)
        {
            const Box& gbx = mfi.growntilebox();
            (*a_costs[lev])[mfi.index()] += costs_heuristic_cells_wt*gbx.numPts();
        }
    }
}

void
WarpX::ResetCosts ()
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        costs[lev]->assign((*costs[lev]).size(), 0.0);
    }
}

void
WarpX::ComputeDistributionMappingEfficiency (const DistributionMapping& dm,
                                             const Vector<Real>& cost,
                                             Real& efficiency)
{
    const Real nprocs = ParallelDescriptor::NProcs();

    // Collect costs per fab corresponding to each rank, then collapse into vector
    // of total cost per proc

    // This will store mapping from (proc) --> ([cost_FAB_1, cost_FAB_2, ... ])
    // for each proc
    std::map<int, Vector<Real>> rankToCosts;

    for (int i=0; i<cost.size(); ++i)
    {
        rankToCosts[dm[i]].push_back(cost[i]);
    }

    Real maxCost = -1.0;

    // This will store mapping from (proc) --> (sum of cost) for each proc
    Vector<Real> rankToCost = {0.0};
    rankToCost.resize(nprocs);
    for (int i=0; i<nprocs; ++i) {
        const Real rwSum = std::accumulate(rankToCosts[i].begin(),
                                           rankToCosts[i].end(), 0.0);
        rankToCost[i] = rwSum;
        maxCost = std::max(maxCost, rwSum);
    }

    // `efficiency` is mean cost per proc
    efficiency = (std::accumulate(rankToCost.begin(),
                  rankToCost.end(), 0.0) / (nprocs*maxCost));
}
