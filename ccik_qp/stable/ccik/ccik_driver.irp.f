subroutine ccik_banner
  implicit none
  print *, '============================================'
  print *, ' Quantum Package plugin: baseline ccik'
  print *, ' Determinant-driven compressed Krylov solver'
  print *, '============================================'
end


subroutine ccik_print_parameters
  implicit none
  print *, 'ccik_generator_keep  = ', ccik_generator_keep
  print *, 'ccik_generator_fraction = ', ccik_generator_fraction
  print *, 'ccik_generator_abs_coeff_cutoff = ', ccik_generator_abs_coeff_cutoff
  print *, 'ccik_max_parents_for_expansion = ', ccik_max_parents_for_expansion
  print *, 'ccik_include_current_support = ', ccik_include_current_support
  print *, 'ccik_selector_keep   = ', ccik_selector_keep
  print *, 'ccik_external_keep   = ', ccik_external_keep
  print *, 'ccik_external_candidate_factor = ', ccik_external_candidate_factor
  print *, 'ccik_pt2_rank_weight = ', ccik_pt2_rank_weight
  print *, 'ccik_selector_mode   = ', trim(adjustl(ccik_selector_mode))
  print *, 'ccik_matvec_backend = ', trim(adjustl(ccik_matvec_backend))
  print *, 'ccik_compression_policy = ', trim(adjustl(ccik_compression_policy))
  print *, 'ccik_fixed_store_keep = ', ccik_fixed_store_keep
  print *, 'ccik_fixed_propagate_keep = ', ccik_fixed_propagate_keep
  print *, 'ccik_secondortho    = ', ccik_secondortho
  print *, 'ccik_store_abs_tol  = ', ccik_store_abs_tol
  print *, 'ccik_propagate_abs_tol = ', ccik_propagate_abs_tol
  print *, 'ccik_nkeep           = ', ccik_nkeep
  print *, 'ccik_tmp_keep_factor = ', ccik_tmp_keep_factor
  print *, 'ccik_preserve_prev_support = ', ccik_preserve_prev_support
  print *, 'ccik_preserve_prev_support_cap = ', ccik_preserve_prev_support_cap
  print *, 'ccik_total_det_budget= ', ccik_total_det_budget
  print *, 'ccik_basis_size      = ', ccik_basis_size
  print *, 'ccik_conv_tol        = ', ccik_conv_tol
  print *, 'ccik_ritz_overlap_tol= ', ccik_ritz_overlap_tol
  print *, 'ccik_ritz_max_drop   = ', ccik_ritz_max_drop
  print *, 'ccik_save_final_wf   = ', ccik_save_final_wf
  print *, 'ccik_do_post_pt2     = ', ccik_do_post_pt2
  print *, 'ccik_max_cycle       = ', ccik_max_cycle
  print *, 'ccik_restart_m_cycle = ', ccik_restart_m_cycle
  print *, 'ccik_restart_nroot   = ', ccik_restart_nroot
  print *, 'ccik_exactized_checkpoint_every = ', ccik_exactized_checkpoint_every
  print *, 'ccik_exactized_resid_tol = ', ccik_exactized_resid_tol
  print *, 'ccik_exactized_max_iter = ', ccik_exactized_max_iter
  print *, 'ccik_exactized_max_subspace = ', ccik_exactized_max_subspace
  print *, 'ccik_exactized_restart_keep = ', ccik_exactized_restart_keep
  print *, 'ccik_verbose         = ', ccik_verbose
  print *, 'ccik_debug_trace     = ', ccik_debug_trace
  print *, 'ccik_debug_trace_topn= ', ccik_debug_trace_topn
  print *, 'ccik_debug_allow_legacy_fixed_budget = ', ccik_debug_allow_legacy_fixed_budget
end


subroutine ccik_debug_trace_reset(label)
  implicit none
  character(len=*), intent(in) :: label
  integer, parameter :: trace_unit = 916

  if (.not. ccik_debug_trace) return
  open(unit=trace_unit, file='ccik_debug_trace.log', status='replace', action='write')
  write(trace_unit,'(A)') '# ccik debug trace'
  write(trace_unit,'(A,1X,A)') '# label', trim(label)
  close(trace_unit)
end


subroutine ccik_debug_trace_line(line)
  implicit none
  character(len=*), intent(in) :: line
  integer, parameter :: trace_unit = 916

  if (.not. ccik_debug_trace) return
  open(unit=trace_unit, file='ccik_debug_trace.log', status='unknown', position='append', action='write')
  write(trace_unit,'(A)') trim(line)
  close(trace_unit)
end


subroutine ccik_debug_trace_vector(tag, vec)
  use ccik_sparse
  implicit none
  character(len=*), intent(in) :: tag
  type(ccik_vector), intent(in) :: vec
  double precision, allocatable :: abscoef(:)
  integer, allocatable :: idx(:)
  integer :: i, nshow, pos
  integer, parameter :: trace_unit = 916

  if (.not. ccik_debug_trace) return
  if (vec%n <= 0) return

  allocate(abscoef(vec%n), idx(vec%n))
  do i = 1, vec%n
    abscoef(i) = dabs(vec%coef(i))
    idx(i) = i
  end do
  call dsort(abscoef, idx, vec%n)
  nshow = min(vec%n, max(0, ccik_debug_trace_topn))

  open(unit=trace_unit, file='ccik_debug_trace.log', status='unknown', position='append', action='write')
  write(trace_unit,'(A,1X,A,1X,I8)') 'CCIK_TRACE_VEC', trim(tag), nshow
  do i = 1, nshow
    pos = idx(vec%n - i + 1)
    if (size(vec%det,1) >= 1) then
      write(trace_unit,'(A,1X,A,1X,I8,1X,I0,1X,I0,1X,ES24.16,1X,ES24.16)') 'CCIK_TRACE_COEF', trim(tag), i, &
           vec%det(1,1,pos), vec%det(1,2,pos), vec%coef(pos), dabs(vec%coef(pos))
    end if
  end do
  close(trace_unit)
  deallocate(abscoef, idx)
end


subroutine ccik_debug_compare_vectors(tag, lhs, rhs)
  use ccik_sparse
  implicit none
  character(len=*), intent(in) :: tag
  type(ccik_vector), intent(in) :: lhs, rhs
  integer :: i, nmin, ndiff
  logical :: same_order
  double precision :: lhs_norm2, rhs_norm2, overlap, maxdiff
  character(len=256) :: trace_line

  if (.not. ccik_debug_trace) return

  nmin = min(lhs%n, rhs%n)
  ndiff = abs(lhs%n - rhs%n)
  same_order = (lhs%n == rhs%n)
  lhs_norm2 = 0.d0
  rhs_norm2 = 0.d0
  overlap = 0.d0
  maxdiff = 0.d0

  do i = 1, nmin
    lhs_norm2 = lhs_norm2 + lhs%coef(i) * lhs%coef(i)
    rhs_norm2 = rhs_norm2 + rhs%coef(i) * rhs%coef(i)
    overlap = overlap + lhs%coef(i) * rhs%coef(i)
    if (lhs%det(1,1,i) /= rhs%det(1,1,i) .or. lhs%det(1,2,i) /= rhs%det(1,2,i)) then
      same_order = .false.
      ndiff = ndiff + 1
    else
      maxdiff = max(maxdiff, dabs(lhs%coef(i) - rhs%coef(i)))
    end if
  end do

  do i = nmin + 1, lhs%n
    lhs_norm2 = lhs_norm2 + lhs%coef(i) * lhs%coef(i)
  end do
  do i = nmin + 1, rhs%n
    rhs_norm2 = rhs_norm2 + rhs%coef(i) * rhs%coef(i)
  end do

  write(trace_line,'(A,1X,A,1X,I8,1X,I8,1X,L1,1X,ES24.16,1X,ES24.16,1X,ES24.16,1X,ES24.16,1X,I8)') &
       'CCIK_TRACE_COMPARE', trim(tag), lhs%n, rhs%n, same_order, dsqrt(lhs_norm2), dsqrt(rhs_norm2), overlap, maxdiff, ndiff
  call ccik_debug_trace_line(trace_line)
end subroutine ccik_debug_compare_vectors


logical function ccik_det_entry_is_zero(det_src, idx)
  use bitmasks
  implicit none
  integer, intent(in) :: idx
  integer(bit_kind), intent(in) :: det_src(N_int,2,*)
  integer :: l, isp

  ccik_det_entry_is_zero = .true.
  do isp = 1, 2
    do l = 1, N_int
      if (det_src(l,isp,idx) /= 0_bit_kind) then
        ccik_det_entry_is_zero = .false.
        return
      end if
    end do
  end do
end function ccik_det_entry_is_zero


logical function ccik_same_det_entry(det_src, ia, ib)
  use bitmasks
  implicit none
  integer, intent(in) :: ia, ib
  integer(bit_kind), intent(in) :: det_src(N_int,2,*)
  integer :: l, isp

  ccik_same_det_entry = .true.
  do isp = 1, 2
    do l = 1, N_int
      if (det_src(l,isp,ia) /= det_src(l,isp,ib)) then
        ccik_same_det_entry = .false.
        return
      end if
    end do
  end do
end function ccik_same_det_entry


subroutine ccik_debug_trace_detset(tag, det_src, coef_src, nvec)
  use bitmasks
  implicit none
  character(len=*), intent(in) :: tag
  integer, intent(in) :: nvec
  integer(bit_kind), intent(in) :: det_src(N_int,2,*)
  double precision, intent(in) :: coef_src(psi_det_size,*)
  integer, parameter :: trace_unit = 916
  integer :: i, nshow
  integer(bit_kind) :: a1, a2, b1, b2

  if (.not. ccik_debug_trace) return
  if (nvec <= 0) return

  nshow = min(nvec, max(0, ccik_debug_trace_topn))
  open(unit=trace_unit, file='ccik_debug_trace.log', status='unknown', position='append', action='write')
  write(trace_unit,'(A,1X,A,1X,I8,1X,I8)') 'CCIK_TRACE_DETSET', trim(tag), nvec, nshow
  do i = 1, nshow
    a1 = det_src(1,1,i)
    b1 = det_src(1,2,i)
    a2 = 0_bit_kind
    b2 = 0_bit_kind
    if (N_int >= 2) then
      a2 = det_src(2,1,i)
      b2 = det_src(2,2,i)
    end if
    write(trace_unit,'(A,1X,A,1X,I8,1X,I0,1X,I0,1X,I0,1X,I0,1X,ES24.16)') 'CCIK_TRACE_DET', trim(tag), i, &
         a1, a2, b1, b2, coef_src(i,1)
  end do
  close(trace_unit)
end subroutine ccik_debug_trace_detset


subroutine ccik_validate_detset(tag, det_src, coef_src, nvec, max_size)
  use bitmasks
  implicit none
  character(len=*), intent(in) :: tag
  integer, intent(in) :: nvec, max_size
  integer(bit_kind), intent(in) :: det_src(N_int,2,*)
  double precision, intent(in) :: coef_src(psi_det_size,*)
  integer :: i, j, l, dup_limit, nalpha, nbeta
  logical, external :: ccik_det_entry_is_zero, ccik_same_det_entry

  if (nvec < 0 .or. nvec > max_size) then
    write(*,'(A)') 'ccik error: invalid determinant-set size before connected_sd_fullq internal pass'
    write(*,'(A,A,A,I10,A,I10)') 'ccik error: set=', trim(tag), '  nvec=', nvec, '  max=', max_size
    call flush(6)
    stop 2
  end if
  if (nvec <= 0) return

  do i = 1, nvec
    if (ccik_det_entry_is_zero(det_src, i)) then
      write(*,'(A)') 'ccik error: zero determinant encountered before connected_sd_fullq internal pass'
      write(*,'(A,A,A,I10)') 'ccik error: set=', trim(tag), '  slot=', i
      call flush(6)
      stop 2
    end if
    nalpha = 0
    nbeta = 0
    do l = 1, N_int
      nalpha = nalpha + popcnt(det_src(l,1,i))
      nbeta = nbeta + popcnt(det_src(l,2,i))
    end do
    if (nalpha /= elec_alpha_num .or. nbeta /= elec_beta_num) then
      write(*,'(A)') 'ccik error: invalid determinant electron count before connected_sd_fullq internal pass'
      write(*,'(A,A,A,I10,A,I10,A,I10,A,I10,A,I10)') 'ccik error: set=', trim(tag), '  slot=', i, &
           '  alpha=', nalpha, '  beta=', nbeta, '  expected_alpha=', elec_alpha_num, '  expected_beta=', elec_beta_num
      call flush(6)
      stop 2
    end if
    if (coef_src(i,1) /= coef_src(i,1)) then
      write(*,'(A)') 'ccik error: NaN coefficient encountered before connected_sd_fullq internal pass'
      write(*,'(A,A,A,I10)') 'ccik error: set=', trim(tag), '  slot=', i
      call flush(6)
      stop 2
    end if
  end do

  if (.not. ccik_debug_trace) return
  call ccik_debug_trace_detset(tag, det_src, coef_src, nvec)

  dup_limit = min(nvec, 4096)
  do i = 1, dup_limit
    do j = i + 1, dup_limit
      if (ccik_same_det_entry(det_src, i, j)) then
        write(*,'(A)') 'ccik error: duplicate determinant encountered before connected_sd_fullq internal pass'
        write(*,'(A,A,A,I10,A,I10)') 'ccik error: set=', trim(tag), '  first=', i, '  second=', j
        call flush(6)
        stop 2
      end if
    end do
  end do
end subroutine ccik_validate_detset


subroutine ccik_validate_order_array(tag, order_src, nvec)
  implicit none
  character(len=*), intent(in) :: tag
  integer, intent(in) :: nvec
  integer, intent(in) :: order_src(*)
  logical, allocatable :: seen(:)
  integer :: i, idx

  if (nvec <= 0) return
  allocate(seen(nvec))
  seen = .false.
  do i = 1, nvec
    idx = order_src(i)
    if (idx < 1 .or. idx > nvec) then
      write(*,'(A)') 'ccik error: invalid order-array entry before connected_sd_fullq internal pass'
      write(*,'(A,A,A,I10,A,I10)') 'ccik error: order=', trim(tag), '  slot=', i, '  idx=', idx
      call flush(6)
      stop 2
    end if
    if (seen(idx)) then
      write(*,'(A)') 'ccik error: duplicate order-array entry before connected_sd_fullq internal pass'
      write(*,'(A,A,A,I10,A,I10)') 'ccik error: order=', trim(tag), '  slot=', i, '  idx=', idx
      call flush(6)
      stop 2
    end if
    seen(idx) = .true.
  end do
  deallocate(seen)
end subroutine ccik_validate_order_array


subroutine ccik_check_connected_sd_fullq_provider_state(projected_call_count, fixed_budget_mode)
  use bitmasks
  implicit none
  integer, intent(in) :: projected_call_count
  logical, intent(in) :: fixed_budget_mode
  character(len=512) :: trace_line

  if (N_det <= 0 .or. N_det > psi_det_size) then
    write(*,'(A)') 'ccik error: invalid working-wavefunction size before connected_sd_fullq internal pass'
    write(*,'(A,I10,A,I10)') 'ccik error: N_det=', N_det, '  psi_det_size=', psi_det_size
    call flush(6)
    stop 2
  end if

  if (fixed_budget_mode) then
    if (N_det_propagators <= 0 .or. N_det_propagators > psi_det_size) then
      write(*,'(A)') 'ccik error: invalid propagator count before connected_sd_fullq internal pass'
      write(*,'(A,I10,A,I10)') 'ccik error: N_det_propagators=', N_det_propagators, '  psi_det_size=', psi_det_size
      call flush(6)
      stop 2
    end if
    if (N_det_generators <= 0 .or. N_det_generators > psi_det_size) then
      write(*,'(A)') 'ccik error: invalid generator count before connected_sd_fullq internal pass'
      write(*,'(A,I10,A,I10)') 'ccik error: N_det_generators=', N_det_generators, '  psi_det_size=', psi_det_size
      call flush(6)
      stop 2
    end if
    if (N_det_selectors <= 0 .or. N_det_selectors > psi_selectors_size) then
      write(*,'(A)') 'ccik error: invalid selector count before connected_sd_fullq internal pass'
      write(*,'(A,I10,A,I10)') 'ccik error: N_det_selectors=', N_det_selectors, '  psi_selectors_size=', psi_selectors_size
      call flush(6)
      stop 2
    end if
  end if

  if (ccik_debug_trace) then
    write(trace_line,'(A,1X,I8,1X,6I10)') 'CCIK_TRACE_PROVIDER_COUNTS', projected_call_count, &
         N_det, N_det_propagators, N_det_generators, N_det_selectors, psi_det_size, N_int
    call ccik_debug_trace_line(trace_line)
  end if

  call ccik_validate_detset('working', psi_det, psi_coef, N_det, psi_det_size)
  if (.not. fixed_budget_mode) return

  call ccik_validate_order_array('working_sorted_order', psi_det_working_sorted_order, N_det)
  call ccik_validate_detset('propagators', psi_det_propagators, psi_coef_propagators, N_det_propagators, psi_det_size)
  call ccik_validate_order_array('propagators_sorted_order', psi_det_propagators_sorted_order, N_det_propagators)
  call ccik_validate_detset('generators', psi_det_generators, psi_coef_generators, N_det_generators, psi_det_size)
  call ccik_validate_detset('selectors', psi_selectors, psi_selectors_coef, N_det_selectors, psi_selectors_size)
end subroutine ccik_check_connected_sd_fullq_provider_state


subroutine ccik_report_iteration_error(mode_label, cycle, iter_idx, electronic_energy)
  implicit none
  character(len=*), intent(in) :: mode_label
  integer, intent(in) :: cycle, iter_idx
  double precision, intent(in) :: electronic_energy
  double precision :: total_energy, delta_meh
  character(len=256) :: trace_line

  if (.not. ccik_print_iteration_error) return

  total_energy = electronic_energy + nuclear_repulsion
  delta_meh = (total_energy - ccik_reference_total_energy) * 1000.d0

  if (cycle > 0) then
    write(*,'(A,A,A,I4,A,I4,2(A,F24.12),A,F16.6)') 'ccik(', trim(mode_label), ') cycle=', cycle, &
         ' iter=', iter_idx, '  total=', total_energy, '  ref=', ccik_reference_total_energy, '  delta_mEh=', delta_meh
  else
    write(*,'(A,A,A,I4,2(A,F24.12),A,F16.6)') 'ccik(', trim(mode_label), ') iter=', iter_idx, &
         '  total=', total_energy, '  ref=', ccik_reference_total_energy, '  delta_mEh=', delta_meh
  end if
  call flush(6)

  write(trace_line,'(A,1X,A,1X,I8,1X,I8,1X,ES24.16,1X,ES24.16,1X,ES24.16)') &
       'CCIK_TRACE_ITERERR', trim(mode_label), cycle, iter_idx, total_energy, ccik_reference_total_energy, delta_meh
  call ccik_debug_trace_line(trace_line)
end subroutine ccik_report_iteration_error


integer function ccik_cumulative_new_krylov(cycle, iter_idx)
  implicit none
  integer, intent(in) :: cycle, iter_idx
  integer :: generated_in_cycle1, generated_in_later

  if (cycle <= 1) then
    ccik_cumulative_new_krylov = max(0, iter_idx - 1)
    return
  end if

  generated_in_cycle1 = max(0, ccik_restart_m_cycle - 1)
  generated_in_later = max(0, ccik_restart_m_cycle - max(1, ccik_restart_nroot))
  ccik_cumulative_new_krylov = generated_in_cycle1 + (cycle - 2) * generated_in_later + &
       max(0, iter_idx - max(1, ccik_restart_nroot))
end function ccik_cumulative_new_krylov


subroutine ccik_report_exactized_checkpoint(cycle, iter_idx, cum_new_krylov, support_size, electronic_energy, residual_norm, &
    solver_iters, matvec_count, converged, support_build_s, prep_s, solve_s, checkpoint_wall_s, checkpoint_cpu_s)
  implicit none
  integer, intent(in) :: cycle, iter_idx, cum_new_krylov, support_size, solver_iters, matvec_count
  logical, intent(in) :: converged
  double precision, intent(in) :: electronic_energy, residual_norm
  double precision, intent(in) :: support_build_s, prep_s, solve_s, checkpoint_wall_s, checkpoint_cpu_s
  double precision :: total_energy, delta_meh
  character(len=2048) :: trace_line
  character(len=1) :: conv_char
  character(len=32) :: s_cycle, s_iter, s_k, s_support, s_iters, s_matvecs
  character(len=32) :: s_converged
  character(len=40) :: s_electronic, s_total, s_ref, s_delta, s_resid
  character(len=40) :: s_union, s_prep, s_solve, s_wall, s_cpu

  if (.not. ccik_print_iteration_error) return

  total_energy = electronic_energy + nuclear_repulsion
  delta_meh = (total_energy - ccik_reference_total_energy) * 1000.d0
  conv_char = 'F'
  if (converged) conv_char = 'T'

  write(*,*) 'ccik(exactized) cycle=', cycle, ' iter=', iter_idx, ' k=', cum_new_krylov, ' support=', support_size, &
       ' total=', total_energy, ' ref=', ccik_reference_total_energy, ' delta_mEh=', delta_meh, &
       ' resid=', residual_norm, ' solver=davidson', ' iters=', solver_iters, ' matvecs=', matvec_count, &
       ' converged=', converged, ' t_union_s=', support_build_s, ' t_prep_s=', prep_s, ' t_solve_s=', solve_s, &
       ' t_wall_s=', checkpoint_wall_s, ' t_cpu_s=', checkpoint_cpu_s
  call flush(6)

  write(s_cycle,'(I0)') cycle
  write(s_iter,'(I0)') iter_idx
  write(s_k,'(I0)') cum_new_krylov
  write(s_support,'(I0)') support_size
  write(s_iters,'(I0)') solver_iters
  write(s_matvecs,'(I0)') matvec_count
  write(s_electronic,'(ES24.16)') electronic_energy
  write(s_total,'(ES24.16)') total_energy
  write(s_ref,'(ES24.16)') ccik_reference_total_energy
  write(s_delta,'(ES24.16)') delta_meh
  write(s_resid,'(ES24.16)') residual_norm
  write(s_union,'(ES24.16)') support_build_s
  write(s_prep,'(ES24.16)') prep_s
  write(s_solve,'(ES24.16)') solve_s
  write(s_wall,'(ES24.16)') checkpoint_wall_s
  write(s_cpu,'(ES24.16)') checkpoint_cpu_s
  s_converged = conv_char
  trace_line = 'CCIK_TRACE_EXACTIZED cycle=' // trim(adjustl(s_cycle)) // &
       ' iter=' // trim(adjustl(s_iter)) // &
       ' k=' // trim(adjustl(s_k)) // &
       ' support=' // trim(adjustl(s_support)) // &
       ' electronic=' // trim(adjustl(s_electronic)) // &
       ' total=' // trim(adjustl(s_total)) // &
       ' ref=' // trim(adjustl(s_ref)) // &
       ' delta_mEh=' // trim(adjustl(s_delta)) // &
       ' solver=davidson' // &
       ' iters=' // trim(adjustl(s_iters)) // &
       ' matvecs=' // trim(adjustl(s_matvecs)) // &
       ' converged=' // trim(adjustl(s_converged)) // &
       ' resid=' // trim(adjustl(s_resid)) // &
       ' t_union_s=' // trim(adjustl(s_union)) // &
       ' t_prep_s=' // trim(adjustl(s_prep)) // &
       ' t_solve_s=' // trim(adjustl(s_solve)) // &
       ' t_wall_s=' // trim(adjustl(s_wall)) // &
       ' t_cpu_s=' // trim(adjustl(s_cpu))
  write(*,'(A)') trim(trace_line)
  call flush(6)
  call ccik_debug_trace_line(trace_line)
end subroutine ccik_report_exactized_checkpoint


subroutine ccik_build_support_union(qbasis, n_basis, support_vec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: qbasis(*)
  integer, intent(in) :: n_basis
  type(ccik_vector), intent(inout) :: support_vec
  integer(bit_kind), allocatable :: det_tmp(:,:,:)
  double precision, allocatable :: coef_tmp(:)
  integer :: total_n, i, offset

  call ccik_vec_free(support_vec)
  total_n = 0
  do i = 1, n_basis
    total_n = total_n + max(0, qbasis(i)%n)
  end do
  if (total_n <= 0) return

  allocate(det_tmp(N_int,2,total_n), coef_tmp(total_n))
  offset = 0
  do i = 1, n_basis
    if (qbasis(i)%n <= 0) cycle
    det_tmp(:,:,offset+1:offset+qbasis(i)%n) = qbasis(i)%det(:,:,1:qbasis(i)%n)
    coef_tmp(offset+1:offset+qbasis(i)%n) = 1.d0
    offset = offset + qbasis(i)%n
  end do
  call ccik_vec_assign(support_vec, det_tmp, coef_tmp, total_n, N_int)
  deallocate(det_tmp, coef_tmp)
end subroutine ccik_build_support_union


subroutine ccik_write_support_union_snapshot(support_vec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: support_vec
  integer :: iunit, i, l
  character(len=512) :: snapshot_path

  snapshot_path = trim(ezfio_filename)//'/work/ccik_final_support_union.txt'
  open(newunit=iunit, file=trim(snapshot_path), status='replace', action='write', form='formatted')
  write(iunit,'(I0,1X,I0)') N_int, support_vec%n
  do i = 1, support_vec%n
    do l = 1, N_int
      write(iunit,'(I0,1X)',advance='no') support_vec%det(l,1,i)
    end do
    do l = 1, N_int
      write(iunit,'(I0,1X)',advance='no') support_vec%det(l,2,i)
    end do
    write(iunit,*)
  end do
  close(iunit)
  write(*,'(A)') 'CCIK_FINAL_SUPPORT_PATH = '//trim(snapshot_path)
  write(*,'(A,I0)') 'CCIK_FINAL_SUPPORT_SIZE = ', support_vec%n
end subroutine ccik_write_support_union_snapshot


subroutine ccik_vec_to_dense_on_support(support_vec, src_vec, support_dim, coeffs)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: support_vec, src_vec
  integer, intent(in) :: support_dim
  double precision, intent(out) :: coeffs(support_dim)
  integer :: isup, isrc, isp, l
  logical :: same

  coeffs = 0.d0
  if (support_dim <= 0 .or. src_vec%n <= 0) return

  do isrc = 1, src_vec%n
    do isup = 1, support_dim
      same = .true.
      do isp = 1, 2
        do l = 1, N_int
          if (support_vec%det(l,isp,isup) /= src_vec%det(l,isp,isrc)) then
            same = .false.
            exit
          end if
        end do
        if (.not. same) exit
      end do
      if (same) then
        coeffs(isup) = src_vec%coef(isrc)
        exit
      end if
    end do
  end do
end subroutine ccik_vec_to_dense_on_support


subroutine ccik_build_lowest_ritz_vector_raw(qbasis, hq_basis, n_basis, vec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: qbasis(*), hq_basis(*)
  integer, intent(in) :: n_basis
  type(ccik_vector), intent(inout) :: vec
  double precision, allocatable :: eig(:), eigvec(:,:)
  integer :: j
  logical :: ok

  call ccik_vec_free(vec)
  allocate(eig(n_basis), eigvec(n_basis,n_basis))
  call ccik_solve_ritz_problem(qbasis, hq_basis, n_basis, eig, eigvec)
  do j = 1, n_basis
    call ccik_vec_axpy(vec, eigvec(j,1), qbasis(j))
  end do
  call ccik_normalize_vector(vec, ok)
  if (.not. ok) then
    call ccik_vec_free(vec)
    call ccik_vec_copy(vec, qbasis(1))
  end if
  deallocate(eig, eigvec)
end subroutine ccik_build_lowest_ritz_vector_raw


subroutine ccik_count_exactized_cache(support_vec, support_dim, row_count, diag, total_mini)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: support_vec
  integer, intent(in) :: support_dim
  integer, intent(out) :: row_count(support_dim), total_mini
  double precision, intent(out) :: diag(support_dim)
  integer(bit_kind), allocatable :: mini_det_tmp(:,:,:)
  integer, allocatable :: idx_tmp(:)
  integer :: i, n_minilist
  double precision, external :: diag_H_mat_elem

  if (support_dim <= 0) return

  allocate(mini_det_tmp(N_int,2,support_dim), idx_tmp(support_dim))

  total_mini = 0
  do i = 1, support_dim
    call create_minilist(support_vec%det(1,1,i), support_vec%det, mini_det_tmp, idx_tmp, support_dim, n_minilist, N_int)
    row_count(i) = n_minilist
    total_mini = total_mini + n_minilist
    diag(i) = diag_H_mat_elem(support_vec%det(1,1,i), N_int)
  end do

  deallocate(mini_det_tmp, idx_tmp)
end subroutine ccik_count_exactized_cache


subroutine ccik_fill_exactized_cache(support_vec, support_dim, row_count, total_mini, row_offset, mini_det_flat, mini_idx_flat)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: support_vec
  integer, intent(in) :: support_dim, row_count(support_dim), total_mini
  integer, intent(out) :: row_offset(support_dim), mini_idx_flat(max(1,total_mini))
  integer(bit_kind), intent(out) :: mini_det_flat(N_int,2,max(1,total_mini))
  integer(bit_kind), allocatable :: mini_det_tmp(:,:,:)
  integer, allocatable :: idx_tmp(:)
  integer :: i, n_minilist, offset

  if (support_dim <= 0) return

  allocate(mini_det_tmp(N_int,2,support_dim), idx_tmp(support_dim))
  offset = 0
  do i = 1, support_dim
    row_offset(i) = offset + 1
    call create_minilist(support_vec%det(1,1,i), support_vec%det, mini_det_tmp, idx_tmp, support_dim, n_minilist, N_int)
    if (n_minilist > 0) then
      mini_det_flat(:,:,offset+1:offset+n_minilist) = mini_det_tmp(:,:,1:n_minilist)
      mini_idx_flat(offset+1:offset+n_minilist) = idx_tmp(1:n_minilist)
      offset = offset + n_minilist
    end if
  end do

  deallocate(mini_det_tmp, idx_tmp)
end subroutine ccik_fill_exactized_cache


subroutine ccik_apply_exactized_matvec_cached(support_vec, support_dim, row_offset, row_count, total_mini, mini_det_flat, mini_idx_flat, u, v)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: support_vec
  integer, intent(in) :: support_dim, total_mini
  integer, intent(in) :: row_offset(support_dim), row_count(support_dim), mini_idx_flat(max(1,total_mini))
  integer(bit_kind), intent(in) :: mini_det_flat(N_int,2,max(1,total_mini))
  double precision, intent(in) :: u(support_dim)
  double precision, intent(out) :: v(support_dim)
  double precision, allocatable :: coef_mat(:,:)
  double precision :: hpsi(1)
  integer(bit_kind), allocatable :: mini_det_row(:,:,:)
  integer, allocatable :: mini_idx_row(:)
  integer :: i, offset_i, count_i, max_row_count

  allocate(coef_mat(support_dim,1))
  coef_mat(:,1) = u(:)
  max_row_count = max(1, maxval(row_count))
  allocate(mini_det_row(N_int,2,max_row_count), mini_idx_row(max_row_count))
  v = 0.d0
  do i = 1, support_dim
    count_i = row_count(i)
    if (count_i <= 0) cycle
    offset_i = row_offset(i)
    mini_det_row(:,:,1:count_i) = mini_det_flat(:,:,offset_i:offset_i+count_i-1)
    mini_idx_row(1:count_i) = mini_idx_flat(offset_i:offset_i+count_i-1)
    call i_H_psi_minilist( &
         support_vec%det(1,1,i), &
         mini_det_row, &
         mini_idx_row, &
         count_i, &
         coef_mat, &
         N_int, &
         support_dim, &
         support_dim, &
         1, &
         hpsi &
    )
    v(i) = hpsi(1)
  end do
  deallocate(coef_mat, mini_det_row, mini_idx_row)
end subroutine ccik_apply_exactized_matvec_cached


subroutine ccik_exactized_lowest_root_davidson(support_vec, guess_vec, support_dim, row_offset, row_count, total_mini, mini_det_flat, mini_idx_flat, diag, &
    energy, residual_norm, davidson_iters, matvec_count, converged)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: support_vec, guess_vec
  integer, intent(in) :: support_dim, total_mini
  integer, intent(in) :: row_offset(support_dim), row_count(support_dim), mini_idx_flat(max(1,total_mini))
  integer(bit_kind), intent(in) :: mini_det_flat(N_int,2,max(1,total_mini))
  double precision, intent(in) :: diag(support_dim)
  double precision, intent(out) :: energy, residual_norm
  integer, intent(out) :: davidson_iters, matvec_count
  logical, intent(out) :: converged
  integer :: max_subspace, max_iter, restart_keep
  double precision :: tol, prev_energy, delta_energy, denom, vec_norm, alpha, energy_gap_tol
  double precision, allocatable :: dav_basis(:,:), dav_h_basis(:,:), dav_basis_new(:,:), dav_h_basis_new(:,:)
  double precision, allocatable :: hsmall(:,:), eigvals(:), work(:), trial(:), ritz(:), hritz(:), corr(:), coeff(:)
  integer :: m, iter, i, j, kkeep, info, lwork

  energy = 0.d0
  residual_norm = huge(1.d0)
  davidson_iters = 0
  matvec_count = 0
  converged = .false.
  if (support_dim <= 0) return

  max_subspace = max(2, ccik_exactized_max_subspace)
  max_iter = max(1, ccik_exactized_max_iter)
  restart_keep = max(1, min(ccik_exactized_restart_keep, max_subspace))
  tol = max(1.d-14, ccik_exactized_resid_tol)
  energy_gap_tol = max(1.d-12, 0.1d0 * tol)

  allocate(dav_basis(support_dim,max_subspace), dav_h_basis(support_dim,max_subspace))
  allocate(dav_basis_new(support_dim,restart_keep), dav_h_basis_new(support_dim,restart_keep))
  allocate(hsmall(max_subspace,max_subspace), eigvals(max_subspace), work(max(1, 3*max_subspace*max_subspace)))
  allocate(trial(support_dim), ritz(support_dim), hritz(support_dim), corr(support_dim), coeff(max_subspace))

  call ccik_vec_to_dense_on_support(support_vec, guess_vec, support_dim, trial)
  vec_norm = dsqrt(dot_product(trial, trial))
  if (vec_norm <= 0.d0) then
    trial = 0.d0
    trial(1) = 1.d0
    vec_norm = 1.d0
  end if
  dav_basis(:,1) = trial / vec_norm
  call ccik_apply_exactized_matvec_cached(support_vec, support_dim, row_offset, row_count, total_mini, mini_det_flat, mini_idx_flat, &
       dav_basis(:,1), dav_h_basis(:,1))
  matvec_count = 1
  m = 1
  prev_energy = huge(1.d0)

  do iter = 1, max_iter
    hsmall(1:m,1:m) = matmul(transpose(dav_basis(:,1:m)), dav_h_basis(:,1:m))
    do i = 1, m
      do j = i+1, m
        hsmall(i,j) = 0.5d0 * (hsmall(i,j) + hsmall(j,i))
        hsmall(j,i) = hsmall(i,j)
      end do
    end do

    lwork = max(1, 3*m*m)
    call dsyev('V', 'U', m, hsmall, max_subspace, eigvals, work, lwork, info)
    if (info /= 0) exit

    coeff(1:m) = hsmall(1:m,1)
    ritz = matmul(dav_basis(:,1:m), coeff(1:m))
    hritz = matmul(dav_h_basis(:,1:m), coeff(1:m))
    energy = eigvals(1)
    corr = hritz - energy * ritz
    residual_norm = dsqrt(max(0.d0, dot_product(corr, corr)))
    delta_energy = dabs(energy - prev_energy)
    davidson_iters = iter
    if (residual_norm <= tol .and. (iter == 1 .or. delta_energy <= energy_gap_tol)) then
      converged = .true.
      exit
    end if
    prev_energy = energy

    do i = 1, support_dim
      denom = diag(i) - energy
      if (dabs(denom) < 1.d-8) denom = sign(1.d-8, denom)
      corr(i) = corr(i) / denom
    end do
    do j = 1, m
      alpha = dot_product(dav_basis(:,j), corr)
      corr = corr - alpha * dav_basis(:,j)
    end do
    vec_norm = dsqrt(max(0.d0, dot_product(corr, corr)))
    if (vec_norm <= 1.d-12) exit
    corr = corr / vec_norm

    if (m < max_subspace) then
      m = m + 1
      dav_basis(:,m) = corr
      call ccik_apply_exactized_matvec_cached(support_vec, support_dim, row_offset, row_count, total_mini, mini_det_flat, mini_idx_flat, &
           dav_basis(:,m), dav_h_basis(:,m))
      matvec_count = matvec_count + 1
    else
      kkeep = min(restart_keep, m)
      dav_basis_new(:,1:kkeep) = matmul(dav_basis(:,1:m), hsmall(1:m,1:kkeep))
      dav_h_basis_new(:,1:kkeep) = matmul(dav_h_basis(:,1:m), hsmall(1:m,1:kkeep))
      dav_basis(:,1:kkeep) = dav_basis_new(:,1:kkeep)
      dav_h_basis(:,1:kkeep) = dav_h_basis_new(:,1:kkeep)
      m = kkeep
    end if
  end do

  deallocate(dav_basis, dav_h_basis, dav_basis_new, dav_h_basis_new, hsmall, eigvals, work, trial, ritz, hritz, corr, coeff)
end subroutine ccik_exactized_lowest_root_davidson


subroutine ccik_make_single_det_guess(support_vec, guess_vec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: support_vec
  type(ccik_vector), intent(inout) :: guess_vec
  integer(bit_kind) :: det_one(N_int,2,1)
  double precision :: coef_one(1)

  call ccik_vec_free(guess_vec)
  if (support_vec%n <= 0) return
  det_one(:,:,1) = support_vec%det(:,:,1)
  coef_one(1) = 1.d0
  call ccik_vec_assign(guess_vec, det_one, coef_one, 1, N_int)
end subroutine ccik_make_single_det_guess


subroutine ccik_maybe_run_exactized_checkpoint(qbasis, hq_basis, n_basis, cycle, iter_idx)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: qbasis(*), hq_basis(*)
  integer, intent(in) :: n_basis, cycle, iter_idx
  type(ccik_vector) :: support_vec, ritz_vec, guess_vec
  integer, allocatable :: row_offset(:), row_count(:), mini_idx_flat(:)
  integer(bit_kind), allocatable :: mini_det_flat(:,:,:)
  double precision, allocatable :: diag(:)
  integer :: checkpoint_period, cum_new_krylov, total_mini
  integer :: support_size, davidson_iters, matvec_count
  integer :: ccik_cumulative_new_krylov
  double precision :: exactized_energy, residual_norm
  double precision :: wall_total_0, wall_total_1, cpu_total_0, cpu_total_1
  double precision :: wall_union_0, wall_union_1, wall_prep_0, wall_prep_1, wall_solve_0, wall_solve_1
  logical :: converged, ok

  checkpoint_period = max(0, ccik_exactized_checkpoint_every)
  if (checkpoint_period <= 0) return
  cum_new_krylov = ccik_cumulative_new_krylov(cycle, iter_idx)
  if (cum_new_krylov <= 0) return
  if (mod(cum_new_krylov, checkpoint_period) /= 0) return

  call wall_time(wall_total_0)
  call cpu_time(cpu_total_0)

  call wall_time(wall_union_0)
  call ccik_build_support_union(qbasis, n_basis, support_vec)
  call wall_time(wall_union_1)
  support_size = support_vec%n
  if (support_size <= 0) then
    call ccik_vec_free(support_vec)
    return
  end if

  call wall_time(wall_prep_0)
  call ccik_build_lowest_ritz_vector_raw(qbasis, hq_basis, n_basis, ritz_vec)
  call ccik_vec_project_support(guess_vec, ritz_vec, support_vec)
  call ccik_normalize_vector(guess_vec, ok)
  if (.not. ok) then
    call ccik_make_single_det_guess(support_vec, guess_vec)
  end if
  allocate(row_offset(support_size), row_count(support_size), diag(support_size))
  call ccik_count_exactized_cache(support_vec, support_size, row_count, diag, total_mini)
  allocate(mini_det_flat(N_int,2,max(1,total_mini)), mini_idx_flat(max(1,total_mini)))
  call ccik_fill_exactized_cache(support_vec, support_size, row_count, total_mini, row_offset, mini_det_flat, mini_idx_flat)
  call wall_time(wall_prep_1)

  call wall_time(wall_solve_0)
  call ccik_exactized_lowest_root_davidson( &
       support_vec, &
       guess_vec, &
       support_size, &
       row_offset, &
       row_count, &
       total_mini, &
       mini_det_flat, &
       mini_idx_flat, &
       diag, &
       exactized_energy, &
       residual_norm, &
       davidson_iters, &
       matvec_count, &
       converged &
  )
  call wall_time(wall_solve_1)

  call cpu_time(cpu_total_1)
  call wall_time(wall_total_1)
  call ccik_report_exactized_checkpoint( &
       cycle, &
       iter_idx, &
       cum_new_krylov, &
       support_size, &
       exactized_energy, &
       residual_norm, &
       davidson_iters, &
       matvec_count, &
       converged, &
       wall_union_1 - wall_union_0, &
       wall_prep_1 - wall_prep_0, &
       wall_solve_1 - wall_solve_0, &
       wall_total_1 - wall_total_0, &
       cpu_total_1 - cpu_total_0 &
  )

  if (allocated(row_offset)) deallocate(row_offset)
  if (allocated(row_count)) deallocate(row_count)
  if (allocated(mini_idx_flat)) deallocate(mini_idx_flat)
  if (allocated(mini_det_flat)) deallocate(mini_det_flat)
  if (allocated(diag)) deallocate(diag)
  call ccik_vec_free(support_vec)
  call ccik_vec_free(ritz_vec)
  call ccik_vec_free(guess_vec)
end subroutine ccik_maybe_run_exactized_checkpoint


subroutine ccik_thick_restart_driver
  use ccik_sparse
  implicit none
  integer :: ccik_thick_effective_nkeep
  type(ccik_vector), allocatable :: qbasis(:), hq_basis(:), restart_basis(:)
  type(ccik_vector) :: q0, final_vec, final_support
  integer :: cycle, i, n_basis, n_keep, m_cycle, nkeep_eff, target_size
  integer :: queue_len, queue_pos, parent_idx
  double precision :: energy, prev_cycle_energy, total_energy, rayleigh, pt2_corr, rpt2_corr
  double precision :: t_build_start, t_build_stop
  logical :: converged, ok
  integer, allocatable :: expand_queue(:)

  call ezfio_set_determinants_read_wf(ccik_use_existing_wf)

  m_cycle = max(2, ccik_restart_m_cycle)
  nkeep_eff = ccik_thick_effective_nkeep(m_cycle, max(1, ccik_restart_nroot))
  target_size = max(2, max(nkeep_eff, ccik_selector_keep + ccik_external_keep))
  call ccik_prepare_working_capacity(target_size)
  call ccik_debug_trace_reset('thick_restart')
  if (ccik_verbose .or. ccik_total_det_budget > 0) then
    print *, 'ccik_thick_effective_nkeep = ', nkeep_eff
  end if

  PROVIDE all_mo_integrals psi_det psi_coef
  call ccik_capture_current_vector(q0)
  if (q0%n == 0) then
    print *, 'ccik(thick): failed to initialize the reference vector'
    stop 2
  end if
  call ccik_normalize_vector(q0, ok)
  if (.not. ok) then
    print *, 'ccik(thick): zero-norm initial vector'
    stop 2
  end if

  allocate(qbasis(m_cycle), hq_basis(m_cycle), restart_basis(max(1, ccik_restart_nroot)))
  allocate(expand_queue(m_cycle))
  do i = 1, m_cycle
    call ccik_vec_free(qbasis(i))
    call ccik_vec_free(hq_basis(i))
  end do
  do i = 1, max(1, ccik_restart_nroot)
    call ccik_vec_free(restart_basis(i))
  end do
  call ccik_vec_copy(restart_basis(1), q0)
  call ccik_vec_free(q0)

  prev_cycle_energy = huge(1.d0)
  n_basis = 0
  converged = .false.

  do cycle = 1, ccik_max_cycle
    do i = 1, m_cycle
      call ccik_vec_free(qbasis(i))
      call ccik_vec_free(hq_basis(i))
    end do

    n_basis = 0
    do i = 1, max(1, ccik_restart_nroot)
      if (restart_basis(i)%n == 0) exit
      n_basis = n_basis + 1
      call ccik_vec_copy(qbasis(n_basis), restart_basis(i))
    end do
    if (n_basis == 0) exit

    do i = 1, n_basis
      call ccik_load_vector_into_working_wf(qbasis(i))
      if (ccik_verbose) then
        write(*,'(A,I4,A,I4,2(A,I10))') 'ccik(thick) cycle=', cycle, &
             ' seed=', i, '  |q|=', qbasis(i)%n, '  nkeep=', nkeep_eff
        call flush(6)
      end if
      call cpu_time(t_build_start)
      call ccik_build_projected_hq(hq_basis(i), rayleigh)
      call cpu_time(t_build_stop)
      if (ccik_verbose) then
        write(*,'(A,I4,A,I4,A,I10,A,F12.4)') 'ccik(thick) cycle=', cycle, &
             ' seed_done=', i, '  |hq|=', hq_basis(i)%n, '  cpu_s=', t_build_stop - t_build_start
        call flush(6)
      end if
    end do

    call ccik_smallest_ritz_energy(qbasis, hq_basis, n_basis, energy)
    call ccik_report_iteration_error('thick', cycle, n_basis, energy)
    call ccik_maybe_run_exactized_checkpoint(qbasis, hq_basis, n_basis, cycle, n_basis)
    if (ccik_verbose) then
      write(*,'(A,I4,A,I4,A,F24.12)') 'ccik(thick) cycle=', cycle, &
           ' restart_basis=', n_basis, '  ritz=', energy
    end if
    queue_len = n_basis
    queue_pos = 1
    do i = 1, n_basis
      expand_queue(i) = i
    end do

    do while (n_basis < m_cycle .and. queue_pos <= queue_len)
      parent_idx = expand_queue(queue_pos)
      queue_pos = queue_pos + 1

      call ccik_generate_next_vector(qbasis, hq_basis, parent_idx, n_basis, nkeep_eff, qbasis(n_basis+1), ok)
      if (.not. ok) exit
      n_basis = n_basis + 1
      queue_len = queue_len + 1
      expand_queue(queue_len) = n_basis

      call ccik_load_vector_into_working_wf(qbasis(n_basis))
      if (ccik_verbose) then
        write(*,'(A,I4,A,I4,A,I4,2(A,I10))') 'ccik(thick) cycle=', cycle, &
             ' build_hq_for_iter=', n_basis, '  parent=', parent_idx, &
             '  |q|=', qbasis(n_basis)%n, '  nkeep=', nkeep_eff
        call flush(6)
      end if
      call cpu_time(t_build_start)
      call ccik_build_projected_hq(hq_basis(n_basis), rayleigh)
      call cpu_time(t_build_stop)
      call ccik_smallest_ritz_energy(qbasis, hq_basis, n_basis, energy)
      call ccik_report_iteration_error('thick', cycle, n_basis, energy)
      call ccik_maybe_run_exactized_checkpoint(qbasis, hq_basis, n_basis, cycle, n_basis)

      if (ccik_verbose) then
        write(*,'(A,I4,A,I4,2(A,F24.12),2(A,I10))') 'ccik(thick) cycle=', cycle, &
             ' iter=', n_basis, '  ritz=', energy, '  rayleigh=', rayleigh, &
             '  |q|=', qbasis(n_basis)%n, '  |hq|=', hq_basis(n_basis)%n
        write(*,'(A,F12.4)') 'ccik(thick) projected_hq_cpu_s=', t_build_stop - t_build_start
        call flush(6)
      end if

    end do

    if (cycle > 1) then
      converged = abs(energy - prev_cycle_energy) < ccik_conv_tol
      if (converged) exit
    end if
    prev_cycle_energy = energy

    if (converged) exit
    n_keep = min(ccik_restart_nroot, n_basis)
    call ccik_restart_from_ritz(qbasis, hq_basis, n_basis, nkeep_eff, restart_basis, n_keep, max(1, ccik_restart_nroot), cycle)
    if (n_keep <= 0) exit
  end do

  total_energy = energy + nuclear_repulsion
  if (ccik_save_final_wf) then
    call ccik_build_support_union(qbasis, n_basis, final_support)
    call ccik_write_support_union_snapshot(final_support)
    call ccik_vec_free(final_support)
    call ccik_build_lowest_ritz_vector(qbasis, hq_basis, n_basis, final_vec)
    call ccik_save_vector_to_ezfio(final_vec)
    if (ccik_do_post_pt2) then
      call ccik_compute_post_pt2(final_vec, energy, pt2_corr, rpt2_corr)
      write(*,'(A,F24.12)') 'CCIK_FINAL_PT2_CORRECTION = ', pt2_corr
      write(*,'(A,F24.12)') 'CCIK_FINAL_E_PLUS_PT2 = ', total_energy + pt2_corr
      write(*,'(A,F24.12)') 'CCIK_FINAL_E_PLUS_RPT2 = ', total_energy + rpt2_corr
    end if
    call ccik_vec_free(final_vec)
  end if
  write(*,'(A,F24.12)') 'CCIK_FINAL_ELECTRONIC_ENERGY = ', energy
  write(*,'(A,F24.12)') 'CCIK_FINAL_TOTAL_ENERGY = ', total_energy
  write(*,'(A,I8)') 'CCIK_FINAL_BASIS = ', n_basis

  do i = 1, m_cycle
    call ccik_vec_free(qbasis(i))
    call ccik_vec_free(hq_basis(i))
  end do
  do i = 1, max(1, ccik_restart_nroot)
    call ccik_vec_free(restart_basis(i))
  end do
  deallocate(qbasis, hq_basis, restart_basis, expand_queue)
end


subroutine ccik_baseline_driver
  use ccik_sparse
  implicit none
  type(ccik_vector), allocatable :: qbasis(:), hq_basis(:)
  type(ccik_vector) :: q0, qnext, final_vec, final_support
  integer :: i, n_basis, target_size
  double precision :: energy, prev_energy, total_energy, rayleigh, pt2_corr, rpt2_corr
  logical :: converged, ok

  call ezfio_set_determinants_read_wf(ccik_use_existing_wf)

  target_size = max(2, max(ccik_nkeep, ccik_selector_keep + ccik_external_keep))
  call ccik_prepare_working_capacity(target_size)
  call ccik_debug_trace_reset('baseline')

  PROVIDE all_mo_integrals psi_det psi_coef
  call ccik_capture_current_vector(q0)
  if (q0%n == 0) then
    print *, 'ccik: failed to initialize the reference vector'
    stop 2
  end if
  call ccik_normalize_vector(q0, ok)
  if (.not. ok) then
    print *, 'ccik: zero-norm initial vector'
    stop 2
  end if

  allocate(qbasis(ccik_basis_size), hq_basis(ccik_basis_size))
  do i = 1, ccik_basis_size
    call ccik_vec_free(qbasis(i))
    call ccik_vec_free(hq_basis(i))
  end do
  call ccik_vec_copy(qbasis(1), q0)
  call ccik_vec_free(q0)

  prev_energy = huge(1.d0)
  n_basis = 0
  converged = .false.

  do i = 1, ccik_basis_size
    if (qbasis(i)%n == 0) exit
    n_basis = i

    call ccik_load_vector_into_working_wf(qbasis(i))
    rayleigh = 0.d0
    call ccik_build_projected_hq(hq_basis(i), rayleigh)

    call ccik_smallest_ritz_energy(qbasis, hq_basis, n_basis, energy)
    call ccik_report_iteration_error('baseline', 0, n_basis, energy)
    if (ccik_verbose) then
      write(*,'(A,I4,2(A,F24.12),2(A,I10))') 'ccik iter=', n_basis, &
           '  ritz=', energy, '  rayleigh=', rayleigh, &
           '  |q|=', qbasis(i)%n, '  |hq|=', hq_basis(i)%n
    end if

    if (n_basis > 1) then
      converged = abs(energy - prev_energy) < ccik_conv_tol
      if (converged) exit
    end if
    prev_energy = energy

    if (i == ccik_basis_size) exit

    call ccik_generate_next_vector(qbasis, hq_basis, i, n_basis, ccik_nkeep, qnext, ok)
    if (.not. ok) exit
    call ccik_vec_copy(qbasis(i+1), qnext)
    call ccik_vec_free(qnext)
  end do

  total_energy = energy + nuclear_repulsion
  if (ccik_save_final_wf) then
    call ccik_build_support_union(qbasis, n_basis, final_support)
    call ccik_write_support_union_snapshot(final_support)
    call ccik_vec_free(final_support)
    call ccik_build_lowest_ritz_vector(qbasis, hq_basis, n_basis, final_vec)
    call ccik_save_vector_to_ezfio(final_vec)
    if (ccik_do_post_pt2) then
      call ccik_compute_post_pt2(final_vec, energy, pt2_corr, rpt2_corr)
      write(*,'(A,F24.12)') 'CCIK_FINAL_PT2_CORRECTION = ', pt2_corr
      write(*,'(A,F24.12)') 'CCIK_FINAL_E_PLUS_PT2 = ', total_energy + pt2_corr
      write(*,'(A,F24.12)') 'CCIK_FINAL_E_PLUS_RPT2 = ', total_energy + rpt2_corr
    end if
    call ccik_vec_free(final_vec)
  end if
  write(*,'(A,F24.12)') 'CCIK_FINAL_ELECTRONIC_ENERGY = ', energy
  write(*,'(A,F24.12)') 'CCIK_FINAL_TOTAL_ENERGY = ', total_energy
  write(*,'(A,I8)') 'CCIK_FINAL_BASIS = ', n_basis

  do i = 1, ccik_basis_size
    call ccik_vec_free(qbasis(i))
    call ccik_vec_free(hq_basis(i))
  end do
  deallocate(qbasis, hq_basis)
end


subroutine ccik_prepare_working_capacity(target_size)
  implicit none
  integer, intent(in) :: target_size
  integer :: required_size, copy_n
  integer(bit_kind), allocatable :: det_copy(:,:,:)
  double precision, allocatable :: coef_copy(:,:)

  required_size = max(1, target_size)
  PROVIDE psi_det_size psi_det psi_coef N_states N_det
  if (psi_det_size >= required_size) return

  copy_n = min(N_det, size(psi_det,3))
  if (copy_n > 0) then
    allocate(det_copy(N_int,2,copy_n), coef_copy(copy_n,N_states))
    det_copy = psi_det(:,:,1:copy_n)
    coef_copy = psi_coef(1:copy_n,1:N_states)
  end if

  psi_det_size = required_size
  if (allocated(psi_det)) then
    deallocate(psi_det)
  end if
  allocate(psi_det(N_int,2,psi_det_size))
  if (allocated(psi_coef)) then
    deallocate(psi_coef)
  end if
  allocate(psi_coef(psi_det_size,N_states))
  psi_det = 0_bit_kind
  psi_coef = 0.d0
  if (copy_n > 0) then
    psi_det(:,:,1:copy_n) = det_copy(:,:,1:copy_n)
    psi_coef(1:copy_n,1:N_states) = coef_copy(1:copy_n,1:N_states)
    deallocate(det_copy, coef_copy)
  end if
end subroutine ccik_prepare_working_capacity


subroutine ccik_capture_current_vector(vec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(inout) :: vec
  integer(bit_kind), allocatable :: det_tmp(:,:,:)
  double precision, allocatable :: coef_tmp(:)

  call ccik_vec_free(vec)
  if (N_det <= 0) return

  allocate(det_tmp(N_int,2,N_det), coef_tmp(N_det))
  det_tmp = psi_det(:,:,1:N_det)
  coef_tmp = psi_coef(1:N_det,1)
  call ccik_vec_assign(vec, det_tmp, coef_tmp, N_det, N_int)
  deallocate(det_tmp, coef_tmp)
end subroutine ccik_capture_current_vector


subroutine ccik_capture_provider_vector(vec, det_src, coef_src, nvec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(inout) :: vec
  integer, intent(in) :: nvec
  integer(bit_kind), intent(in) :: det_src(N_int,2,*)
  double precision, intent(in) :: coef_src(psi_det_size,*)
  integer(bit_kind), allocatable :: det_tmp(:,:,:)
  double precision, allocatable :: coef_tmp(:)

  call ccik_vec_free(vec)
  if (nvec <= 0) return

  allocate(det_tmp(N_int,2,nvec), coef_tmp(nvec))
  det_tmp = det_src(:,:,1:nvec)
  coef_tmp = coef_src(1:nvec,1)
  call ccik_vec_assign(vec, det_tmp, coef_tmp, nvec, N_int)
  deallocate(det_tmp, coef_tmp)
end subroutine ccik_capture_provider_vector


subroutine ccik_load_vector_into_working_wf(vec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: vec
  type(ccik_vector) :: prop_vec, roundtrip_vec
  integer(bit_kind), allocatable :: det_tmp(:,:,:)
  double precision, allocatable :: coef_tmp(:,:)
  integer :: nload
  character(len=32) :: compression_policy
  character(len=256) :: trace_line

  if (N_states /= 1) then
    print *, 'ccik error: working-wavefunction reload currently supports only N_states=1'
    stop 2
  end if

  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'
  nload = max(1, vec%n)
  N_det = nload
  psi_det(:,:,1:nload) = 0_bit_kind
  psi_coef(1:nload,1:N_states) = 0.d0
  if (vec%n > 0) then
    psi_det(:,:,1:vec%n) = vec%det(:,:,1:vec%n)
    psi_coef(1:vec%n,1) = vec%coef(1:vec%n)
  end if
  SOFT_TOUCH N_det psi_det psi_coef

  if (ccik_debug_trace) then
    write(trace_line,'(A,1X,A,1X,I8,1X,I8)') 'CCIK_TRACE_LOAD', trim(compression_policy), vec%n, nload
    call ccik_debug_trace_line(trace_line)
    call ccik_debug_trace_vector('load_input', vec)
    call ccik_capture_current_vector(roundtrip_vec)
    call ccik_debug_trace_vector('load_roundtrip', roundtrip_vec)
    call ccik_debug_compare_vectors('load_roundtrip_cmp', vec, roundtrip_vec)
    call ccik_vec_free(roundtrip_vec)
  end if

  call ccik_vec_free(prop_vec)
  if (vec%prop_n > 0) then
    call ccik_vec_assign(prop_vec, vec%prop_det, vec%prop_coef, vec%prop_n, size(vec%prop_det,1))
    call ccik_runtime_set_propagation_override(prop_vec)
  else
    call ccik_runtime_clear_propagation_override()
  end if
  call ccik_vec_free(prop_vec)
end subroutine ccik_load_vector_into_working_wf


subroutine ccik_refresh_working_wf_providers()
  implicit none
end subroutine ccik_refresh_working_wf_providers


subroutine ccik_build_projected_hq(vec, rayleigh)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(inout) :: vec
  double precision, intent(out) :: rayleigh
  integer, save :: projected_call_count = 0
  character(len=32) :: matvec_backend, compression_policy

  projected_call_count = projected_call_count + 1
  matvec_backend = trim(adjustl(ccik_matvec_backend))
  if (len_trim(matvec_backend) == 0) matvec_backend = 'connected_sd_fullq'
  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'
  if (compression_policy == 'threshold') then
    matvec_backend = 'connected_sd_fullq'
  else if (compression_policy == 'fixed_budget') then
    if (.not. ccik_debug_allow_legacy_fixed_budget) matvec_backend = 'connected_sd_fullq'
  end if

  select case (trim(matvec_backend))
  case ('connected_sd_fullq')
    call ccik_build_projected_hq_connected_sd_fullq(vec, rayleigh, projected_call_count, matvec_backend)
  case default
    call ccik_build_projected_hq_legacy(vec, rayleigh, projected_call_count, matvec_backend)
  end select
end subroutine ccik_build_projected_hq


subroutine ccik_build_projected_hq_legacy(vec, rayleigh, projected_call_count, matvec_backend)
  use selection_types
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(inout) :: vec
  double precision, intent(out) :: rayleigh
  integer, intent(in) :: projected_call_count
  character(len=*), intent(in) :: matvec_backend
  type(selection_buffer) :: buf
  type(pt2_type) :: pt2_data
  integer(bit_kind), allocatable :: det_tmp(:,:,:), mini_det(:,:,:)
  double precision, allocatable :: coef_tmp(:), hpsi_internal(:), ext_coef(:), score_sort(:), pt2_score(:)
  integer, allocatable :: ext_order(:), idx_minilist(:)
  double precision :: e0(1), hpsi(1)
  double precision, external :: diag_H_mat_elem
  integer :: i, j, n_ext, n_total, n_candidates, n_minilist, candidate_keep, tmp_keep_est
  integer :: ccik_tmp_keep_value
  double precision :: hdiag_i, score_denom, max_proj_score, max_pt2_score, pt2_val, abscoef
  double precision :: t0, t1, t_select, t_sort, t_internal, t_external, t_assign
  character(len=32) :: selector_mode
  character(len=512) :: trace_line

  call ccik_vec_free(vec)
  rayleigh = 0.d0
  t_select = 0.d0
  t_sort = 0.d0
  t_internal = 0.d0
  t_external = 0.d0
  t_assign = 0.d0

  if (ccik_verbose) then
    write(*,'(A,I6,1X,A,1X,A,5(A,I10),A,F24.12)') 'ccik(hq) begin call=', projected_call_count, &
         'backend=', trim(matvec_backend), &
         '  |q|=', N_det, '  ngen=', N_det_generators, '  nsel=', N_det_selectors, &
         '  ext_keep=', ccik_external_keep, '  ext_factor=', ccik_external_candidate_factor, &
         '  rayleigh=', rayleigh
    call flush(6)
  end if

  allocate(hpsi_internal(max(1, N_det)))
  call cpu_time(t0)
  do i = 1, N_det
    call i_H_psi(psi_det(1,1,i), psi_selectors, psi_selectors_coef, N_int, &
         N_det_selectors, psi_selectors_size, 1, hpsi)
    hpsi_internal(i) = hpsi(1)
    rayleigh = rayleigh + psi_coef(i,1) * hpsi(1)
  end do
  call cpu_time(t1)
  t_internal = t1 - t0
  e0(1) = rayleigh
  selector_mode = trim(adjustl(ccik_selector_mode))
  if (len_trim(selector_mode) == 0) selector_mode = 'abs_residual'
  if (selector_mode == 'pt2_legacy') selector_mode = 'pt2_residual'

  if (ccik_verbose) then
    write(*,'(A,I6,A,F24.12,A,F12.4)') 'ccik(hq) internal pass call=', projected_call_count, &
         '  rayleigh=', rayleigh, '  cpu_s=', t_internal
    call flush(6)
  end if

  tmp_keep_est = ccik_tmp_keep_value(max(1, ccik_nkeep))
  candidate_keep = max(1, ccik_external_candidate_factor * max(1, max(ccik_external_keep, tmp_keep_est)))
  call create_selection_buffer(candidate_keep, max(2, 2*candidate_keep), buf)
  call pt2_alloc(pt2_data, N_states)

  call cpu_time(t0)
  do i = 1, N_det_generators
    call select_connected(i, e0, pt2_data, buf, 1, 1)
  end do
  call cpu_time(t1)
  t_select = t1 - t0

  call cpu_time(t0)
  call sort_selection_buffer(buf)
  call remove_duplicates_in_selection_buffer(buf)
  call cpu_time(t1)
  t_sort = t1 - t0

  n_candidates = min(candidate_keep, buf%cur)
  if (n_candidates <= 0) then
    allocate(det_tmp(N_int,2,max(1, N_det)), coef_tmp(max(1, N_det)))
    do i = 1, N_det
      det_tmp(:,:,i) = psi_det(:,:,i)
      coef_tmp(i) = hpsi_internal(i)
    end do
    call ccik_vec_assign(vec, det_tmp, coef_tmp, N_det, N_int)
    deallocate(det_tmp, coef_tmp)
    deallocate(hpsi_internal)
    call pt2_dealloc(pt2_data)
    call delete_selection_buffer(buf)
    return
  end if
  n_ext = min(max(ccik_external_keep, tmp_keep_est), n_candidates)
  n_total = N_det + n_ext
  allocate(det_tmp(N_int,2,n_total), coef_tmp(n_total))
  allocate(ext_coef(max(1, n_candidates)), pt2_score(max(1, n_candidates)))
  allocate(score_sort(max(1, n_candidates)), ext_order(max(1, n_candidates)))
  allocate(mini_det(N_int,2,max(1, N_det_selectors)), idx_minilist(max(1, N_det_selectors)))

  do i = 1, N_det
    det_tmp(:,:,i) = psi_det(:,:,i)
    coef_tmp(i) = hpsi_internal(i)
  end do

  call cpu_time(t0)
  do i = 1, n_candidates
    call create_minilist(buf%det(1,1,i), psi_selectors, mini_det, idx_minilist, N_det_selectors, n_minilist, N_int)
    if (n_minilist > 0) then
      call i_H_psi_minilist(buf%det(1,1,i), mini_det, idx_minilist, n_minilist, psi_selectors_coef, &
           N_int, n_minilist, psi_selectors_size, 1, hpsi)
    else
      hpsi(1) = 0.d0
    end if
    ext_coef(i) = hpsi(1)
    abscoef = dabs(hpsi(1))
    if (selector_mode == 'pt2_legacy') then
      hdiag_i = diag_H_mat_elem(buf%det(1,1,i), N_int)
      score_denom = max(1.d-8, dabs(rayleigh - hdiag_i))
      score_sort(i) = abscoef*abscoef / score_denom
      pt2_val = -min(buf%val(i), 0.d0)
      pt2_score(i) = pt2_val
    else if (selector_mode == 'sq_hq' .or. selector_mode == 'sq_residual') then
      score_sort(i) = abscoef*abscoef
      pt2_score(i) = 0.d0
    else
      score_sort(i) = abscoef
      pt2_score(i) = 0.d0
    end if
    ext_order(i) = i
  end do
  call cpu_time(t1)
  t_external = t1 - t0

  if (selector_mode == 'pt2_legacy') then
    max_proj_score = max(1.d-30, maxval(score_sort(1:n_candidates)))
    max_pt2_score = max(1.d-30, maxval(pt2_score(1:n_candidates)))
    do i = 1, n_candidates
      score_sort(i) = score_sort(i) / max_proj_score
      if (ccik_pt2_rank_weight > 0.d0) then
        score_sort(i) = score_sort(i) + ccik_pt2_rank_weight * pt2_score(i) / max_pt2_score
      end if
    end do
  end if

  if (n_candidates > 1) then
    call dsort(score_sort, ext_order, n_candidates)
  end if

  do i = 1, n_ext
    j = ext_order(n_candidates - n_ext + i)
    det_tmp(:,:,N_det+i) = buf%det(:,:,j)
    coef_tmp(N_det+i) = ext_coef(j)
  end do

  call cpu_time(t0)
  call ccik_vec_assign(vec, det_tmp, coef_tmp, n_total, N_int)
  call cpu_time(t1)
  t_assign = t1 - t0

  if (ccik_verbose) then
    write(*,'(A,I6,1X,A,1X,A,4(A,I10),A,A,5(A,F12.4))') 'ccik(hq) done  call=', projected_call_count, &
         'backend=', trim(matvec_backend), &
         '  buf=', buf%cur, '  n_ext=', n_ext, '  n_total=', n_total, '  |vec|=', vec%n, &
         '  selector=', trim(selector_mode), &
         '  select_s=', t_select, '  sort_s=', t_sort, '  int_s=', t_internal, &
         '  ext_s=', t_external, '  assign_s=', t_assign
    call flush(6)
  end if

  if (ccik_debug_trace) then
    write(trace_line,'(A,1X,I8,1X,A,1X,A,1X,I8,1X,I8,1X,I8,1X,ES24.16,1X,ES24.16,1X,ES24.16)') &
         'CCIK_TRACE_HQ', projected_call_count, trim(selector_mode), trim(matvec_backend), N_det, n_candidates, n_ext, rayleigh, &
         dsqrt(max(0.d0, sum(hpsi_internal(1:max(1,N_det))**2))), dsqrt(max(0.d0, ccik_vec_norm2(vec)))
    call ccik_debug_trace_line(trace_line)
    call ccik_debug_trace_vector('projected_hq', vec)
  end if

  deallocate(det_tmp, coef_tmp)
  deallocate(ext_coef, pt2_score, score_sort, ext_order, mini_det, idx_minilist)
  deallocate(hpsi_internal)
  call pt2_dealloc(pt2_data)
  call delete_selection_buffer(buf)
end subroutine ccik_build_projected_hq_legacy


subroutine ccik_build_connected_sd_candidates(candidate_vec, generated_before_dedup, generated_unique)
  use ccik_sparse
  use bitmasks
  implicit none
  type(ccik_vector), intent(inout) :: candidate_vec
  integer, intent(out) :: generated_before_dedup, generated_unique
  integer :: occ_a(max(1, mo_num)), occ_b(max(1, mo_num)), virt_a(max(1, mo_num)), virt_b(max(1, mo_num))
  integer :: occ_a_act(max(1, mo_num)), occ_b_act(max(1, mo_num))
  logical :: occ_mask_a(max(1, mo_num)), occ_mask_b(max(1, mo_num)), act_mask(max(1, mo_num))
  integer(bit_kind), allocatable :: det_work(:,:,:)
  double precision, allocatable :: coef_work(:)
  integer(bit_kind) :: det_trial(N_int,2)
  integer, allocatable :: parent_cap(:), parent_offset(:), parent_actual(:)
  integer*8, allocatable :: parent_cap64(:)
  integer :: ipar, i, j, k, l, base, local_count, copy_idx
  integer :: n_occ_a, n_occ_b, n_vir_a, n_vir_b
  integer :: n_occ_a_act, n_occ_b_act
  integer :: n_space_orb, orb_i
  integer :: i_ok1, i_ok2, nraw, raw_cap
  integer*8 :: raw_cap64, raw_candidate_bytes, parent_raw_cap64, raw_cap_unclipped64, remaining64
  integer*8, parameter :: threshold_candidate_limit_bytes = 4294967296_8
  integer*8, parameter :: threshold_candidate_limit_det = 1000000_8
  character(len=32) :: compression_policy
  logical :: threshold_mode, fixed_budget_mode, sparse_mode, threshold_candidate_clipped
  logical, external :: is_in_wavefunction

  call ccik_vec_free(candidate_vec)
  generated_before_dedup = 0
  generated_unique = 0
  if (N_det_generators <= 0 .or. mo_num <= 0) return

  PROVIDE psi_det_generators psi_det_sorted_bit n_act_orb list_act list_act_reverse psi_det_propagators_sorted psi_det_working_sorted
  allocate(parent_cap(max(1, N_det_generators)))
  allocate(parent_offset(max(1, N_det_generators)))
  allocate(parent_actual(max(1, N_det_generators)))
  allocate(parent_cap64(max(1, N_det_generators)))

  act_mask = .false.
  if (n_act_orb > 0) then
    n_space_orb = n_act_orb
    do i = 1, n_act_orb
      if (list_act(i) > 0 .and. list_act(i) <= mo_num) then
        act_mask(list_act(i)) = .true.
      end if
    end do
  else
    n_space_orb = mo_num
    act_mask = .true.
  end if

  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'
  threshold_mode = (compression_policy == 'threshold')
  fixed_budget_mode = (compression_policy == 'fixed_budget')
  sparse_mode = threshold_mode .or. fixed_budget_mode
  raw_cap64 = 0_8
  raw_cap_unclipped64 = 0_8
  remaining64 = threshold_candidate_limit_det
  threshold_candidate_clipped = .false.
  do ipar = 1, N_det_generators
    call bitstring_to_list(psi_det_generators(1,1,ipar), occ_a, n_occ_a, N_int)
    call bitstring_to_list(psi_det_generators(1,2,ipar), occ_b, n_occ_b, N_int)
    n_occ_a_act = 0
    do i = 1, n_occ_a
      if (occ_a(i) > 0 .and. occ_a(i) <= mo_num) then
        if (act_mask(occ_a(i))) then
          n_occ_a_act = n_occ_a_act + 1
          occ_a_act(n_occ_a_act) = occ_a(i)
        end if
      end if
    end do
    n_occ_b_act = 0
    do i = 1, n_occ_b
      if (occ_b(i) > 0 .and. occ_b(i) <= mo_num) then
        if (act_mask(occ_b(i))) then
          n_occ_b_act = n_occ_b_act + 1
          occ_b_act(n_occ_b_act) = occ_b(i)
        end if
      end if
    end do
    n_vir_a = max(0, n_space_orb - n_occ_a_act)
    n_vir_b = max(0, n_space_orb - n_occ_b_act)
    parent_raw_cap64 = 0_8
    parent_raw_cap64 = parent_raw_cap64 + int(n_occ_a_act,8) * int(n_vir_a,8)
    parent_raw_cap64 = parent_raw_cap64 + int(n_occ_b_act,8) * int(n_vir_b,8)
    parent_raw_cap64 = parent_raw_cap64 + (int(n_occ_a_act,8) * int(max(0, n_occ_a_act-1),8) / 2_8) * &
         (int(n_vir_a,8) * int(max(0, n_vir_a-1),8) / 2_8)
    parent_raw_cap64 = parent_raw_cap64 + (int(n_occ_b_act,8) * int(max(0, n_occ_b_act-1),8) / 2_8) * &
         (int(n_vir_b,8) * int(max(0, n_vir_b-1),8) / 2_8)
    parent_raw_cap64 = parent_raw_cap64 + int(n_occ_a_act,8) * int(n_vir_a,8) * int(n_occ_b_act,8) * int(n_vir_b,8)
    raw_cap_unclipped64 = raw_cap_unclipped64 + parent_raw_cap64
    if (threshold_mode) then
      if (remaining64 <= 0_8) then
        parent_cap64(ipar) = 0_8
        threshold_candidate_clipped = .true.
      else
        parent_cap64(ipar) = min(parent_raw_cap64, remaining64)
        if (parent_cap64(ipar) < parent_raw_cap64) threshold_candidate_clipped = .true.
        remaining64 = remaining64 - parent_cap64(ipar)
      end if
    else
      parent_cap64(ipar) = parent_raw_cap64
    end if
    raw_cap64 = raw_cap64 + parent_cap64(ipar)
  end do

  if (raw_cap64 <= 0_8) then
    return
  end if
  if (raw_cap64 > int(huge(1),8)) then
    print *, 'ccik(hq): connected_sd_fullq raw candidate buffer exceeds INTEGER range'
    stop 1
  end if
  raw_candidate_bytes = raw_cap64 * int((2 * N_int * bit_kind_size) / 8 + 8, 8)
  if (threshold_mode .and. threshold_candidate_clipped) then
    print *, 'ccik(hq): threshold candidate generation clipped to per-krylov limit'
    print *, 'ccik(hq): unclipped raw connected-SD count = ', raw_cap_unclipped64
    print *, 'ccik(hq): clipped raw connected-SD count = ', raw_cap64
  else if (threshold_mode .and. raw_candidate_bytes > threshold_candidate_limit_bytes) then
    print *, 'ccik(hq): threshold compression policy exceeded internal candidate limit'
    print *, 'ccik(hq): raw connected-SD count = ', raw_cap64
    print *, 'ccik(hq): estimated candidate buffer bytes = ', raw_candidate_bytes
    print *, 'ccik(hq): loosen thresholds or switch back to topk compression'
    stop 2
  end if

  raw_cap = 0
  do ipar = 1, N_det_generators
    if (parent_cap64(ipar) > int(huge(1),8)) then
      print *, 'ccik(hq): connected_sd_fullq parent candidate buffer exceeds INTEGER range'
      stop 1
    end if
    parent_cap(ipar) = int(parent_cap64(ipar))
    parent_offset(ipar) = raw_cap + 1
    parent_actual(ipar) = 0
    raw_cap = raw_cap + parent_cap(ipar)
  end do

  allocate(det_work(N_int,2,max(1, raw_cap)), coef_work(max(1, raw_cap)))
  nraw = 0

!$OMP PARALLEL DO DEFAULT(shared) SCHEDULE(dynamic) &
!$OMP& PRIVATE(ipar, occ_a, occ_b, virt_a, virt_b, occ_a_act, occ_b_act, occ_mask_a, occ_mask_b, det_trial, &
!$OMP& i, j, k, l, n_occ_a, n_occ_b, n_vir_a, n_vir_b, n_occ_a_act, n_occ_b_act, orb_i, i_ok1, i_ok2, base, local_count)
  do ipar = 1, N_det_generators
    base = parent_offset(ipar)
    local_count = 0
    if (parent_cap(ipar) <= 0) then
      parent_actual(ipar) = 0
      cycle
    end if
    occ_mask_a = .false.
    occ_mask_b = .false.
    call bitstring_to_list(psi_det_generators(1,1,ipar), occ_a, n_occ_a, N_int)
    call bitstring_to_list(psi_det_generators(1,2,ipar), occ_b, n_occ_b, N_int)

    n_occ_a_act = 0
    do i = 1, n_occ_a
      if (occ_a(i) > 0 .and. occ_a(i) <= mo_num) then
        occ_mask_a(occ_a(i)) = .true.
        if (act_mask(occ_a(i))) then
          n_occ_a_act = n_occ_a_act + 1
          occ_a_act(n_occ_a_act) = occ_a(i)
        end if
      end if
    end do
    n_occ_b_act = 0
    do i = 1, n_occ_b
      if (occ_b(i) > 0 .and. occ_b(i) <= mo_num) then
        occ_mask_b(occ_b(i)) = .true.
        if (act_mask(occ_b(i))) then
          n_occ_b_act = n_occ_b_act + 1
          occ_b_act(n_occ_b_act) = occ_b(i)
        end if
      end if
    end do

    n_vir_a = 0
    n_vir_b = 0
    do i = 1, n_space_orb
      if (n_act_orb > 0) then
        orb_i = list_act(i)
      else
        orb_i = i
      end if
      if (.not. occ_mask_a(orb_i)) then
        n_vir_a = n_vir_a + 1
        virt_a(n_vir_a) = orb_i
      end if
      if (.not. occ_mask_b(orb_i)) then
        n_vir_b = n_vir_b + 1
        virt_b(n_vir_b) = orb_i
      end if
    end do

single_a_loop: do i = 1, n_occ_a_act
      if (local_count >= parent_cap(ipar)) exit single_a_loop
      do j = 1, n_vir_a
        if (local_count >= parent_cap(ipar)) exit
        det_trial = psi_det_generators(:,:,ipar)
        call do_single_excitation(det_trial, occ_a_act(i), virt_a(j), 1, i_ok1)
        if (i_ok1 /= 1) cycle
        if (threshold_mode) then
          if (ccik_det_binary_search(det_trial, psi_det_propagators_sorted, N_det_propagators, N_int)) cycle
        else if (fixed_budget_mode) then
          if (is_in_wavefunction(det_trial, N_int)) cycle
        else
          if (is_in_wavefunction(det_trial, N_int)) cycle
        end if
        local_count = local_count + 1
        det_work(:,:,base + local_count - 1) = det_trial
      end do
    end do single_a_loop

single_b_loop: do i = 1, n_occ_b_act
      if (local_count >= parent_cap(ipar)) exit single_b_loop
      do j = 1, n_vir_b
        if (local_count >= parent_cap(ipar)) exit
        det_trial = psi_det_generators(:,:,ipar)
        call do_single_excitation(det_trial, occ_b_act(i), virt_b(j), 2, i_ok1)
        if (i_ok1 /= 1) cycle
        if (threshold_mode) then
          if (ccik_det_binary_search(det_trial, psi_det_propagators_sorted, N_det_propagators, N_int)) cycle
        else if (fixed_budget_mode) then
          if (is_in_wavefunction(det_trial, N_int)) cycle
        else
          if (is_in_wavefunction(det_trial, N_int)) cycle
        end if
        local_count = local_count + 1
        det_work(:,:,base + local_count - 1) = det_trial
      end do
    end do single_b_loop

double_a_loop: do i = 1, n_occ_a_act-1
      if (local_count >= parent_cap(ipar)) exit double_a_loop
      do j = i+1, n_occ_a_act
        if (local_count >= parent_cap(ipar)) exit
        do k = 1, n_vir_a-1
          if (local_count >= parent_cap(ipar)) exit
          do l = k+1, n_vir_a
            if (local_count >= parent_cap(ipar)) exit
            det_trial = psi_det_generators(:,:,ipar)
            call do_single_excitation(det_trial, occ_a_act(i), virt_a(k), 1, i_ok1)
            if (i_ok1 /= 1) cycle
            call do_single_excitation(det_trial, occ_a_act(j), virt_a(l), 1, i_ok2)
            if (i_ok2 /= 1) cycle
            if (threshold_mode) then
              if (ccik_det_binary_search(det_trial, psi_det_propagators_sorted, N_det_propagators, N_int)) cycle
            else if (fixed_budget_mode) then
              if (is_in_wavefunction(det_trial, N_int)) cycle
            else
              if (is_in_wavefunction(det_trial, N_int)) cycle
            end if
            local_count = local_count + 1
            det_work(:,:,base + local_count - 1) = det_trial
          end do
        end do
      end do
    end do double_a_loop

double_b_loop: do i = 1, n_occ_b_act-1
      if (local_count >= parent_cap(ipar)) exit double_b_loop
      do j = i+1, n_occ_b_act
        if (local_count >= parent_cap(ipar)) exit
        do k = 1, n_vir_b-1
          if (local_count >= parent_cap(ipar)) exit
          do l = k+1, n_vir_b
            if (local_count >= parent_cap(ipar)) exit
            det_trial = psi_det_generators(:,:,ipar)
            call do_single_excitation(det_trial, occ_b_act(i), virt_b(k), 2, i_ok1)
            if (i_ok1 /= 1) cycle
            call do_single_excitation(det_trial, occ_b_act(j), virt_b(l), 2, i_ok2)
            if (i_ok2 /= 1) cycle
            if (threshold_mode) then
              if (ccik_det_binary_search(det_trial, psi_det_propagators_sorted, N_det_propagators, N_int)) cycle
            else if (fixed_budget_mode) then
              if (is_in_wavefunction(det_trial, N_int)) cycle
            else
              if (is_in_wavefunction(det_trial, N_int)) cycle
            end if
            local_count = local_count + 1
            det_work(:,:,base + local_count - 1) = det_trial
          end do
        end do
      end do
    end do double_b_loop

cross_ab_loop: do i = 1, n_occ_a_act
      if (local_count >= parent_cap(ipar)) exit cross_ab_loop
      do j = 1, n_vir_a
        if (local_count >= parent_cap(ipar)) exit
        do k = 1, n_occ_b_act
          if (local_count >= parent_cap(ipar)) exit
          do l = 1, n_vir_b
            if (local_count >= parent_cap(ipar)) exit
            det_trial = psi_det_generators(:,:,ipar)
            call do_single_excitation(det_trial, occ_a_act(i), virt_a(j), 1, i_ok1)
            if (i_ok1 /= 1) cycle
            call do_single_excitation(det_trial, occ_b_act(k), virt_b(l), 2, i_ok2)
            if (i_ok2 /= 1) cycle
            if (threshold_mode) then
              if (ccik_det_binary_search(det_trial, psi_det_propagators_sorted, N_det_propagators, N_int)) cycle
            else if (fixed_budget_mode) then
              if (is_in_wavefunction(det_trial, N_int)) cycle
            else
              if (is_in_wavefunction(det_trial, N_int)) cycle
            end if
            local_count = local_count + 1
            det_work(:,:,base + local_count - 1) = det_trial
          end do
        end do
      end do
    end do cross_ab_loop

    parent_actual(ipar) = local_count
  end do
!$OMP END PARALLEL DO

  nraw = 0
  do ipar = 1, N_det_generators
    local_count = parent_actual(ipar)
    if (local_count <= 0) cycle
    base = parent_offset(ipar)
    if (nraw + 1 < base) then
      do copy_idx = 1, local_count
        det_work(:,:,nraw + copy_idx) = det_work(:,:,base + copy_idx - 1)
      end do
    end if
    nraw = nraw + local_count
  end do

  generated_before_dedup = nraw
  if (nraw > 0) then
    coef_work(1:nraw) = 1.d0
    call ccik_vec_assign(candidate_vec, det_work, coef_work, nraw, N_int)
    generated_unique = candidate_vec%n
  end if

  deallocate(det_work, coef_work)
  deallocate(parent_cap, parent_offset, parent_actual, parent_cap64)
end subroutine ccik_build_connected_sd_candidates


subroutine ccik_capture_working_vector(vec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(inout) :: vec
  integer(bit_kind), allocatable :: det_tmp(:,:,:)
  double precision, allocatable :: coef_tmp(:)
  integer :: i

  call ccik_vec_free(vec)
  if (N_det <= 0) return

  allocate(det_tmp(N_int,2,N_det), coef_tmp(N_det))
  do i = 1, N_det
    det_tmp(:,:,i) = psi_det(:,:,i)
    coef_tmp(i) = psi_coef(i,1)
  end do
  call ccik_vec_assign(vec, det_tmp, coef_tmp, N_det, N_int)
  deallocate(det_tmp, coef_tmp)
end subroutine ccik_capture_working_vector


subroutine ccik_select_top_pt2_residual(source_vec, rayleigh, nkeep, selected_vec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: source_vec
  double precision, intent(in) :: rayleigh
  integer, intent(in) :: nkeep
  type(ccik_vector), intent(inout) :: selected_vec
  integer :: i, n_keep, first_keep
  integer, allocatable :: idx(:)
  integer(bit_kind), allocatable :: det_keep(:,:,:)
  double precision, allocatable :: score(:), coef_keep(:)
  double precision :: hdiag_i, denom
  double precision, external :: diag_H_mat_elem

  call ccik_vec_free(selected_vec)
  if (source_vec%n <= 0 .or. nkeep <= 0) return
  if (source_vec%n <= nkeep) then
    call ccik_vec_copy(selected_vec, source_vec)
    return
  end if

  allocate(score(source_vec%n), idx(source_vec%n))
  do i = 1, source_vec%n
    hdiag_i = diag_H_mat_elem(source_vec%det(1,1,i), N_int)
    denom = max(1.d-8, dabs(rayleigh - hdiag_i))
    score(i) = dabs(source_vec%coef(i)) * dabs(source_vec%coef(i)) / denom
    idx(i) = i
  end do
  if (source_vec%n > 1) call dsort(score, idx, source_vec%n)

  n_keep = min(max(1, nkeep), source_vec%n)
  first_keep = source_vec%n - n_keep + 1
  allocate(det_keep(N_int,2,n_keep), coef_keep(n_keep))
  do i = 1, n_keep
    det_keep(:,:,i) = source_vec%det(:,:,idx(first_keep + i - 1))
    coef_keep(i) = source_vec%coef(idx(first_keep + i - 1))
  end do
  call ccik_vec_assign(selected_vec, det_keep, coef_keep, n_keep, N_int)
  deallocate(score, idx, det_keep, coef_keep)
end subroutine ccik_select_top_pt2_residual


subroutine ccik_build_connected_sd_candidates_from_parent_vec(parent_vec, candidate_vec, generated_before_dedup, generated_unique)
  use ccik_sparse
  use bitmasks
  implicit none
  type(ccik_vector), intent(in) :: parent_vec
  type(ccik_vector), intent(inout) :: candidate_vec
  integer, intent(out) :: generated_before_dedup, generated_unique
  integer :: occ_a(max(1, mo_num)), occ_b(max(1, mo_num)), virt_a(max(1, mo_num)), virt_b(max(1, mo_num))
  integer :: occ_a_act(max(1, mo_num)), occ_b_act(max(1, mo_num))
  logical :: occ_mask_a(max(1, mo_num)), occ_mask_b(max(1, mo_num)), act_mask(max(1, mo_num))
  integer(bit_kind), allocatable :: det_work(:,:,:)
  double precision, allocatable :: coef_work(:)
  integer(bit_kind) :: det_trial(N_int,2)
  integer, allocatable :: parent_cap(:), parent_offset(:), parent_actual(:)
  integer*8, allocatable :: parent_cap64(:)
  integer :: ipar, i, j, k, l, base, local_count, copy_idx
  integer :: n_occ_a, n_occ_b, n_vir_a, n_vir_b
  integer :: n_occ_a_act, n_occ_b_act
  integer :: n_space_orb, orb_i
  integer :: i_ok1, i_ok2, nraw, raw_cap
  integer*8 :: raw_cap64, parent_raw_cap64
  logical, external :: is_in_wavefunction

  call ccik_vec_free(candidate_vec)
  generated_before_dedup = 0
  generated_unique = 0
  if (parent_vec%n <= 0 .or. mo_num <= 0) return

  PROVIDE n_act_orb list_act psi_det_working_sorted
  allocate(parent_cap(max(1, parent_vec%n)))
  allocate(parent_offset(max(1, parent_vec%n)))
  allocate(parent_actual(max(1, parent_vec%n)))
  allocate(parent_cap64(max(1, parent_vec%n)))

  act_mask = .false.
  if (n_act_orb > 0) then
    n_space_orb = n_act_orb
    do i = 1, n_act_orb
      if (list_act(i) > 0 .and. list_act(i) <= mo_num) then
        act_mask(list_act(i)) = .true.
      end if
    end do
  else
    n_space_orb = mo_num
    act_mask = .true.
  end if

  raw_cap64 = 0_8
  do ipar = 1, parent_vec%n
    call bitstring_to_list(parent_vec%det(1,1,ipar), occ_a, n_occ_a, N_int)
    call bitstring_to_list(parent_vec%det(1,2,ipar), occ_b, n_occ_b, N_int)
    n_occ_a_act = 0
    do i = 1, n_occ_a
      if (occ_a(i) > 0 .and. occ_a(i) <= mo_num) then
        if (act_mask(occ_a(i))) then
          n_occ_a_act = n_occ_a_act + 1
          occ_a_act(n_occ_a_act) = occ_a(i)
        end if
      end if
    end do
    n_occ_b_act = 0
    do i = 1, n_occ_b
      if (occ_b(i) > 0 .and. occ_b(i) <= mo_num) then
        if (act_mask(occ_b(i))) then
          n_occ_b_act = n_occ_b_act + 1
          occ_b_act(n_occ_b_act) = occ_b(i)
        end if
      end if
    end do
    n_vir_a = max(0, n_space_orb - n_occ_a_act)
    n_vir_b = max(0, n_space_orb - n_occ_b_act)
    parent_raw_cap64 = 0_8
    parent_raw_cap64 = parent_raw_cap64 + int(n_occ_a_act,8) * int(n_vir_a,8)
    parent_raw_cap64 = parent_raw_cap64 + int(n_occ_b_act,8) * int(n_vir_b,8)
    parent_raw_cap64 = parent_raw_cap64 + (int(n_occ_a_act,8) * int(max(0, n_occ_a_act-1),8) / 2_8) * &
         (int(n_vir_a,8) * int(max(0, n_vir_a-1),8) / 2_8)
    parent_raw_cap64 = parent_raw_cap64 + (int(n_occ_b_act,8) * int(max(0, n_occ_b_act-1),8) / 2_8) * &
         (int(n_vir_b,8) * int(max(0, n_vir_b-1),8) / 2_8)
    parent_raw_cap64 = parent_raw_cap64 + int(n_occ_a_act,8) * int(n_vir_a,8) * int(n_occ_b_act,8) * int(n_vir_b,8)
    parent_cap64(ipar) = parent_raw_cap64
    raw_cap64 = raw_cap64 + parent_cap64(ipar)
  end do

  if (raw_cap64 <= 0_8) then
    deallocate(parent_cap, parent_offset, parent_actual, parent_cap64)
    return
  end if
  if (raw_cap64 > int(huge(1),8)) then
    print *, 'ccik(hq): connected_sd_fullq raw candidate buffer exceeds INTEGER range'
    stop 1
  end if

  raw_cap = 0
  do ipar = 1, parent_vec%n
    parent_cap(ipar) = int(parent_cap64(ipar))
    parent_offset(ipar) = raw_cap + 1
    parent_actual(ipar) = 0
    raw_cap = raw_cap + parent_cap(ipar)
  end do

  allocate(det_work(N_int,2,max(1, raw_cap)), coef_work(max(1, raw_cap)))
  nraw = 0

!$OMP PARALLEL DO DEFAULT(shared) SCHEDULE(dynamic) &
!$OMP& PRIVATE(ipar, occ_a, occ_b, virt_a, virt_b, occ_a_act, occ_b_act, occ_mask_a, occ_mask_b, det_trial, &
!$OMP& i, j, k, l, n_occ_a, n_occ_b, n_vir_a, n_vir_b, n_occ_a_act, n_occ_b_act, orb_i, i_ok1, i_ok2, base, local_count)
  do ipar = 1, parent_vec%n
    base = parent_offset(ipar)
    local_count = 0
    if (parent_cap(ipar) <= 0) then
      parent_actual(ipar) = 0
      cycle
    end if
    occ_mask_a = .false.
    occ_mask_b = .false.
    call bitstring_to_list(parent_vec%det(1,1,ipar), occ_a, n_occ_a, N_int)
    call bitstring_to_list(parent_vec%det(1,2,ipar), occ_b, n_occ_b, N_int)

    n_occ_a_act = 0
    do i = 1, n_occ_a
      if (occ_a(i) > 0 .and. occ_a(i) <= mo_num) then
        occ_mask_a(occ_a(i)) = .true.
        if (act_mask(occ_a(i))) then
          n_occ_a_act = n_occ_a_act + 1
          occ_a_act(n_occ_a_act) = occ_a(i)
        end if
      end if
    end do
    n_occ_b_act = 0
    do i = 1, n_occ_b
      if (occ_b(i) > 0 .and. occ_b(i) <= mo_num) then
        occ_mask_b(occ_b(i)) = .true.
        if (act_mask(occ_b(i))) then
          n_occ_b_act = n_occ_b_act + 1
          occ_b_act(n_occ_b_act) = occ_b(i)
        end if
      end if
    end do

    n_vir_a = 0
    n_vir_b = 0
    do i = 1, n_space_orb
      if (n_act_orb > 0) then
        orb_i = list_act(i)
      else
        orb_i = i
      end if
      if (.not. occ_mask_a(orb_i)) then
        n_vir_a = n_vir_a + 1
        virt_a(n_vir_a) = orb_i
      end if
      if (.not. occ_mask_b(orb_i)) then
        n_vir_b = n_vir_b + 1
        virt_b(n_vir_b) = orb_i
      end if
    end do

single_a_loop: do i = 1, n_occ_a_act
      if (local_count >= parent_cap(ipar)) exit single_a_loop
      do j = 1, n_vir_a
        if (local_count >= parent_cap(ipar)) exit
        det_trial = parent_vec%det(:,:,ipar)
        call do_single_excitation(det_trial, occ_a_act(i), virt_a(j), 1, i_ok1)
        if (i_ok1 /= 1) cycle
        if (is_in_wavefunction(det_trial, N_int)) cycle
        local_count = local_count + 1
        det_work(:,:,base + local_count - 1) = det_trial
      end do
    end do single_a_loop

single_b_loop: do i = 1, n_occ_b_act
      if (local_count >= parent_cap(ipar)) exit single_b_loop
      do j = 1, n_vir_b
        if (local_count >= parent_cap(ipar)) exit
        det_trial = parent_vec%det(:,:,ipar)
        call do_single_excitation(det_trial, occ_b_act(i), virt_b(j), 2, i_ok1)
        if (i_ok1 /= 1) cycle
        if (is_in_wavefunction(det_trial, N_int)) cycle
        local_count = local_count + 1
        det_work(:,:,base + local_count - 1) = det_trial
      end do
    end do single_b_loop

double_a_loop: do i = 1, n_occ_a_act-1
      if (local_count >= parent_cap(ipar)) exit double_a_loop
      do j = i+1, n_occ_a_act
        if (local_count >= parent_cap(ipar)) exit
        do k = 1, n_vir_a-1
          if (local_count >= parent_cap(ipar)) exit
          do l = k+1, n_vir_a
            if (local_count >= parent_cap(ipar)) exit
            det_trial = parent_vec%det(:,:,ipar)
            call do_single_excitation(det_trial, occ_a_act(i), virt_a(k), 1, i_ok1)
            if (i_ok1 /= 1) cycle
            call do_single_excitation(det_trial, occ_a_act(j), virt_a(l), 1, i_ok2)
            if (i_ok2 /= 1) cycle
            if (is_in_wavefunction(det_trial, N_int)) cycle
            local_count = local_count + 1
            det_work(:,:,base + local_count - 1) = det_trial
          end do
        end do
      end do
    end do double_a_loop

double_b_loop: do i = 1, n_occ_b_act-1
      if (local_count >= parent_cap(ipar)) exit double_b_loop
      do j = i+1, n_occ_b_act
        if (local_count >= parent_cap(ipar)) exit
        do k = 1, n_vir_b-1
          if (local_count >= parent_cap(ipar)) exit
          do l = k+1, n_vir_b
            if (local_count >= parent_cap(ipar)) exit
            det_trial = parent_vec%det(:,:,ipar)
            call do_single_excitation(det_trial, occ_b_act(i), virt_b(k), 2, i_ok1)
            if (i_ok1 /= 1) cycle
            call do_single_excitation(det_trial, occ_b_act(j), virt_b(l), 2, i_ok2)
            if (i_ok2 /= 1) cycle
            if (is_in_wavefunction(det_trial, N_int)) cycle
            local_count = local_count + 1
            det_work(:,:,base + local_count - 1) = det_trial
          end do
        end do
      end do
    end do double_b_loop

cross_ab_loop: do i = 1, n_occ_a_act
      if (local_count >= parent_cap(ipar)) exit cross_ab_loop
      do j = 1, n_vir_a
        if (local_count >= parent_cap(ipar)) exit
        do k = 1, n_occ_b_act
          if (local_count >= parent_cap(ipar)) exit
          do l = 1, n_vir_b
            if (local_count >= parent_cap(ipar)) exit
            det_trial = parent_vec%det(:,:,ipar)
            call do_single_excitation(det_trial, occ_a_act(i), virt_a(j), 1, i_ok1)
            if (i_ok1 /= 1) cycle
            call do_single_excitation(det_trial, occ_b_act(k), virt_b(l), 2, i_ok2)
            if (i_ok2 /= 1) cycle
            if (is_in_wavefunction(det_trial, N_int)) cycle
            local_count = local_count + 1
            det_work(:,:,base + local_count - 1) = det_trial
          end do
        end do
      end do
    end do cross_ab_loop

    parent_actual(ipar) = local_count
  end do
!$OMP END PARALLEL DO

  nraw = 0
  do ipar = 1, parent_vec%n
    local_count = parent_actual(ipar)
    if (local_count <= 0) cycle
    base = parent_offset(ipar)
    if (nraw + 1 < base) then
      do copy_idx = 1, local_count
        det_work(:,:,nraw + copy_idx) = det_work(:,:,base + copy_idx - 1)
      end do
    end if
    nraw = nraw + local_count
  end do

  generated_before_dedup = nraw
  if (nraw > 0) then
    coef_work(1:nraw) = 1.d0
    call ccik_vec_assign(candidate_vec, det_work, coef_work, nraw, N_int)
    generated_unique = candidate_vec%n
  end if

  deallocate(det_work, coef_work)
  deallocate(parent_cap, parent_offset, parent_actual, parent_cap64)
end subroutine ccik_build_connected_sd_candidates_from_parent_vec


subroutine ccik_build_projected_hq_from_parent_vec(parent_vec, vec, rayleigh, generated_before_dedup, generated_unique)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: parent_vec
  type(ccik_vector), intent(inout) :: vec
  double precision, intent(out) :: rayleigh
  integer, intent(out) :: generated_before_dedup, generated_unique
  type(ccik_vector) :: candidate_vec
  integer(bit_kind), allocatable :: det_tmp(:,:,:), mini_det_thread(:,:,:)
  double precision, allocatable :: coef_tmp(:), hpsi_internal(:), ext_coef(:)
  integer, allocatable :: idx_minilist_thread(:)
  double precision :: hpsi_local(1)
  integer :: i, n_internal, n_candidates, n_minilist, n_total

  call ccik_vec_free(vec)
  call ccik_vec_free(candidate_vec)
  rayleigh = 0.d0
  generated_before_dedup = 0
  generated_unique = 0
  n_internal = N_det
  if (n_internal <= 0) return

  allocate(hpsi_internal(max(1, n_internal)))
!$OMP PARALLEL DO DEFAULT(shared) PRIVATE(i, hpsi_local) REDUCTION(+:rayleigh) SCHEDULE(dynamic)
  do i = 1, n_internal
    call i_H_psi(psi_det(1,1,i), psi_det, psi_coef, N_int, N_det, psi_det_size, 1, hpsi_local)
    hpsi_internal(i) = hpsi_local(1)
    rayleigh = rayleigh + psi_coef(i,1) * hpsi_local(1)
  end do
!$OMP END PARALLEL DO

  call ccik_build_connected_sd_candidates_from_parent_vec(parent_vec, candidate_vec, generated_before_dedup, generated_unique)
  n_candidates = candidate_vec%n
  if (n_candidates <= 0) then
    allocate(det_tmp(N_int,2,max(1, n_internal)), coef_tmp(max(1, n_internal)))
    do i = 1, n_internal
      det_tmp(:,:,i) = psi_det(:,:,i)
      coef_tmp(i) = hpsi_internal(i)
    end do
    call ccik_vec_assign(vec, det_tmp, coef_tmp, n_internal, N_int)
    deallocate(det_tmp, coef_tmp)
    deallocate(hpsi_internal)
    call ccik_vec_free(candidate_vec)
    return
  end if

  allocate(ext_coef(max(1, n_candidates)))
!$OMP PARALLEL DEFAULT(shared) PRIVATE(i, n_minilist, hpsi_local, mini_det_thread, idx_minilist_thread)
  allocate(mini_det_thread(N_int,2,max(1, n_internal)), idx_minilist_thread(max(1, n_internal)))
!$OMP DO SCHEDULE(dynamic)
  do i = 1, n_candidates
    call create_minilist(candidate_vec%det(1,1,i), psi_det, mini_det_thread, idx_minilist_thread, N_det, n_minilist, N_int)
    if (n_minilist > 0) then
      call i_H_psi_minilist(candidate_vec%det(1,1,i), mini_det_thread, idx_minilist_thread, n_minilist, psi_coef, &
           N_int, n_minilist, psi_det_size, 1, hpsi_local)
    else
      hpsi_local(1) = 0.d0
    end if
    ext_coef(i) = hpsi_local(1)
  end do
!$OMP END DO
  deallocate(mini_det_thread, idx_minilist_thread)
!$OMP END PARALLEL

  n_total = n_internal + n_candidates
  allocate(det_tmp(N_int,2,n_total), coef_tmp(n_total))
  do i = 1, n_internal
    det_tmp(:,:,i) = psi_det(:,:,i)
    coef_tmp(i) = hpsi_internal(i)
  end do
  do i = 1, n_candidates
    det_tmp(:,:,n_internal+i) = candidate_vec%det(:,:,i)
    coef_tmp(n_internal+i) = ext_coef(i)
  end do
  call ccik_vec_assign(vec, det_tmp, coef_tmp, n_total, N_int)

  deallocate(det_tmp, coef_tmp, ext_coef, hpsi_internal)
  call ccik_vec_free(candidate_vec)
end subroutine ccik_build_projected_hq_from_parent_vec


subroutine ccik_build_projected_hq_connected_sd_fullq(vec, rayleigh, projected_call_count, matvec_backend)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(inout) :: vec
  type(ccik_vector) :: current_vec, parent_vec, support_vec
  type(ccik_vector) :: internal_trace_vec, external_trace_vec
  double precision, intent(out) :: rayleigh
  integer, intent(in) :: projected_call_count
  character(len=*), intent(in) :: matvec_backend
  type(ccik_vector) :: candidate_vec, full_vec
  integer(bit_kind), allocatable :: det_tmp(:,:,:), det_full(:,:,:), mini_det_thread(:,:,:)
  double precision, allocatable :: coef_tmp(:), hpsi_internal(:), ext_coef(:), score_sort(:), pt2_score(:), coef_full(:)
  integer, allocatable :: ext_order(:), idx_minilist_thread(:)
  double precision :: hpsi(1)
  double precision :: hpsi_local(1)
  double precision :: hdiag_i, score_denom, max_proj_score, max_pt2_score, pt2_val, abscoef
  double precision :: t0, t1, t_select, t_sort, t_internal, t_external, t_assign, source_dropped_norm
  double precision :: projected_full_norm, projected_final_norm
  integer :: i, j, n_ext, n_total, n_total_full, n_candidates, n_minilist, tmp_keep_est, n_internal
  integer :: generated_before_dedup, generated_unique
  integer :: ccik_tmp_keep_value
  character(len=32) :: selector_mode, compression_policy
  character(len=512) :: trace_line
  logical :: threshold_mode, fixed_budget_mode, sparse_mode, have_delayed_support
  double precision, external :: diag_H_mat_elem

  call ccik_vec_free(vec)
  call ccik_vec_free(candidate_vec)
  call ccik_vec_free(full_vec)
  call ccik_vec_free(internal_trace_vec)
  call ccik_vec_free(external_trace_vec)
  rayleigh = 0.d0
  t_select = 0.d0
  t_sort = 0.d0
  t_internal = 0.d0
  t_external = 0.d0
  t_assign = 0.d0
  generated_before_dedup = 0
  generated_unique = 0
  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'
  threshold_mode = (compression_policy == 'threshold')
  fixed_budget_mode = (compression_policy == 'fixed_budget')
  sparse_mode = threshold_mode .or. fixed_budget_mode
  source_dropped_norm = 0.d0
  if (sparse_mode) source_dropped_norm = ccik_propagator_dropped_norm
  n_internal = N_det
  if (threshold_mode) n_internal = N_det_propagators

  if (ccik_verbose) then
    write(*,'(A,I6,1X,A,1X,A,4(A,I10),A,F24.12)') 'ccik(hq) begin call=', projected_call_count, &
         'backend=', trim(matvec_backend), '  |q|=', N_det, '  fullq=', n_internal, '  ngen=', N_det_generators, &
         '  ext_keep=', ccik_external_keep, '  rayleigh=', rayleigh
    call flush(6)
  end if

  call ccik_check_connected_sd_fullq_provider_state(projected_call_count, fixed_budget_mode)

  allocate(hpsi_internal(max(1, n_internal)))
  call cpu_time(t0)
!$OMP PARALLEL DO DEFAULT(shared) PRIVATE(i, hpsi_local) REDUCTION(+:rayleigh) SCHEDULE(dynamic)
  do i = 1, n_internal
    if (threshold_mode) then
      call i_H_psi(psi_det_propagators(1,1,i), psi_det_propagators, psi_coef_propagators, N_int, N_det_propagators, psi_det_size, 1, hpsi_local)
      hpsi_internal(i) = hpsi_local(1)
      rayleigh = rayleigh + psi_coef_propagators(i,1) * hpsi_local(1)
    else
      call i_H_psi(psi_det(1,1,i), psi_det, psi_coef, N_int, N_det, psi_det_size, 1, hpsi_local)
      hpsi_internal(i) = hpsi_local(1)
      rayleigh = rayleigh + psi_coef(i,1) * hpsi_local(1)
    end if
  end do
!$OMP END PARALLEL DO
  call cpu_time(t1)
  t_internal = t1 - t0

  selector_mode = trim(adjustl(ccik_selector_mode))
  if (len_trim(selector_mode) == 0) selector_mode = 'abs_residual'
  if (selector_mode == 'pt2_legacy') selector_mode = 'pt2_residual'

  if (selector_mode == 'pt2_residual' .and. .not. fixed_budget_mode) then
    print *, "ccik error: selector_mode='pt2_residual' requires compression_policy='fixed_budget'"
    stop 2
  end if

  if (selector_mode == 'pt2_residual' .and. fixed_budget_mode) then
    call ccik_capture_working_vector(current_vec)
    call ccik_runtime_get_propagation_override(support_vec, have_delayed_support)
    if (have_delayed_support) then
      call ccik_vec_project_support(parent_vec, current_vec, support_vec)
      if (parent_vec%n <= 0) have_delayed_support = .false.
    end if
    if (.not. have_delayed_support) then
      call ccik_vec_copy(parent_vec, current_vec)
      if (parent_vec%n > max(1, ccik_fixed_propagate_keep)) then
        call ccik_vec_compress(parent_vec, max(1, ccik_fixed_propagate_keep))
      end if
    end if
    source_dropped_norm = dsqrt(max(0.d0, ccik_vec_norm2(current_vec) - ccik_vec_norm2(parent_vec)))
    call cpu_time(t0)
    call ccik_build_projected_hq_from_parent_vec(parent_vec, full_vec, rayleigh, generated_before_dedup, generated_unique)
    call cpu_time(t1)
    t_internal = t1 - t0
    call ccik_vec_copy(vec, full_vec)
    if (vec%n > max(1, ccik_fixed_store_keep)) then
      call cpu_time(t0)
      call ccik_vec_compress(vec, max(1, ccik_fixed_store_keep))
      call cpu_time(t1)
      t_assign = t1 - t0
    end if
    if (ccik_verbose) then
      write(*,'(A,I6,1X,A,1X,A,4(A,I10),A,A,2(A,F12.4))') 'ccik(hq) done  call=', projected_call_count, &
           'backend=', trim(matvec_backend), '  raw=', generated_before_dedup, '  uniq=', generated_unique, &
           '  n_ext=', max(0, full_vec%n - N_det), '  |vec|=', vec%n, '  selector=', trim(selector_mode), &
           '  int_s=', t_internal, '  assign_s=', t_assign
      call flush(6)
    end if
    if (ccik_debug_trace) then
      projected_full_norm = dsqrt(max(0.d0, ccik_vec_norm2(full_vec)))
      projected_final_norm = dsqrt(max(0.d0, ccik_vec_norm2(vec)))
      write(trace_line,'(A,1X,I8,1X,A,1X,A,1X,A,1X,I8,1X,I8,1X,I8,1X,I8,1X,ES24.16,1X,L1,1X,ES24.16,1X,ES24.16,1X,ES24.16)') &
           'CCIK_TRACE_HQ', projected_call_count, trim(selector_mode), trim(matvec_backend), trim(compression_policy), N_det, parent_vec%n, &
           generated_before_dedup, generated_unique, rayleigh, .not. have_delayed_support, source_dropped_norm, projected_full_norm, projected_final_norm
      call ccik_debug_trace_line(trace_line)
      call ccik_debug_trace_vector('projected_hq_full', full_vec)
      call ccik_debug_trace_vector('projected_hq', vec)
    end if
    call ccik_vec_free(support_vec)
    call ccik_vec_free(parent_vec)
    call ccik_vec_free(current_vec)
    call ccik_vec_free(full_vec)
    call ccik_vec_free(candidate_vec)
    return
  end if

  if (ccik_verbose) then
    write(*,'(A,I6,A,F24.12,A,F12.4)') 'ccik(hq) internal pass call=', projected_call_count, &
         '  rayleigh=', rayleigh, '  cpu_s=', t_internal
    call flush(6)
  end if

  if (ccik_debug_trace .and. fixed_budget_mode) then
    call ccik_capture_provider_vector(support_vec, psi_det_propagators, psi_coef_propagators, N_det_propagators)
    call ccik_debug_trace_vector('propagation_source', support_vec)
    call ccik_capture_provider_vector(parent_vec, psi_det_generators, psi_coef_generators, N_det_generators)
    call ccik_debug_trace_vector('parent_source', parent_vec)
  end if

  tmp_keep_est = ccik_tmp_keep_value(max(1, ccik_nkeep))

  call cpu_time(t0)
  call ccik_build_connected_sd_candidates(candidate_vec, generated_before_dedup, generated_unique)
  call cpu_time(t1)
  t_select = t1 - t0

  if (compression_policy == 'topk') then
    if (n_internal > 1 .and. N_det_generators <= 1 .and. ccik_generator_keep > 1 &
        .and. ccik_generator_fraction <= 0.d0 .and. ccik_generator_abs_coeff_cutoff <= 0.d0 &
        .and. (ccik_max_parents_for_expansion <= 0 .or. ccik_max_parents_for_expansion > 1)) then
      write(*,'(A)') 'ccik error: stale generator/provider state detected in connected_sd_fullq'
      write(*,'(A,I10,A,I10,A,I10)') 'ccik error: |q|=', N_det, '  fullq=', n_internal, '  ngen=', N_det_generators
      write(*,'(A,I10,A,F12.6,A,ES12.4,A,I10)') 'ccik error: generator_keep=', ccik_generator_keep, &
           '  generator_fraction=', ccik_generator_fraction, '  abs_coeff_cutoff=', ccik_generator_abs_coeff_cutoff, &
           '  max_parents=', ccik_max_parents_for_expansion
      call flush(6)
      stop 1
    end if
  end if

  n_candidates = candidate_vec%n
  if (n_candidates <= 0) then
    allocate(det_tmp(N_int,2,max(1, n_internal)), coef_tmp(max(1, n_internal)))
    do i = 1, n_internal
      if (threshold_mode) then
        det_tmp(:,:,i) = psi_det_propagators(:,:,i)
      else
        det_tmp(:,:,i) = psi_det(:,:,i)
      end if
      coef_tmp(i) = hpsi_internal(i)
    end do
    call ccik_vec_assign(vec, det_tmp, coef_tmp, n_internal, N_int)
    if (threshold_mode .and. vec%n > 1000000) then
      call ccik_vec_compress(vec, 1000000)
    end if
    if (ccik_debug_trace) then
      projected_final_norm = dsqrt(max(0.d0, ccik_vec_norm2(vec)))
      write(trace_line,'(A,1X,I8,1X,A,1X,A,1X,A,1X,I8,1X,I8,1X,I8,1X,I8,1X,ES24.16,1X,L1,1X,ES24.16,1X,ES24.16,1X,ES24.16)') &
           'CCIK_TRACE_HQ', projected_call_count, trim(selector_mode), trim(matvec_backend), trim(compression_policy), n_internal, N_det_generators, &
           generated_before_dedup, 0, rayleigh, ccik_propagator_used_fallback, source_dropped_norm, 0.d0, projected_final_norm
      call ccik_debug_trace_line(trace_line)
      call ccik_debug_trace_vector('projected_hq', vec)
    end if
    deallocate(det_tmp, coef_tmp)
    deallocate(hpsi_internal)
    call ccik_vec_free(candidate_vec)
    return
  end if

  if (fixed_budget_mode) then
    n_ext = n_candidates
  else
    n_ext = min(max(ccik_external_keep, tmp_keep_est), n_candidates)
  end if
  n_total = n_internal + n_ext
  allocate(det_tmp(N_int,2,n_total), coef_tmp(n_total))
  allocate(ext_coef(max(1, n_candidates)), pt2_score(max(1, n_candidates)))
  allocate(score_sort(max(1, n_candidates)), ext_order(max(1, n_candidates)))

  do i = 1, n_internal
      if (threshold_mode) then
        det_tmp(:,:,i) = psi_det_propagators(:,:,i)
      else
        det_tmp(:,:,i) = psi_det(:,:,i)
    end if
    coef_tmp(i) = hpsi_internal(i)
  end do

  call cpu_time(t0)
!$OMP PARALLEL DEFAULT(shared) PRIVATE(i, n_minilist, abscoef, hdiag_i, score_denom, pt2_val, hpsi_local, mini_det_thread, idx_minilist_thread)
  allocate(mini_det_thread(N_int,2,max(1, n_internal)), idx_minilist_thread(max(1, n_internal)))
!$OMP DO SCHEDULE(dynamic)
  do i = 1, n_candidates
    if (threshold_mode) then
      call create_minilist(candidate_vec%det(1,1,i), psi_det_propagators, mini_det_thread, idx_minilist_thread, N_det_propagators, n_minilist, N_int)
    else
      call create_minilist(candidate_vec%det(1,1,i), psi_det, mini_det_thread, idx_minilist_thread, N_det, n_minilist, N_int)
    end if
    if (n_minilist > 0) then
      if (threshold_mode) then
        call i_H_psi_minilist(candidate_vec%det(1,1,i), mini_det_thread, idx_minilist_thread, n_minilist, psi_coef_propagators, &
             N_int, n_minilist, psi_det_size, 1, hpsi_local)
      else
        call i_H_psi_minilist(candidate_vec%det(1,1,i), mini_det_thread, idx_minilist_thread, n_minilist, psi_coef, &
             N_int, n_minilist, psi_det_size, 1, hpsi_local)
      end if
    else
      hpsi_local(1) = 0.d0
    end if
    ext_coef(i) = hpsi_local(1)
    abscoef = dabs(hpsi_local(1))
    if (selector_mode == 'pt2_legacy') then
      hdiag_i = diag_H_mat_elem(candidate_vec%det(1,1,i), N_int)
      score_denom = max(1.d-8, dabs(rayleigh - hdiag_i))
      score_sort(i) = abscoef*abscoef / score_denom
      pt2_val = 0.d0
      pt2_score(i) = pt2_val
    else if (selector_mode == 'sq_hq' .or. selector_mode == 'sq_residual') then
      score_sort(i) = abscoef*abscoef
      pt2_score(i) = 0.d0
    else
      score_sort(i) = abscoef
      pt2_score(i) = 0.d0
    end if
    ext_order(i) = i
  end do
!$OMP END DO
  deallocate(mini_det_thread, idx_minilist_thread)
!$OMP END PARALLEL
  call cpu_time(t1)
  t_external = t1 - t0

  if (.not. fixed_budget_mode .and. selector_mode == 'pt2_legacy') then
    max_proj_score = max(1.d-30, maxval(score_sort(1:n_candidates)))
    max_pt2_score = max(1.d-30, maxval(pt2_score(1:n_candidates)))
    do i = 1, n_candidates
      score_sort(i) = score_sort(i) / max_proj_score
      if (ccik_pt2_rank_weight > 0.d0) then
        score_sort(i) = score_sort(i) + ccik_pt2_rank_weight * pt2_score(i) / max_pt2_score
      end if
    end do
  end if

  if (.not. fixed_budget_mode) then
    call cpu_time(t0)
    if (n_candidates > 1) then
      call dsort(score_sort, ext_order, n_candidates)
    end if
    call cpu_time(t1)
    t_sort = t1 - t0
  else
    t_sort = 0.d0
  end if

  if (fixed_budget_mode .or. ccik_debug_trace) then
    n_total_full = n_internal + n_candidates
    allocate(det_full(N_int,2,n_total_full), coef_full(n_total_full))
    do i = 1, n_internal
      if (threshold_mode) then
        det_full(:,:,i) = psi_det_propagators(:,:,i)
      else
        det_full(:,:,i) = psi_det(:,:,i)
      end if
      coef_full(i) = hpsi_internal(i)
    end do
    do i = 1, n_candidates
      det_full(:,:,n_internal+i) = candidate_vec%det(:,:,i)
      coef_full(n_internal+i) = ext_coef(i)
    end do
    call ccik_vec_assign(full_vec, det_full, coef_full, n_total_full, N_int)
    if (ccik_debug_trace) then
      call ccik_debug_trace_vector('projected_hq_full', full_vec)
    end if
    deallocate(det_full, coef_full)
  end if

  if (fixed_budget_mode) then
    call cpu_time(t0)
    ! Match the Python fixed-budget path: keep the full projected H q_k here and
    ! apply the Kstore compression only after orthogonalizing against the current
    ! Krylov basis in ccik_generate_next_vector.
    call ccik_vec_copy(vec, full_vec)
    call cpu_time(t1)
    t_assign = t1 - t0
  else
    do i = 1, n_ext
      j = ext_order(n_candidates - n_ext + i)
      det_tmp(:,:,n_internal+i) = candidate_vec%det(:,:,j)
      coef_tmp(n_internal+i) = ext_coef(j)
    end do
    call cpu_time(t0)
    call ccik_vec_assign(vec, det_tmp, coef_tmp, n_total, N_int)
    call cpu_time(t1)
    t_assign = t1 - t0
  end if
  if (threshold_mode .and. vec%n > 1000000) then
    call ccik_vec_compress(vec, 1000000)
  end if

  if (ccik_verbose) then
    write(*,'(A,I6,1X,A,1X,A,6(A,I10),A,A,5(A,F12.4))') 'ccik(hq) done  call=', projected_call_count, &
         'backend=', trim(matvec_backend), '  raw=', generated_before_dedup, '  uniq=', generated_unique, &
         '  n_ext=', n_ext, '  n_total=', n_total, '  |vec|=', vec%n, '  fullq=', n_internal, &
         '  selector=', trim(selector_mode), '  select_s=', t_select, '  sort_s=', t_sort, &
         '  int_s=', t_internal, '  ext_s=', t_external, '  assign_s=', t_assign
    call flush(6)
  end if

    if (ccik_debug_trace) then
      if (full_vec%n > 0) then
        projected_full_norm = dsqrt(max(0.d0, ccik_vec_norm2(full_vec)))
      else
        projected_full_norm = 0.d0
      end if
      if (fixed_budget_mode) then
        call ccik_vec_assign(internal_trace_vec, det_tmp(:,:,1:n_internal), coef_tmp(1:n_internal), n_internal, N_int)
        call ccik_debug_trace_vector('projected_hq_internal', internal_trace_vec)
        if (n_candidates > 0) then
          call ccik_vec_assign(external_trace_vec, candidate_vec%det, ext_coef, n_candidates, N_int)
          call ccik_debug_trace_vector('projected_hq_external', external_trace_vec)
        end if
      end if
      projected_final_norm = dsqrt(max(0.d0, ccik_vec_norm2(vec)))
      write(trace_line,'(A,1X,I8,1X,A,1X,A,1X,A,1X,I8,1X,I8,1X,I8,1X,I8,1X,ES24.16,1X,L1,1X,ES24.16,1X,ES24.16,1X,ES24.16)') &
         'CCIK_TRACE_HQ', projected_call_count, trim(selector_mode), trim(matvec_backend), trim(compression_policy), n_internal, N_det_generators, &
         generated_before_dedup, n_candidates, rayleigh, ccik_propagator_used_fallback, source_dropped_norm, projected_full_norm, projected_final_norm
    call ccik_debug_trace_line(trace_line)
    call ccik_debug_trace_vector('projected_hq', vec)
  end if

  deallocate(det_tmp, coef_tmp)
  deallocate(ext_coef, pt2_score, score_sort, ext_order)
  deallocate(hpsi_internal)
  call ccik_vec_free(support_vec)
  call ccik_vec_free(parent_vec)
  call ccik_vec_free(current_vec)
  call ccik_vec_free(internal_trace_vec)
  call ccik_vec_free(external_trace_vec)
  call ccik_vec_free(candidate_vec)
  call ccik_vec_free(full_vec)
end subroutine ccik_build_projected_hq_connected_sd_fullq


subroutine ccik_orthogonalize_against_basis(vec, qbasis, n_basis)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(inout) :: vec
  type(ccik_vector), intent(in) :: qbasis(*)
  integer, intent(in) :: n_basis
  integer :: i
  double precision :: overlap

  do i = 1, n_basis
    overlap = ccik_vec_inner(qbasis(i), vec)
    call ccik_vec_axpy(vec, -overlap, qbasis(i))
  end do
end subroutine ccik_orthogonalize_against_basis


subroutine ccik_secondortho_on_support(vec, qbasis, n_basis, ok)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(inout) :: vec
  type(ccik_vector), intent(in) :: qbasis(*)
  integer, intent(in) :: n_basis
  logical, intent(out) :: ok
  type(ccik_vector), allocatable :: projected_basis(:)
  type(ccik_vector) :: support_vec, projected, restricted
  double precision, allocatable :: gram(:,:), rhs(:), evals(:), coeff(:), work(:)
  double precision :: before_norm, after_norm, max_eval, cutoff, scale, denom
  integer :: i, j, k, n_projected, lwork, info
  character(len=512) :: trace_line
  external :: dsyev

  ok = .true.
  if (vec%n <= 0 .or. n_basis <= 0) return

  before_norm = dsqrt(max(0.d0, ccik_vec_norm2(vec)))
  call ccik_vec_copy(support_vec, vec)
  allocate(projected_basis(n_basis))
  n_projected = 0

  do i = 1, n_basis
    call ccik_vec_project_support(projected, qbasis(i), support_vec)
    if (ccik_vec_norm2(projected) > 1.d-24) then
      n_projected = n_projected + 1
      call ccik_vec_copy(projected_basis(n_projected), projected)
    end if
    call ccik_vec_free(projected)
  end do

  if (n_projected <= 0) then
    call ccik_vec_free(support_vec)
    deallocate(projected_basis)
    return
  end if

  allocate(gram(n_projected,n_projected), rhs(n_projected), evals(n_projected), coeff(n_projected))
  gram = 0.d0
  rhs = 0.d0
  coeff = 0.d0
  do i = 1, n_projected
    rhs(i) = ccik_vec_inner(projected_basis(i), vec)
    do j = i, n_projected
      gram(i,j) = ccik_vec_inner(projected_basis(i), projected_basis(j))
      gram(j,i) = gram(i,j)
    end do
  end do

  lwork = max(1, 3*n_projected - 1)
  allocate(work(lwork))
  call dsyev('V', 'U', n_projected, gram, n_projected, evals, work, lwork, info)

  if (info == 0) then
    max_eval = 0.d0
    do k = 1, n_projected
      if (evals(k) > max_eval) max_eval = evals(k)
    end do
    cutoff = max(1.d-14, ccik_ritz_overlap_tol * max(1.d0, dabs(max_eval)))
    do k = 1, n_projected
      if (evals(k) <= cutoff) cycle
      scale = 0.d0
      do i = 1, n_projected
        scale = scale + gram(i,k) * rhs(i)
      end do
      scale = scale / evals(k)
      do i = 1, n_projected
        coeff(i) = coeff(i) + gram(i,k) * scale
      end do
    end do
    do i = 1, n_projected
      if (dabs(coeff(i)) > 0.d0) call ccik_vec_axpy(vec, -coeff(i), projected_basis(i))
    end do
  else
    do k = 1, 2
      do i = 1, n_projected
        denom = ccik_vec_norm2(projected_basis(i))
        if (denom <= 1.d-24) cycle
        scale = ccik_vec_inner(projected_basis(i), vec) / denom
        if (dabs(scale) > 0.d0) call ccik_vec_axpy(vec, -scale, projected_basis(i))
      end do
    end do
  end if

  call ccik_vec_project_support(restricted, vec, support_vec)
  call ccik_vec_copy(vec, restricted)
  after_norm = dsqrt(max(0.d0, ccik_vec_norm2(vec)))
  ok = (after_norm > 1.d-18)

  if (ccik_debug_trace) then
    write(trace_line,'(A,1X,I8,1X,I8,1X,ES24.16,1X,ES24.16,1X,L1)') &
         'CCIK_TRACE_SECONDORTHO', n_projected, support_vec%n, before_norm, after_norm, ok
    call ccik_debug_trace_line(trace_line)
  end if

  do i = 1, n_projected
    call ccik_vec_free(projected_basis(i))
  end do
  deallocate(projected_basis, gram, rhs, evals, coeff, work)
  call ccik_vec_free(support_vec)
  call ccik_vec_free(projected)
  call ccik_vec_free(restricted)
end subroutine ccik_secondortho_on_support


subroutine ccik_normalize_vector(vec, ok)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(inout) :: vec
  logical, intent(out) :: ok
  double precision :: norm2

  norm2 = ccik_vec_norm2(vec)
  ok = norm2 > 1.d-18
  if (.not. ok) return

  call ccik_vec_scale(vec, 1.d0 / dsqrt(norm2))
end subroutine ccik_normalize_vector


integer function ccik_tmp_keep_value(nkeep_eff)
  implicit none
  integer, intent(in) :: nkeep_eff

  ccik_tmp_keep_value = max(ccik_tmp_keep_factor * max(1, nkeep_eff), &
       max(1, nkeep_eff) + max(0, ccik_selector_keep) + max(0, ccik_external_keep))
end function ccik_tmp_keep_value


integer function ccik_preserve_cap_value(nkeep_eff)
  implicit none
  integer, intent(in) :: nkeep_eff

  if (ccik_preserve_prev_support_cap > 0) then
    ccik_preserve_cap_value = ccik_preserve_prev_support_cap
  else
    ccik_preserve_cap_value = min(2048, max(1, nkeep_eff) / 4)
  end if
end function ccik_preserve_cap_value


integer function ccik_thick_effective_nkeep(m_cycle, nroot)
  implicit none
  integer, intent(in) :: m_cycle, nroot
  integer :: denom, min_budget, budget_left

  ccik_thick_effective_nkeep = ccik_nkeep
  if (ccik_total_det_budget <= 0) return

  denom = 2*m_cycle + nroot
  min_budget = m_cycle*ccik_external_keep + denom
  if (ccik_total_det_budget < min_budget) then
    print *, 'ccik(thick): ccik_total_det_budget is too small for the requested restart dimensions'
    print *, 'ccik(thick): need at least ', min_budget, ' determinants with current m_cycle/external_keep/nroot'
    stop 2
  end if

  budget_left = ccik_total_det_budget - m_cycle*ccik_external_keep
  ccik_thick_effective_nkeep = max(1, budget_left / denom)
end function ccik_thick_effective_nkeep


subroutine ccik_generate_next_vector(qbasis, hq_basis, source_idx, n_basis, nkeep_eff, qnext, ok)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: qbasis(*), hq_basis(*)
  integer, intent(in) :: source_idx, n_basis
  integer, intent(in) :: nkeep_eff
  type(ccik_vector), intent(inout) :: qnext
  logical, intent(out) :: ok
  type(ccik_vector) :: score_vec, v_temp, probe_hq, prop_vec, rebuild_hq, store_support, orth_trace
  integer :: tmp_keep, preserve_cap, i, generated_before_dedup, generated_unique
  integer :: ccik_tmp_keep_value, ccik_preserve_cap_value
  double precision :: q_norm, v_proj_norm, v_temp_norm, discarded_weight, store_dropped_norm, selector_energy
  double precision :: orth_norm, keep_norm, overlap, max_overlap
  character(len=32) :: selector_mode, compression_policy
  character(len=512) :: trace_line
  logical :: threshold_mode, fixed_budget_mode, store_used_fallback

  call ccik_vec_free(qnext)
  call ccik_vec_free(score_vec)
  call ccik_vec_free(v_temp)
  call ccik_vec_free(probe_hq)
  call ccik_vec_free(prop_vec)
  call ccik_vec_free(rebuild_hq)
  call ccik_vec_free(store_support)
  call ccik_vec_free(orth_trace)

  call ccik_vec_copy(qnext, hq_basis(source_idx))
  q_norm = dsqrt(max(0.d0, ccik_vec_norm2(qbasis(source_idx))))
  v_proj_norm = dsqrt(max(0.d0, ccik_vec_norm2(qnext)))
  selector_mode = trim(adjustl(ccik_selector_mode))
  if (len_trim(selector_mode) == 0) selector_mode = 'abs_residual'
  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'
  threshold_mode = (compression_policy == 'threshold')
  fixed_budget_mode = (compression_policy == 'fixed_budget')
  store_dropped_norm = 0.d0
  store_used_fallback = .false.
  tmp_keep = 0
  preserve_cap = 0
  generated_before_dedup = 0
  generated_unique = 0
  ok = .true.

  if (selector_mode == 'pt2_residual' .and. .not. fixed_budget_mode) then
    print *, "ccik error: selector_mode='pt2_residual' requires compression_policy='fixed_budget'"
    stop 2
  end if

  if (threshold_mode) then
    call ccik_vec_copy(v_temp, qnext)
    v_temp_norm = dsqrt(max(0.d0, ccik_vec_norm2(v_temp)))
    discarded_weight = 0.d0
    call ccik_orthogonalize_against_basis(qnext, qbasis, n_basis)
    orth_norm = dsqrt(max(0.d0, ccik_vec_norm2(qnext)))
    if (orth_norm <= 1.d-18) then
      ok = .false.
      call ccik_vec_free(score_vec)
      call ccik_vec_free(v_temp)
      call ccik_vec_free(orth_trace)
      call ccik_vec_free(qnext)
      return
    end if
    if (ccik_debug_trace) then
      call ccik_vec_copy(orth_trace, qnext)
      call ccik_debug_trace_vector('orth_residual', orth_trace)
      call ccik_vec_free(orth_trace)
    end if
    if (selector_mode == 'abs_residual' .or. selector_mode == 'sq_residual') then
      call ccik_vec_copy(score_vec, qnext)
    else
      call ccik_vec_copy(score_vec, v_temp)
    end if
    call ccik_normalize_vector(qnext, ok)
    if (.not. ok) then
      call ccik_vec_free(score_vec)
      call ccik_vec_free(v_temp)
      call ccik_vec_free(qnext)
      return
    end if
    call ccik_vec_threshold(qnext, ccik_store_abs_tol, store_dropped_norm, store_used_fallback)
    if (qnext%n > 1000000) then
      call ccik_vec_compress(qnext, 1000000)
    end if
    keep_norm = dsqrt(max(0.d0, ccik_vec_norm2(qnext)))
  else if (fixed_budget_mode) then
    if (selector_mode == 'pt2_residual') then
      selector_energy = ccik_vec_inner(qbasis(source_idx), qnext)
      call ccik_vec_copy(v_temp, qnext)
      v_temp_norm = dsqrt(max(0.d0, ccik_vec_norm2(v_temp)))
      discarded_weight = 0.d0
      call ccik_orthogonalize_against_basis(qnext, qbasis, n_basis)
      orth_norm = dsqrt(max(0.d0, ccik_vec_norm2(qnext)))
      if (orth_norm <= 1.d-18) then
        ok = .false.
        call ccik_vec_free(score_vec)
        call ccik_vec_free(v_temp)
        call ccik_vec_free(probe_hq)
        call ccik_vec_free(prop_vec)
        call ccik_vec_free(rebuild_hq)
        call ccik_vec_free(store_support)
        call ccik_vec_free(orth_trace)
        call ccik_vec_free(qnext)
        return
      end if
      if (ccik_debug_trace) then
        call ccik_vec_copy(orth_trace, qnext)
        call ccik_debug_trace_vector('orth_residual', orth_trace)
        call ccik_vec_free(orth_trace)
      end if
      call ccik_select_top_pt2_residual(qnext, selector_energy, max(1, ccik_fixed_store_keep), store_support)
      call ccik_vec_project_support(v_temp, qnext, store_support)
      store_dropped_norm = dsqrt(max(0.d0, ccik_vec_norm2(qnext) - ccik_vec_norm2(v_temp)))
      call ccik_vec_copy(qnext, v_temp)
      if (ccik_secondortho) call ccik_secondortho_on_support(qnext, qbasis, n_basis, ok)
      if (ok) call ccik_select_top_pt2_residual(qnext, selector_energy, max(1, ccik_fixed_propagate_keep), prop_vec)
      keep_norm = dsqrt(max(0.d0, ccik_vec_norm2(qnext)))
    else
      call ccik_vec_copy(v_temp, qnext)
      v_temp_norm = dsqrt(max(0.d0, ccik_vec_norm2(v_temp)))
      discarded_weight = 0.d0
      call ccik_orthogonalize_against_basis(qnext, qbasis, n_basis)
      orth_norm = dsqrt(max(0.d0, ccik_vec_norm2(qnext)))
      if (orth_norm <= 1.d-18) then
        ok = .false.
        call ccik_vec_free(score_vec)
        call ccik_vec_free(v_temp)
        call ccik_vec_free(probe_hq)
        call ccik_vec_free(prop_vec)
        call ccik_vec_free(rebuild_hq)
        call ccik_vec_free(store_support)
        call ccik_vec_free(orth_trace)
        call ccik_vec_free(qnext)
        return
      end if
      if (ccik_debug_trace) then
        call ccik_vec_copy(orth_trace, qnext)
        call ccik_debug_trace_vector('orth_residual', orth_trace)
        call ccik_vec_free(orth_trace)
      end if
      if (selector_mode == 'abs_residual' .or. selector_mode == 'sq_residual') then
        call ccik_vec_copy(score_vec, qnext)
      else
        call ccik_vec_copy(score_vec, v_temp)
      end if
      call ccik_vec_compress(qnext, max(1, ccik_fixed_store_keep))
      if (ccik_secondortho) call ccik_secondortho_on_support(qnext, qbasis, n_basis, ok)
      keep_norm = dsqrt(max(0.d0, ccik_vec_norm2(qnext)))
    end if
  else
    tmp_keep = min(qnext%n, ccik_tmp_keep_value(nkeep_eff))

    if (selector_mode == 'abs_residual' .or. selector_mode == 'sq_residual') then
      call ccik_vec_copy(score_vec, qnext)
      call ccik_orthogonalize_against_basis(score_vec, qbasis, n_basis)
      if (score_vec%n <= 0) then
        ok = .false.
        call ccik_vec_free(score_vec)
        call ccik_vec_free(v_temp)
        call ccik_vec_free(qnext)
        return
      end if
      call ccik_vec_compress(score_vec, tmp_keep)
      call ccik_vec_project_support(v_temp, qnext, score_vec)
    else
      call ccik_vec_copy(v_temp, qnext)
      call ccik_vec_compress(v_temp, tmp_keep)
    end if

    v_temp_norm = dsqrt(max(0.d0, ccik_vec_norm2(v_temp)))
    discarded_weight = dsqrt(max(v_proj_norm*v_proj_norm - v_temp_norm*v_temp_norm, 0.d0))

    call ccik_vec_copy(qnext, v_temp)
    call ccik_orthogonalize_against_basis(qnext, qbasis, n_basis)
    orth_norm = dsqrt(max(0.d0, ccik_vec_norm2(qnext)))
    if (orth_norm <= 1.d-18) then
      ok = .false.
      call ccik_vec_free(score_vec)
      call ccik_vec_free(v_temp)
      call ccik_vec_free(orth_trace)
      call ccik_vec_free(qnext)
      return
    end if
    if (ccik_debug_trace) then
      call ccik_vec_copy(orth_trace, qnext)
      call ccik_debug_trace_vector('orth_residual', orth_trace)
      call ccik_vec_free(orth_trace)
    end if

    if (ccik_preserve_prev_support .and. ccik_include_current_support) then
      preserve_cap = ccik_preserve_cap_value(nkeep_eff)
    end if
    if (preserve_cap > 0) then
      call ccik_vec_compress_preserve(qnext, nkeep_eff, qbasis(source_idx), preserve_cap)
    else
      call ccik_vec_compress(qnext, nkeep_eff)
    end if
    keep_norm = dsqrt(max(0.d0, ccik_vec_norm2(qnext)))
  end if
  call ccik_normalize_vector(qnext, ok)
  if (ok .and. fixed_budget_mode .and. selector_mode == 'pt2_residual') then
    call ccik_vec_set_propagation_support(qnext, prop_vec)
  end if
  max_overlap = 0.d0
  if (ok) then
    do i = 1, n_basis
      overlap = dabs(ccik_vec_inner(qbasis(i), qnext))
      if (overlap > max_overlap) max_overlap = overlap
    end do
  end if

  if (ccik_verbose) then
    write(*,'(A,I4,A,A,A,A,6(A,F12.4),3(A,I10),A,ES12.4,A,ES12.4)') 'ccik(next) parent=', source_idx, &
         ' policy=', trim(compression_policy), ' selector=', trim(selector_mode), &
         '  |q|=', q_norm, '  |v_proj|=', v_proj_norm, &
         '  |v_temp|=', v_temp_norm, '  dropped=', discarded_weight, '  |r_orth|=', orth_norm, &
         '  |r_keep|=', keep_norm, '  tmp_keep=', tmp_keep, '  final_keep=', qnext%n, &
         '  preserve_cap=', preserve_cap, '  overlap=', max_overlap, '  store_drop=', store_dropped_norm
    call flush(6)
  end if

  if (ccik_debug_trace) then
    write(trace_line,'(A,1X,I8,1X,A,1X,A,1X,I8,1X,I8,1X,I8,1X,L1,1X,ES24.16,1X,ES24.16,1X,ES24.16,1X,ES24.16,1X,ES24.16,1X,ES24.16,1X,ES24.16,1X,ES24.16)') &
         'CCIK_TRACE_NEXT', source_idx, trim(selector_mode), trim(compression_policy), tmp_keep, qnext%n, preserve_cap, store_used_fallback, q_norm, v_proj_norm, &
         v_temp_norm, discarded_weight, orth_norm, keep_norm, store_dropped_norm, max_overlap
    call ccik_debug_trace_line(trace_line)
    call ccik_debug_trace_vector('score_vec', score_vec)
    call ccik_debug_trace_vector('v_temp', v_temp)
    call ccik_debug_trace_vector('q_next', qnext)
  end if

  call ccik_vec_free(score_vec)
  call ccik_vec_free(v_temp)
  call ccik_vec_free(probe_hq)
  call ccik_vec_free(prop_vec)
  call ccik_vec_free(rebuild_hq)
  call ccik_vec_free(store_support)
  call ccik_vec_free(orth_trace)
end subroutine ccik_generate_next_vector


subroutine ccik_restart_from_ritz(qbasis, hq_basis, n_basis, nkeep_eff, restart_basis, n_keep, n_restart_alloc, cycle_idx)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: qbasis(*), hq_basis(*)
  integer, intent(in) :: n_basis, nkeep_eff, n_keep, n_restart_alloc, cycle_idx
  type(ccik_vector), intent(inout) :: restart_basis(*)
  type(ccik_vector), allocatable :: restart_raw(:)
  double precision, allocatable :: eig(:), eigvec(:,:)
  type(ccik_vector) :: trial
  integer :: i, j, nkeep_stage1, n_restart_built
  integer :: ccik_tmp_keep_value
  logical :: ok, store_used_fallback
  double precision :: store_dropped_norm
  character(len=32) :: compression_policy
  character(len=256) :: trace_line
  character(len=64) :: restart_tag

  allocate(eig(n_basis), eigvec(n_basis,n_basis))
  allocate(restart_raw(max(1, n_keep)))
  call ccik_solve_ritz_problem(qbasis, hq_basis, n_basis, eig, eigvec)
  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'

  do i = 1, max(1, n_keep)
    call ccik_vec_free(restart_raw(i))
  end do

  if (compression_policy == 'topk') then
    ! Match the Python thick-restart semantics for legacy top-k:
    ! build/compress/normalize each Ritz vector first, then perform
    ! a separate Gram-Schmidt pass over the restart set.
    do i = 1, n_keep
      call ccik_vec_free(trial)
      do j = 1, n_basis
        call ccik_vec_axpy(trial, eigvec(j,i), qbasis(j))
      end do
      nkeep_stage1 = min(trial%n, ccik_tmp_keep_value(nkeep_eff))
      call ccik_vec_compress(trial, nkeep_stage1)
      call ccik_vec_compress(trial, nkeep_eff)
      call ccik_normalize_vector(trial, ok)
      if (ccik_debug_trace) then
        write(trace_line,'(A,1X,A,1X,I8,1X,I8,1X,L1)') 'CCIK_TRACE_RESTART', 'raw_topk', i, trial%n, ok
        call ccik_debug_trace_line(trace_line)
        call ccik_debug_trace_vector('restart_raw_topk', trial)
      end if
      call ccik_vec_free(restart_raw(i))
      if (ok) then
        call ccik_vec_copy(restart_raw(i), trial)
      end if
      call ccik_vec_free(trial)
    end do

    do i = 1, n_restart_alloc
      call ccik_vec_free(restart_basis(i))
    end do

    n_restart_built = 0
    do i = 1, n_keep
      if (restart_raw(i)%n == 0) cycle
      call ccik_vec_free(trial)
      call ccik_vec_copy(trial, restart_raw(i))
      if (n_restart_built > 0) then
        call ccik_orthogonalize_against_basis(trial, restart_basis, n_restart_built)
      end if
      call ccik_normalize_vector(trial, ok)
      if (ccik_debug_trace) then
        write(trace_line,'(A,1X,A,1X,I8,1X,I8,1X,L1)') 'CCIK_TRACE_RESTART', 'gs_topk', i, trial%n, ok
        call ccik_debug_trace_line(trace_line)
        call ccik_debug_trace_vector('restart_gs_topk', trial)
      end if
      if (.not. ok) then
        call ccik_vec_free(trial)
        cycle
      end if
      n_restart_built = n_restart_built + 1
      call ccik_vec_copy(restart_basis(n_restart_built), trial)
      call ccik_vec_free(trial)
    end do
  else if (compression_policy == 'fixed_budget') then
    ! Match the Python fixed-budget thick-restart semantics:
    ! build/compress/normalize each Ritz vector first, then perform
    ! a separate Gram-Schmidt pass over the restart set.
    do i = 1, n_keep
      call ccik_vec_free(trial)
      do j = 1, n_basis
        call ccik_vec_axpy(trial, eigvec(j,i), qbasis(j))
      end do
      if (ccik_debug_trace) then
        write(trace_line,'(A,1X,A,1X,I8,1X,I8,1X,ES24.16,1X,I8,1X,L1)') 'CCIK_TRACE_RESTART', 'raw', cycle_idx, i, eig(i), trial%n, trial%n > 0
        call ccik_debug_trace_line(trace_line)
        write(restart_tag,'(A,I0,A,I0)') 'restart_fb_raw_c', cycle_idx, '_r', i
        call ccik_debug_trace_vector(trim(restart_tag), trial)
      end if
      call ccik_vec_compress(trial, max(1, ccik_fixed_store_keep))
      if (ccik_debug_trace) then
        write(trace_line,'(A,1X,A,1X,I8,1X,I8,1X,ES24.16,1X,I8,1X,L1)') 'CCIK_TRACE_RESTART', 'compressed', cycle_idx, i, eig(i), trial%n, trial%n > 0
        call ccik_debug_trace_line(trace_line)
        write(restart_tag,'(A,I0,A,I0)') 'restart_fb_compressed_c', cycle_idx, '_r', i
        call ccik_debug_trace_vector(trim(restart_tag), trial)
      end if
      call ccik_normalize_vector(trial, ok)
      if (ccik_debug_trace) then
        write(trace_line,'(A,1X,A,1X,I8,1X,I8,1X,ES24.16,1X,I8,1X,L1)') 'CCIK_TRACE_RESTART', 'normalized', cycle_idx, i, eig(i), trial%n, ok
        call ccik_debug_trace_line(trace_line)
        write(restart_tag,'(A,I0,A,I0)') 'restart_fb_normalized_c', cycle_idx, '_r', i
        call ccik_debug_trace_vector(trim(restart_tag), trial)
      end if
      call ccik_vec_free(restart_raw(i))
      if (ok) then
        call ccik_vec_copy(restart_raw(i), trial)
      end if
      call ccik_vec_free(trial)
    end do

    do i = 1, n_restart_alloc
      call ccik_vec_free(restart_basis(i))
    end do

    n_restart_built = 0
    do i = 1, n_keep
      if (restart_raw(i)%n == 0) cycle
      call ccik_vec_free(trial)
      call ccik_vec_copy(trial, restart_raw(i))
      if (n_restart_built > 0) then
        call ccik_orthogonalize_against_basis(trial, restart_basis, n_restart_built)
      end if
      if (ccik_debug_trace) then
        write(trace_line,'(A,1X,A,1X,I8,1X,I8,1X,ES24.16,1X,I8,1X,L1)') 'CCIK_TRACE_RESTART', 'postorth', cycle_idx, i, eig(i), trial%n, trial%n > 0
        call ccik_debug_trace_line(trace_line)
        write(restart_tag,'(A,I0,A,I0)') 'restart_fb_postorth_c', cycle_idx, '_r', i
        call ccik_debug_trace_vector(trim(restart_tag), trial)
      end if
      ! The fixed-budget restart vectors must still satisfy the stored-vector
      ! cap after the second Gram-Schmidt pass. Post-orthogonalization can
      ! reintroduce determinants outside the initial top-K support, so compress
      ! once more before normalizing/saving the restart vector for the next
      ! cycle.
      call ccik_vec_compress(trial, max(1, ccik_fixed_store_keep))
      call ccik_normalize_vector(trial, ok)
      if (ccik_debug_trace) then
        write(trace_line,'(A,1X,A,1X,I8,1X,I8,1X,ES24.16,1X,I8,1X,L1)') 'CCIK_TRACE_RESTART', 'final', cycle_idx, i, eig(i), trial%n, ok
        call ccik_debug_trace_line(trace_line)
        write(restart_tag,'(A,I0,A,I0)') 'restart_fb_final_c', cycle_idx, '_r', i
        call ccik_debug_trace_vector(trim(restart_tag), trial)
      end if
      if (.not. ok) then
        call ccik_vec_free(trial)
        cycle
      end if
      n_restart_built = n_restart_built + 1
      call ccik_vec_copy(restart_basis(n_restart_built), trial)
      call ccik_vec_free(trial)
    end do

    do i = n_restart_built + 1, n_restart_alloc
      call ccik_vec_free(restart_basis(i))
    end do
  else
    do i = 1, n_keep
      call ccik_vec_free(trial)
      do j = 1, n_basis
        call ccik_vec_axpy(trial, eigvec(j,i), qbasis(j))
      end do
      if (i > 1) then
        call ccik_orthogonalize_against_basis(trial, restart_basis, i-1)
      end if
      if (compression_policy == 'threshold') then
        call ccik_normalize_vector(trial, ok)
        if (.not. ok) cycle
        call ccik_vec_threshold(trial, ccik_store_abs_tol, store_dropped_norm, store_used_fallback)
        if (trial%n > 1000000) then
          call ccik_vec_compress(trial, 1000000)
        end if
      else
        nkeep_stage1 = min(trial%n, ccik_tmp_keep_value(nkeep_eff))
        call ccik_vec_compress(trial, nkeep_stage1)
        if (i > 1) then
          call ccik_orthogonalize_against_basis(trial, restart_basis, i-1)
        end if
        call ccik_vec_compress(trial, nkeep_eff)
      end if
      call ccik_normalize_vector(trial, ok)
      call ccik_vec_free(restart_basis(i))
      if (ok) then
        call ccik_vec_copy(restart_basis(i), trial)
      end if
      call ccik_vec_free(trial)
    end do

    do i = n_keep + 1, n_restart_alloc
      call ccik_vec_free(restart_basis(i))
    end do
  end if

  do i = 1, max(1, n_keep)
    call ccik_vec_free(restart_raw(i))
  end do

  deallocate(eig, eigvec)
  deallocate(restart_raw)
end subroutine ccik_restart_from_ritz


subroutine ccik_build_lowest_ritz_vector(qbasis, hq_basis, n_basis, vec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: qbasis(*), hq_basis(*)
  integer, intent(in) :: n_basis
  type(ccik_vector), intent(inout) :: vec
  double precision, allocatable :: eig(:), eigvec(:,:)
  character(len=32) :: compression_policy
  integer :: j
  logical :: ok

  call ccik_vec_free(vec)
  allocate(eig(n_basis), eigvec(n_basis,n_basis))
  call ccik_solve_ritz_problem(qbasis, hq_basis, n_basis, eig, eigvec)
  do j = 1, n_basis
    call ccik_vec_axpy(vec, eigvec(j,1), qbasis(j))
  end do
  call ccik_normalize_vector(vec, ok)
  if (.not. ok) then
    call ccik_vec_free(vec)
    call ccik_vec_copy(vec, qbasis(1))
  else
    compression_policy = trim(adjustl(ccik_compression_policy))
    if (len_trim(compression_policy) == 0) compression_policy = 'topk'
    if (compression_policy == 'threshold' .and. vec%n > 1000000) then
      call ccik_vec_compress(vec, 1000000)
      call ccik_normalize_vector(vec, ok)
      if (.not. ok) then
        call ccik_vec_free(vec)
        call ccik_vec_copy(vec, qbasis(1))
      end if
    else if (compression_policy == 'fixed_budget' .and. vec%n > max(1, ccik_fixed_store_keep)) then
      call ccik_vec_compress(vec, max(1, ccik_fixed_store_keep))
      call ccik_normalize_vector(vec, ok)
      if (.not. ok) then
        call ccik_vec_free(vec)
        call ccik_vec_copy(vec, qbasis(1))
      end if
    end if
  end if
  deallocate(eig, eigvec)
end subroutine ccik_build_lowest_ritz_vector


subroutine ccik_save_vector_to_ezfio(vec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: vec

  if (vec%n <= 0) return
  call ccik_load_vector_into_working_wf(vec)
  call save_wavefunction_unsorted
  if (ccik_verbose) then
    write(*,'(A,I10)') 'ccik: saved final Ritz wavefunction, N_det = ', vec%n
    call flush(6)
  end if
end subroutine ccik_save_vector_to_ezfio


subroutine ccik_compute_post_pt2(vec, electronic_energy, pt2_corr, rpt2_corr)
  use selection_types
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: vec
  double precision, intent(in) :: electronic_energy
  double precision, intent(out) :: pt2_corr, rpt2_corr
  type(selection_buffer) :: buf
  type(pt2_type) :: pt2_data
  double precision :: e0(1), threshold_generators_save
  integer :: i

  pt2_corr = 0.d0
  rpt2_corr = 0.d0
  if (vec%n <= 0) return

  call ccik_load_vector_into_working_wf(vec)
  threshold_generators_save = threshold_generators
  threshold_generators = 1.d0
  PROVIDE psi_selectors psi_selectors_coef psi_selectors_size N_det_selectors N_det_generators
  PROVIDE psi_energy

  call create_selection_buffer(1, 2, buf)
  call pt2_alloc(pt2_data, N_states)
  e0(1) = psi_energy(1)
  do i = 1, N_det_generators
    call select_connected(i, e0, pt2_data, buf, 1, 1)
  end do
  pt2_corr = pt2_data % pt2(1)
  rpt2_corr = pt2_corr / (1.d0 + pt2_data % overlap(1,1))
  call pt2_dealloc(pt2_data)
  call delete_selection_buffer(buf)

  threshold_generators = threshold_generators_save
end subroutine ccik_compute_post_pt2


subroutine ccik_smallest_ritz_energy(qbasis, hq_basis, n_basis, energy)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: qbasis(*), hq_basis(*)
  integer, intent(in) :: n_basis
  double precision, intent(out) :: energy
  double precision, allocatable :: eig(:), eigvec(:,:)

  allocate(eig(n_basis), eigvec(n_basis,n_basis))
  call ccik_solve_ritz_problem(qbasis, hq_basis, n_basis, eig, eigvec)
  energy = eig(1)
  deallocate(eig, eigvec)
end subroutine ccik_smallest_ritz_energy


subroutine ccik_solve_ritz_problem(qbasis, hq_basis, n_basis, eig, eigvec)
  use ccik_sparse
  implicit none
  type(ccik_vector), intent(in) :: qbasis(*), hq_basis(*)
  integer, intent(in) :: n_basis
  double precision, intent(out) :: eig(n_basis), eigvec(n_basis,n_basis)
  double precision, allocatable :: hmat(:,:), smat(:,:), work(:), diag_e(:)
  double precision, allocatable :: swork(:,:), xmat(:,:), hsmall(:,:), eval_small(:)
  integer, allocatable :: diag_order(:)
  integer :: i, j, k, info, lwork, nkeep
  double precision :: smax, stol, diag_min, smat_diag_min, smat_diag_max, smat_offdiag_max
  character(len=256) :: trace_line

  allocate(hmat(n_basis,n_basis), smat(n_basis,n_basis))
  do j = 1, n_basis
    do i = 1, n_basis
      hmat(i,j) = ccik_vec_inner(qbasis(i), hq_basis(j))
      smat(i,j) = ccik_vec_inner(qbasis(i), qbasis(j))
    end do
  end do

  do i = 1, n_basis
    do j = i+1, n_basis
      hmat(i,j) = 0.5d0 * (hmat(i,j) + hmat(j,i))
      hmat(j,i) = hmat(i,j)
      smat(i,j) = 0.5d0 * (smat(i,j) + smat(j,i))
      smat(j,i) = smat(i,j)
    end do
  end do

  allocate(diag_e(n_basis), diag_order(n_basis))
  do i = 1, n_basis
    diag_e(i) = hmat(i,i) / max(1.d-14, smat(i,i))
    diag_order(i) = i
  end do
  diag_min = minval(diag_e)
  smat_diag_min = huge(1.d0)
  smat_diag_max = 0.d0
  smat_offdiag_max = 0.d0
  do i = 1, n_basis
    smat_diag_min = min(smat_diag_min, smat(i,i))
    smat_diag_max = max(smat_diag_max, smat(i,i))
    do j = 1, n_basis
      if (i /= j) smat_offdiag_max = max(smat_offdiag_max, dabs(smat(i,j)))
    end do
  end do
  if (ccik_debug_trace) then
    write(trace_line,'(A,1X,I8,1X,4ES24.16)') 'CCIK_TRACE_RITZ_INPUT', n_basis, diag_min, smat_diag_min, smat_diag_max, smat_offdiag_max
    call ccik_debug_trace_line(trace_line)
  end if

  lwork = max(1, 3*n_basis*n_basis)
  allocate(work(lwork))
  call dsyev('V', 'U', n_basis, smat, n_basis, eig, work, lwork, info)
  if (info /= 0) then
    call ccik_fallback_diagonal_ritz(diag_e, diag_order, n_basis, eig, eigvec)
    deallocate(hmat, smat, work, diag_e, diag_order)
    return
  end if

  smax = max(1.d-14, eig(n_basis))
  stol = max(1.d-14, ccik_ritz_overlap_tol * smax)
  nkeep = 0
  do i = 1, n_basis
    if (eig(i) > stol) nkeep = nkeep + 1
  end do
  if (ccik_debug_trace) then
    write(trace_line,'(A,1X,I8,1X,4ES24.16)') 'CCIK_TRACE_RITZ_S', nkeep, eig(1), eig(n_basis), stol, diag_min
    call ccik_debug_trace_line(trace_line)
  end if

  if (nkeep <= 0) then
    call ccik_fallback_diagonal_ritz(diag_e, diag_order, n_basis, eig, eigvec)
    deallocate(hmat, smat, work, diag_e, diag_order)
    return
  end if

  allocate(xmat(n_basis,nkeep), hsmall(nkeep,nkeep), eval_small(nkeep))
  k = 0
  do i = 1, n_basis
    if (eig(i) <= stol) cycle
    k = k + 1
    xmat(:,k) = smat(:,i) / dsqrt(eig(i))
  end do

  hsmall = matmul(transpose(xmat), matmul(hmat, xmat))
  do i = 1, nkeep
    do j = i+1, nkeep
      hsmall(i,j) = 0.5d0 * (hsmall(i,j) + hsmall(j,i))
      hsmall(j,i) = hsmall(i,j)
    end do
  end do

  deallocate(work)
  lwork = max(1, 3*nkeep*nkeep)
  allocate(work(lwork))
  call dsyev('V', 'U', nkeep, hsmall, nkeep, eval_small, work, lwork, info)
  if (info /= 0) then
    call ccik_fallback_diagonal_ritz(diag_e, diag_order, n_basis, eig, eigvec)
    deallocate(hmat, smat, work, diag_e, diag_order, xmat, hsmall, eval_small)
    return
  end if
  if (ccik_debug_trace) then
    write(trace_line,'(A,1X,I8,1X,3ES24.16)') 'CCIK_TRACE_RITZ_H', nkeep, eval_small(1), eval_small(min(2, nkeep)), diag_min
    call ccik_debug_trace_line(trace_line)
  end if

  eig = huge(1.d0)
  eigvec = 0.d0
  eig(1:nkeep) = eval_small(1:nkeep)
  eigvec(:,1:nkeep) = matmul(xmat, hsmall)

  if (ccik_ritz_max_drop > 0.d0 .and. eig(1) < diag_min - ccik_ritz_max_drop) then
    if (ccik_verbose) then
      write(*,'(A,3F24.12)') 'ccik: rejecting pathological Ritz root. eig(1), diag_min, max_drop = ', &
           eig(1), diag_min, ccik_ritz_max_drop
      call flush(6)
    end if
    if (ccik_debug_trace) then
      write(trace_line,'(A,1X,3ES24.16)') 'CCIK_TRACE_RITZ_REJECT', eig(1), diag_min, ccik_ritz_max_drop
      call ccik_debug_trace_line(trace_line)
    end if
    call ccik_fallback_diagonal_ritz(diag_e, diag_order, n_basis, eig, eigvec)
  end if

  deallocate(hmat, smat, work, diag_e, diag_order, xmat, hsmall, eval_small)
end subroutine ccik_solve_ritz_problem


subroutine ccik_fallback_diagonal_ritz(diag_e, diag_order, n_basis, eig, eigvec)
  implicit none
  integer, intent(in) :: n_basis
  double precision, intent(in) :: diag_e(n_basis)
  integer, intent(inout) :: diag_order(n_basis)
  double precision, intent(out) :: eig(n_basis), eigvec(n_basis,n_basis)
  double precision, allocatable :: sort_work(:)
  integer :: i

  allocate(sort_work(n_basis))
  do i = 1, n_basis
    sort_work(i) = diag_e(i)
    diag_order(i) = i
  end do
  if (n_basis > 1) call dsort(sort_work, diag_order, n_basis)

  eig = huge(1.d0)
  eigvec = 0.d0
  do i = 1, n_basis
    eig(i) = sort_work(i)
    eigvec(diag_order(i), i) = 1.d0
  end do
  deallocate(sort_work)
end subroutine ccik_fallback_diagonal_ritz
