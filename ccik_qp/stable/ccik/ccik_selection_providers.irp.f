use bitmasks

BEGIN_PROVIDER [ integer, N_det_propagators ]
&BEGIN_PROVIDER [ double precision, ccik_propagator_dropped_norm ]
&BEGIN_PROVIDER [ logical, ccik_propagator_used_fallback ]
  implicit none
  integer :: i, count, max_idx, j, left, right, ii, jj, tmp_idx
  integer, parameter :: threshold_propagate_max_det = 10000
  integer, allocatable :: idx(:)
  double precision, allocatable :: abs_sorted(:)
  double precision :: tau, abscoef, max_abs, dropped_sq, total_sq, tmp_abs, tie_tol
  character(len=32) :: compression_policy
  logical :: fixed_budget_mode

  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'
  fixed_budget_mode = (compression_policy == 'fixed_budget')

  if (fixed_budget_mode) then
    if (N_det <= 0) then
      N_det_propagators = 0
      ccik_propagator_dropped_norm = 0.d0
      ccik_propagator_used_fallback = .false.
      return
    end if

    count = min(min(N_det, psi_det_size), max(1, ccik_fixed_propagate_keep))
    allocate(abs_sorted(max(1, N_det)), idx(max(1, N_det)))
    dropped_sq = 0.d0
    do i = 1, N_det
      j = psi_det_working_sorted_order(i)
      abs_sorted(i) = dabs(psi_coef(j,1))
      idx(i) = i
    end do
    if (N_det > 1) call dsort(abs_sorted, idx, N_det)
    left = 1
    do while (left <= N_det)
      tie_tol = max(1.d-15, 1.d-12 * max(1.d0, dabs(abs_sorted(left))))
      right = left
      do while (right < N_det)
        if (dabs(abs_sorted(right+1) - abs_sorted(left)) > tie_tol) exit
        right = right + 1
      end do
      if (right > left) then
        do ii = left + 1, right
          tmp_idx = idx(ii)
          tmp_abs = abs_sorted(ii)
          jj = ii - 1
          do while (jj >= left .and. idx(jj) > tmp_idx)
            idx(jj+1) = idx(jj)
            abs_sorted(jj+1) = abs_sorted(jj)
            jj = jj - 1
          end do
          idx(jj+1) = tmp_idx
          abs_sorted(jj+1) = tmp_abs
        end do
      end if
      left = right + 1
    end do
    do i = 1, max(0, N_det - count)
      j = psi_det_working_sorted_order(idx(i))
      dropped_sq = dropped_sq + psi_coef(j,1) * psi_coef(j,1)
    end do
    deallocate(abs_sorted, idx)
    N_det_propagators = count
    ccik_propagator_dropped_norm = dsqrt(max(0.d0, dropped_sq))
    ccik_propagator_used_fallback = .false.
    return
  end if

  if (compression_policy /= 'threshold') then
    N_det_propagators = N_det
    ccik_propagator_dropped_norm = 0.d0
    ccik_propagator_used_fallback = .false.
    return
  end if

  tau = max(0.d0, ccik_propagate_abs_tol)
  count = 0
  max_idx = 1
  max_abs = -1.d0
  dropped_sq = 0.d0
  total_sq = 0.d0
  do i = 1, N_det
    abscoef = dabs(psi_coef(i,1))
    total_sq = total_sq + psi_coef(i,1) * psi_coef(i,1)
    if (abscoef > max_abs) then
      max_abs = abscoef
      max_idx = i
    end if
    if (abscoef >= tau) then
      count = count + 1
    else
      dropped_sq = dropped_sq + psi_coef(i,1) * psi_coef(i,1)
    end if
  end do

  if (count > 0) then
    if (count > threshold_propagate_max_det) then
      allocate(abs_sorted(max(1, N_det)), idx(max(1, N_det)))
      do i = 1, N_det
        abs_sorted(i) = dabs(psi_coef(i,1))
        idx(i) = i
      end do
      if (N_det > 1) call dsort(abs_sorted, idx, N_det)
      do i = N_det - count + 1, N_det - threshold_propagate_max_det
        dropped_sq = dropped_sq + psi_coef(idx(i),1) * psi_coef(idx(i),1)
      end do
      deallocate(abs_sorted, idx)
      N_det_propagators = threshold_propagate_max_det
    else
      N_det_propagators = count
    end if
    ccik_propagator_dropped_norm = dsqrt(max(0.d0, dropped_sq))
    ccik_propagator_used_fallback = .false.
  else
    N_det_propagators = min(1, N_det)
    ccik_propagator_dropped_norm = dsqrt(max(0.d0, total_sq - psi_coef(max_idx,1) * psi_coef(max_idx,1)))
    ccik_propagator_used_fallback = (N_det > 0)
  end if
END_PROVIDER

BEGIN_PROVIDER [ integer, ccik_fixed_budget_propagator_idx, (psi_det_size) ]
  implicit none
  integer, allocatable :: idx(:)
  integer :: i, j, nkeep, left, right, ii, jj, tmp_idx
  double precision, allocatable :: abscoef(:)
  double precision :: tmp_abs, tie_tol
  character(len=32) :: compression_policy
  logical :: fixed_budget_mode

  ccik_fixed_budget_propagator_idx = 0
  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'
  fixed_budget_mode = (compression_policy == 'fixed_budget')

  if (.not. fixed_budget_mode) return
  if (N_det <= 0 .or. N_det_propagators <= 0) return

  allocate(abscoef(max(1, N_det)), idx(max(1, N_det)))
  do i = 1, N_det
    j = psi_det_working_sorted_order(i)
    if (j <= 0 .or. j > N_det) then
      write(*,'(A)') 'ccik error: invalid psi_det_working_sorted_order during fixed-budget propagator packing'
      write(*,'(A,I10,A,I10,A,I10)') 'ccik error: i=', i, '  order=', j, '  N_det=', N_det
      call flush(6)
      stop 2
    end if
    abscoef(i) = dabs(psi_coef(j,1))
    idx(i) = i
  end do
  if (N_det > 1) call dsort(abscoef, idx, N_det)

  left = 1
  do while (left <= N_det)
    tie_tol = max(1.d-15, 1.d-12 * max(1.d0, dabs(abscoef(left))))
    right = left
    do while (right < N_det)
      if (dabs(abscoef(right+1) - abscoef(left)) > tie_tol) exit
      right = right + 1
    end do
    if (right > left) then
      do ii = left + 1, right
        tmp_idx = idx(ii)
        tmp_abs = abscoef(ii)
        jj = ii - 1
        do while (jj >= left .and. idx(jj) > tmp_idx)
          idx(jj+1) = idx(jj)
          abscoef(jj+1) = abscoef(jj)
          jj = jj - 1
        end do
        idx(jj+1) = tmp_idx
        abscoef(jj+1) = tmp_abs
      end do
    end if
    left = right + 1
  end do

  nkeep = min(N_det_propagators, N_det)
  do i = 1, nkeep
    j = psi_det_working_sorted_order(idx(N_det - nkeep + i))
    if (j <= 0 .or. j > N_det) then
      write(*,'(A)') 'ccik error: invalid selected propagator index during fixed-budget propagator packing'
      write(*,'(A,I10,A,I10,A,I10)') 'ccik error: slot=', i, '  selected=', j, '  N_det=', N_det
      call flush(6)
      stop 2
    end if
    ccik_fixed_budget_propagator_idx(i) = j
  end do

  deallocate(abscoef, idx)
END_PROVIDER

BEGIN_PROVIDER [ integer(bit_kind), psi_det_propagators, (N_int,2,psi_det_size) ]
&BEGIN_PROVIDER [ double precision, psi_coef_propagators, (psi_det_size,N_states) ]
  implicit none
  integer, allocatable :: idx(:)
  integer :: i, j, k, nkeep, max_idx, left, right, ii, jj, tmp_idx
  double precision, allocatable :: abscoef(:)
  double precision :: tau, max_abs, tmp_abs, tie_tol
  character(len=32) :: compression_policy
  logical :: fixed_budget_mode

  psi_det_propagators = 0_bit_kind
  psi_coef_propagators = 0.d0
  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'
  fixed_budget_mode = (compression_policy == 'fixed_budget')

  if (fixed_budget_mode) then
    if (N_det <= 0) return
    nkeep = min(N_det_propagators, N_det)
    do i = 1, nkeep
      j = ccik_fixed_budget_propagator_idx(i)
      if (j <= 0 .or. j > N_det) then
        write(*,'(A)') 'ccik error: fixed-budget propagator packing produced an invalid source determinant index'
        write(*,'(A,I10,A,I10,A,I10)') 'ccik error: slot=', i, '  source=', j, '  N_det=', N_det
        call flush(6)
        stop 2
      end if
      do k = 1, N_int
        psi_det_propagators(k,1,i) = psi_det(k,1,j)
        psi_det_propagators(k,2,i) = psi_det(k,2,j)
      end do
      psi_coef_propagators(i,:) = psi_coef(j,:)
    end do
    return
  end if

  if (compression_policy /= 'threshold') then
    do i = 1, N_det
      do k = 1, N_int
        psi_det_propagators(k,1,i) = psi_det(k,1,i)
        psi_det_propagators(k,2,i) = psi_det(k,2,i)
      end do
      psi_coef_propagators(i,:) = psi_coef(i,:)
    end do
    return
  end if

  tau = max(0.d0, ccik_propagate_abs_tol)
  nkeep = 0
  max_idx = 1
  max_abs = -1.d0
  allocate(abscoef(max(1, N_det)), idx(max(1, N_det)))
  do i = 1, N_det
    abscoef(i) = dabs(psi_coef(i,1))
    idx(i) = i
    if (abscoef(i) > max_abs) then
      max_abs = abscoef(i)
      max_idx = i
    end if
  end do
  if (N_det > 1) call dsort(abscoef, idx, N_det)

  do i = N_det, 1, -1
    if (abscoef(i) < tau) cycle
    nkeep = nkeep + 1
    j = idx(i)
    do k = 1, N_int
      psi_det_propagators(k,1,nkeep) = psi_det(k,1,j)
      psi_det_propagators(k,2,nkeep) = psi_det(k,2,j)
    end do
    psi_coef_propagators(nkeep,:) = psi_coef(j,:)
    if (nkeep >= N_det_propagators) exit
  end do

  if (nkeep == 0 .and. N_det > 0) then
    do k = 1, N_int
      psi_det_propagators(k,1,1) = psi_det(k,1,max_idx)
      psi_det_propagators(k,2,1) = psi_det(k,2,max_idx)
    end do
    psi_coef_propagators(1,:) = psi_coef(max_idx,:)
  end if
  deallocate(abscoef, idx)
END_PROVIDER

BEGIN_PROVIDER [ integer(bit_kind), psi_det_propagators_sorted, (N_int,2,psi_det_size) ]
&BEGIN_PROVIDER [ integer, psi_det_propagators_sorted_order, (psi_det_size) ]
  implicit none
  integer, allocatable :: shortcut(:)
  integer :: i, k
  character(len=32) :: compression_policy

  psi_det_propagators_sorted = 0_bit_kind
  psi_det_propagators_sorted_order = 0
  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'

  if (N_det_propagators <= 0) then
    return
  end if

  do i = 1, N_det_propagators
    do k = 1, N_int
      psi_det_propagators_sorted(k,1,i) = psi_det_propagators(k,1,i)
      psi_det_propagators_sorted(k,2,i) = psi_det_propagators(k,2,i)
    end do
  end do

  if ((compression_policy /= 'threshold' .and. compression_policy /= 'fixed_budget') .or. N_det_propagators <= 1) then
    do i = 1, N_det_propagators
      psi_det_propagators_sorted_order(i) = i
    end do
    return
  end if

  allocate(shortcut(0:N_det_propagators+1))
  call sort_dets_ab(psi_det_propagators_sorted, psi_det_propagators_sorted_order, shortcut, N_det_propagators, N_int)
  deallocate(shortcut)
END_PROVIDER

BEGIN_PROVIDER [ integer(bit_kind), psi_det_working_sorted, (N_int,2,psi_det_size) ]
&BEGIN_PROVIDER [ integer, psi_det_working_sorted_order, (psi_det_size) ]
  implicit none
  integer, allocatable :: shortcut(:)
  integer :: i, k
  character(len=32) :: compression_policy

  psi_det_working_sorted = 0_bit_kind
  psi_det_working_sorted_order = 0
  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'

  if (N_det <= 0) then
    return
  end if

  do i = 1, N_det
    do k = 1, N_int
      psi_det_working_sorted(k,1,i) = psi_det(k,1,i)
      psi_det_working_sorted(k,2,i) = psi_det(k,2,i)
    end do
  end do

  if (compression_policy /= 'fixed_budget' .or. N_det <= 1) then
    do i = 1, N_det
      psi_det_working_sorted_order(i) = i
    end do
    return
  end if

  allocate(shortcut(0:N_det+1))
  call sort_dets_ab(psi_det_working_sorted, psi_det_working_sorted_order, shortcut, N_det, N_int)
  deallocate(shortcut)
END_PROVIDER

BEGIN_PROVIDER [ integer, N_det_generators ]
  implicit none
  integer :: i, adaptive_count, cutoff_count
  double precision :: cumulative_norm, target_norm
  character(len=32) :: compression_policy

  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'
  if (compression_policy == 'threshold' .or. compression_policy == 'fixed_budget') then
    N_det_generators = N_det_propagators
    return
  end if

  adaptive_count = max(1, min(N_det, ccik_generator_keep))
  if (ccik_generator_fraction > 0.d0) then
    target_norm = min(1.d0, max(0.d0, ccik_generator_fraction))
    cumulative_norm = 0.d0
    adaptive_count = N_det
    do i = 1, N_det
      cumulative_norm = cumulative_norm + psi_average_norm_contrib_sorted(i)
      if (cumulative_norm + 1.d-12 >= target_norm) then
        adaptive_count = i
        exit
      end if
    end do
    adaptive_count = max(ccik_generator_keep, adaptive_count)
  end if

  if (ccik_generator_abs_coeff_cutoff > 0.d0) then
    cutoff_count = 0
    do i = 1, N_det
      if (dabs(psi_coef_sorted(i,1)) < ccik_generator_abs_coeff_cutoff) exit
      cutoff_count = i
    end do
    adaptive_count = min(adaptive_count, max(1, cutoff_count))
  end if

  if (ccik_max_parents_for_expansion > 0) then
    adaptive_count = min(adaptive_count, ccik_max_parents_for_expansion)
  end if

  N_det_generators = max(1, min(N_det, adaptive_count))
END_PROVIDER

BEGIN_PROVIDER [ integer(bit_kind), psi_det_generators, (N_int,2,psi_det_size) ]
&BEGIN_PROVIDER [ double precision, psi_coef_generators, (psi_det_size,N_states) ]
  implicit none
  integer :: i, k
  character(len=32) :: compression_policy

  psi_det_generators = 0_bit_kind
  psi_coef_generators = 0.d0
  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'

  if (compression_policy == 'threshold' .or. compression_policy == 'fixed_budget') then
    do i = 1, N_det_generators
      do k = 1, N_int
        psi_det_generators(k,1,i) = psi_det_propagators(k,1,i)
        psi_det_generators(k,2,i) = psi_det_propagators(k,2,i)
      end do
      psi_coef_generators(i,:) = psi_coef_propagators(i,:)
    end do
    return
  end if

  do i = 1, N_det_generators
    do k = 1, N_int
      psi_det_generators(k,1,i) = psi_det_sorted(k,1,i)
      psi_det_generators(k,2,i) = psi_det_sorted(k,2,i)
    end do
    psi_coef_generators(i,:) = psi_coef_sorted(i,:)
  end do
END_PROVIDER

BEGIN_PROVIDER [ integer(bit_kind), psi_det_sorted_gen, (N_int,2,psi_det_size) ]
&BEGIN_PROVIDER [ double precision, psi_coef_sorted_gen, (psi_det_size,N_states) ]
&BEGIN_PROVIDER [ integer, psi_det_sorted_gen_order, (psi_det_size) ]
  implicit none
  psi_det_sorted_gen = psi_det_sorted
  psi_coef_sorted_gen = psi_coef_sorted
  psi_det_sorted_gen_order = psi_det_sorted_order
END_PROVIDER

BEGIN_PROVIDER [ integer, psi_selectors_size ]
  implicit none
  psi_selectors_size = psi_det_size
END_PROVIDER

BEGIN_PROVIDER [ integer, N_det_selectors ]
  implicit none
  if (trim(adjustl(ccik_compression_policy)) == 'threshold' .or. trim(adjustl(ccik_compression_policy)) == 'fixed_budget') then
    N_det_selectors = N_det_propagators
  else
    N_det_selectors = max(N_det_generators, min(N_det, ccik_selector_keep))
  end if
END_PROVIDER

BEGIN_PROVIDER [ integer(bit_kind), psi_selectors, (N_int,2,psi_selectors_size) ]
&BEGIN_PROVIDER [ double precision, psi_selectors_coef, (psi_selectors_size,N_states) ]
  implicit none
  integer :: i, k
  character(len=32) :: compression_policy

  psi_selectors = 0_bit_kind
  psi_selectors_coef = 0.d0
  compression_policy = trim(adjustl(ccik_compression_policy))
  if (len_trim(compression_policy) == 0) compression_policy = 'topk'

  if (compression_policy == 'threshold' .or. compression_policy == 'fixed_budget') then
    do i = 1, N_det_selectors
      do k = 1, N_int
        psi_selectors(k,1,i) = psi_det_propagators(k,1,i)
        psi_selectors(k,2,i) = psi_det_propagators(k,2,i)
      end do
      psi_selectors_coef(i,:) = psi_coef_propagators(i,:)
    end do
    return
  end if

  do i = 1, N_det_selectors
    do k = 1, N_int
      psi_selectors(k,1,i) = psi_det_sorted(k,1,i)
      psi_selectors(k,2,i) = psi_det_sorted(k,2,i)
    end do
    psi_selectors_coef(i,:) = psi_coef_sorted(i,:)
  end do
END_PROVIDER

BEGIN_PROVIDER [ double precision, psi_selectors_coef_transp, (N_states,psi_selectors_size) ]
  implicit none
  integer :: i, k

  psi_selectors_coef_transp = 0.d0
  do i = 1, N_det_selectors
    do k = 1, N_states
      psi_selectors_coef_transp(k,i) = psi_selectors_coef(i,k)
    end do
  end do
END_PROVIDER
