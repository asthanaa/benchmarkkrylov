module ccik_sparse
  implicit none
  private
  integer, parameter :: ccik_bit_kind = 8
  public :: ccik_vector
  public :: ccik_vec_free, ccik_vec_assign, ccik_vec_copy
  public :: ccik_vec_compress, ccik_vec_inner, ccik_vec_norm2
  public :: ccik_vec_scale, ccik_vec_axpy, ccik_vec_compress_preserve
  public :: ccik_vec_project_support, ccik_vec_threshold, ccik_det_in_array, ccik_det_binary_search
  public :: ccik_vec_set_propagation_support, ccik_runtime_set_propagation_override
  public :: ccik_runtime_clear_propagation_override, ccik_runtime_get_propagation_override

  type ccik_vector
    integer :: n = 0
    integer(ccik_bit_kind), allocatable :: det(:,:,:)
    double precision, allocatable :: coef(:)
    integer :: prop_n = 0
    integer(ccik_bit_kind), allocatable :: prop_det(:,:,:)
    double precision, allocatable :: prop_coef(:)
  end type ccik_vector

  type(ccik_vector), save :: ccik_runtime_propagation_override
  logical, save :: ccik_runtime_propagation_override_active = .false.

contains

  subroutine ccik_stabilize_equal_abs_groups(abscoef, idx, n)
    double precision, intent(inout) :: abscoef(*)
    integer, intent(inout) :: idx(*)
    integer, intent(in) :: n
    integer :: left, right, ii, jj, tmp_idx
    double precision :: ref_abs, tmp_abs, tie_tol

    if (n <= 1) return

    left = 1
    do while (left <= n)
      ref_abs = abscoef(left)
      tie_tol = max(1.d-15, 1.d-12 * max(1.d0, dabs(ref_abs)))
      right = left
      do while (right < n)
        if (dabs(abscoef(right+1) - ref_abs) > tie_tol) exit
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
  end subroutine ccik_stabilize_equal_abs_groups

  logical function ccik_det_eq(det_a, det_b, nint)
    integer, intent(in) :: nint
    integer(ccik_bit_kind), intent(in) :: det_a(nint,2), det_b(nint,2)
    integer :: i

    ccik_det_eq = .true.
    do i = 1, nint
      if (det_a(i,1) /= det_b(i,1) .or. det_a(i,2) /= det_b(i,2)) then
        ccik_det_eq = .false.
        return
      end if
    end do
  end function ccik_det_eq


  logical function ccik_det_lt(det_a, det_b, nint)
    integer, intent(in) :: nint
    integer(ccik_bit_kind), intent(in) :: det_a(nint,2), det_b(nint,2)
    integer :: i, j

    ccik_det_lt = .false.
    do i = 1, 2
      do j = nint, 1, -1
        if (det_a(j,i) < det_b(j,i)) then
          ccik_det_lt = .true.
          return
        else if (det_a(j,i) > det_b(j,i)) then
          return
        end if
      end do
    end do
  end function ccik_det_lt


  logical function ccik_det_in_array(det_ref, det_arr, n_det, nint)
    integer, intent(in) :: n_det, nint
    integer(ccik_bit_kind), intent(in) :: det_ref(nint,2)
    integer(ccik_bit_kind), intent(in) :: det_arr(nint,2,*)
    integer :: i

    ccik_det_in_array = .false.
    do i = 1, n_det
      if (ccik_det_eq(det_ref, det_arr(:,:,i), nint)) then
        ccik_det_in_array = .true.
        return
      end if
    end do
  end function ccik_det_in_array


  logical function ccik_det_binary_search(det_ref, det_arr, n_det, nint)
    integer, intent(in) :: n_det, nint
    integer(ccik_bit_kind), intent(in) :: det_ref(nint,2)
    integer(ccik_bit_kind), intent(in) :: det_arr(nint,2,*)
    integer :: lo, hi, mid

    ccik_det_binary_search = .false.
    if (n_det <= 0) return

    lo = 1
    hi = n_det
    do while (lo <= hi)
      mid = lo + (hi - lo) / 2
      if (ccik_det_eq(det_ref, det_arr(:,:,mid), nint)) then
        ccik_det_binary_search = .true.
        return
      else if (ccik_det_lt(det_ref, det_arr(:,:,mid), nint)) then
        hi = mid - 1
      else
        lo = mid + 1
      end if
    end do
  end function ccik_det_binary_search


  subroutine ccik_vec_free(vec)
    type(ccik_vector), intent(inout) :: vec
    if (allocated(vec%det)) deallocate(vec%det)
    if (allocated(vec%coef)) deallocate(vec%coef)
    if (allocated(vec%prop_det)) deallocate(vec%prop_det)
    if (allocated(vec%prop_coef)) deallocate(vec%prop_coef)
    vec%n = 0
    vec%prop_n = 0
  end subroutine ccik_vec_free


  subroutine ccik_vec_assign(vec, det_in, coef_in, n_in, nint)
    type(ccik_vector), intent(inout) :: vec
    integer, intent(in) :: n_in, nint
    integer(ccik_bit_kind), intent(in) :: det_in(nint,2,n_in)
    double precision, intent(in) :: coef_in(n_in)
    integer(ccik_bit_kind), allocatable :: work_det(:,:,:)
    double precision, allocatable :: work_coef(:), sorted_coef(:)
    integer, allocatable :: idx(:), shortcut(:)
    integer :: i, n_out

    call ccik_vec_free(vec)
    if (n_in <= 0) return

    allocate(work_det(nint,2,n_in), work_coef(n_in), sorted_coef(n_in))
    allocate(idx(n_in), shortcut(0:n_in+1))

    work_det = det_in
    work_coef = coef_in

    call sort_dets_ab(work_det, idx, shortcut, n_in, nint)
    do i = 1, n_in
      sorted_coef(i) = work_coef(idx(i))
    end do

    n_out = 0
    do i = 1, n_in
      if (abs(sorted_coef(i)) <= 0.d0) cycle
      if (n_out == 0) then
        n_out = 1
        work_det(:,:,n_out) = work_det(:,:,i)
        work_coef(n_out) = sorted_coef(i)
      else if (ccik_det_eq(work_det(:,:,n_out), work_det(:,:,i), nint)) then
        work_coef(n_out) = work_coef(n_out) + sorted_coef(i)
      else
        if (abs(work_coef(n_out)) <= 1.d-15) n_out = n_out - 1
        n_out = n_out + 1
        work_det(:,:,n_out) = work_det(:,:,i)
        work_coef(n_out) = sorted_coef(i)
      end if
    end do

    if (n_out > 0 .and. abs(work_coef(n_out)) <= 1.d-15) n_out = n_out - 1

    if (n_out > 0) then
      allocate(vec%det(nint,2,n_out), vec%coef(n_out))
      vec%det = work_det(:,:,1:n_out)
      vec%coef = work_coef(1:n_out)
      vec%n = n_out
    end if

    deallocate(work_det, work_coef, sorted_coef, idx, shortcut)
  end subroutine ccik_vec_assign


  subroutine ccik_vec_copy(dst, src)
    type(ccik_vector), intent(inout) :: dst
    type(ccik_vector), intent(in) :: src

    call ccik_vec_free(dst)
    if (src%n <= 0) return

    allocate(dst%det(size(src%det,1),2,src%n), dst%coef(src%n))
    dst%det = src%det
    dst%coef = src%coef
    dst%n = src%n
    if (src%prop_n > 0) then
      allocate(dst%prop_det(size(src%prop_det,1),2,src%prop_n), dst%prop_coef(src%prop_n))
      dst%prop_det = src%prop_det
      dst%prop_coef = src%prop_coef
      dst%prop_n = src%prop_n
    end if
  end subroutine ccik_vec_copy


  subroutine ccik_vec_set_propagation_support(dst, src)
    type(ccik_vector), intent(inout) :: dst
    type(ccik_vector), intent(in) :: src

    if (allocated(dst%prop_det)) deallocate(dst%prop_det)
    if (allocated(dst%prop_coef)) deallocate(dst%prop_coef)
    dst%prop_n = 0
    if (src%n <= 0) return
    allocate(dst%prop_det(size(src%det,1),2,src%n), dst%prop_coef(src%n))
    dst%prop_det = src%det
    dst%prop_coef = src%coef
    dst%prop_n = src%n
  end subroutine ccik_vec_set_propagation_support


  subroutine ccik_runtime_set_propagation_override(src)
    type(ccik_vector), intent(in) :: src

    call ccik_vec_free(ccik_runtime_propagation_override)
    ccik_runtime_propagation_override_active = .false.
    if (src%n <= 0) return
    call ccik_vec_copy(ccik_runtime_propagation_override, src)
    ccik_runtime_propagation_override_active = (ccik_runtime_propagation_override%n > 0)
  end subroutine ccik_runtime_set_propagation_override


  subroutine ccik_runtime_clear_propagation_override()
    call ccik_vec_free(ccik_runtime_propagation_override)
    ccik_runtime_propagation_override_active = .false.
  end subroutine ccik_runtime_clear_propagation_override


  subroutine ccik_runtime_get_propagation_override(dst, active)
    type(ccik_vector), intent(inout) :: dst
    logical, intent(out) :: active

    active = ccik_runtime_propagation_override_active
    call ccik_vec_free(dst)
    if (.not. active) return
    call ccik_vec_copy(dst, ccik_runtime_propagation_override)
  end subroutine ccik_runtime_get_propagation_override


  subroutine ccik_vec_compress(vec, nkeep)
    type(ccik_vector), intent(inout) :: vec
    integer, intent(in) :: nkeep
    double precision, allocatable :: abscoef(:)
    integer, allocatable :: idx(:)
    integer(ccik_bit_kind), allocatable :: det_keep(:,:,:)
    double precision, allocatable :: coef_keep(:)
    integer :: i, first_keep, n_keep, nint

    if (vec%n <= nkeep .or. nkeep <= 0) return
    nint = size(vec%det,1)

    allocate(abscoef(vec%n), idx(vec%n))
    do i = 1, vec%n
      abscoef(i) = abs(vec%coef(i))
      idx(i) = i
    end do
    call dsort(abscoef, idx, vec%n)
    call ccik_stabilize_equal_abs_groups(abscoef, idx, vec%n)

    n_keep = min(nkeep, vec%n)
    first_keep = vec%n - n_keep + 1
    allocate(det_keep(nint,2,n_keep), coef_keep(n_keep))
    do i = 1, n_keep
      det_keep(:,:,i) = vec%det(:,:,idx(first_keep + i - 1))
      coef_keep(i) = vec%coef(idx(first_keep + i - 1))
    end do

    call ccik_vec_assign(vec, det_keep, coef_keep, n_keep, nint)
    deallocate(abscoef, idx, det_keep, coef_keep)
  end subroutine ccik_vec_compress


  subroutine ccik_vec_compress_preserve(vec, nkeep, ref, preserve_cap)
    type(ccik_vector), intent(inout) :: vec
    integer, intent(in) :: nkeep, preserve_cap
    type(ccik_vector), intent(in) :: ref
    double precision, allocatable :: abscoef(:), absref(:)
    integer, allocatable :: idx(:), idx_ref(:)
    logical, allocatable :: chosen(:)
    integer(ccik_bit_kind), allocatable :: det_keep(:,:,:)
    double precision, allocatable :: coef_keep(:)
    integer :: i, j, ref_idx, n_keep, nint, ref_start, nref_keep

    if (vec%n <= nkeep .or. nkeep <= 0) return
    if (preserve_cap <= 0 .or. ref%n <= 0) then
      call ccik_vec_compress(vec, nkeep)
      return
    end if

    nint = size(vec%det,1)
    allocate(chosen(vec%n))
    chosen = .false.
    n_keep = 0

    allocate(absref(ref%n), idx_ref(ref%n))
    do i = 1, ref%n
      absref(i) = abs(ref%coef(i))
      idx_ref(i) = i
    end do
    call dsort(absref, idx_ref, ref%n)
    call ccik_stabilize_equal_abs_groups(absref, idx_ref, ref%n)

    nref_keep = min(preserve_cap, ref%n)
    ref_start = ref%n - nref_keep + 1
    do i = ref_start, ref%n
      ref_idx = idx_ref(i)
      do j = 1, vec%n
        if (chosen(j)) cycle
        if (ccik_det_eq(vec%det(:,:,j), ref%det(:,:,ref_idx), nint)) then
          chosen(j) = .true.
          n_keep = n_keep + 1
          exit
        end if
      end do
      if (n_keep >= nkeep) exit
    end do

    allocate(abscoef(vec%n), idx(vec%n))
    do i = 1, vec%n
      abscoef(i) = abs(vec%coef(i))
      idx(i) = i
    end do
    call dsort(abscoef, idx, vec%n)
    call ccik_stabilize_equal_abs_groups(abscoef, idx, vec%n)

    do i = vec%n, 1, -1
      j = idx(i)
      if (chosen(j)) cycle
      chosen(j) = .true.
      n_keep = n_keep + 1
      if (n_keep >= nkeep) exit
    end do

    allocate(det_keep(nint,2,n_keep), coef_keep(n_keep))
    j = 0
    do i = 1, vec%n
      if (.not. chosen(i)) cycle
      j = j + 1
      det_keep(:,:,j) = vec%det(:,:,i)
      coef_keep(j) = vec%coef(i)
    end do

    call ccik_vec_assign(vec, det_keep, coef_keep, n_keep, nint)
    deallocate(abscoef, absref, idx, idx_ref, chosen, det_keep, coef_keep)
  end subroutine ccik_vec_compress_preserve


  subroutine ccik_vec_threshold(vec, abs_tol, dropped_norm, used_fallback)
    type(ccik_vector), intent(inout) :: vec
    double precision, intent(in) :: abs_tol
    double precision, intent(out) :: dropped_norm
    logical, intent(out) :: used_fallback
    integer(ccik_bit_kind), allocatable :: det_keep(:,:,:)
    double precision, allocatable :: coef_keep(:)
    integer :: i, n_keep, max_idx, nint
    double precision :: tau, max_abs, keep_norm_sq, total_norm_sq

    dropped_norm = 0.d0
    used_fallback = .false.
    if (vec%n <= 0) return

    tau = max(0.d0, abs_tol)
    nint = size(vec%det, 1)
    total_norm_sq = ccik_vec_norm2(vec)
    n_keep = 0
    max_idx = 1
    max_abs = -1.d0

    do i = 1, vec%n
      if (dabs(vec%coef(i)) > max_abs) then
        max_abs = dabs(vec%coef(i))
        max_idx = i
      end if
      if (dabs(vec%coef(i)) >= tau) n_keep = n_keep + 1
    end do

    if (n_keep <= 0) then
      allocate(det_keep(nint,2,1), coef_keep(1))
      det_keep(:,:,1) = vec%det(:,:,max_idx)
      coef_keep(1) = vec%coef(max_idx)
      keep_norm_sq = coef_keep(1) * coef_keep(1)
      dropped_norm = dsqrt(max(0.d0, total_norm_sq - keep_norm_sq))
      used_fallback = .true.
      call ccik_vec_assign(vec, det_keep, coef_keep, 1, nint)
      deallocate(det_keep, coef_keep)
      return
    end if

    allocate(det_keep(nint,2,n_keep), coef_keep(n_keep))
    n_keep = 0
    keep_norm_sq = 0.d0
    do i = 1, vec%n
      if (dabs(vec%coef(i)) < tau) cycle
      n_keep = n_keep + 1
      det_keep(:,:,n_keep) = vec%det(:,:,i)
      coef_keep(n_keep) = vec%coef(i)
      keep_norm_sq = keep_norm_sq + coef_keep(n_keep) * coef_keep(n_keep)
    end do

    dropped_norm = dsqrt(max(0.d0, total_norm_sq - keep_norm_sq))
    call ccik_vec_assign(vec, det_keep, coef_keep, n_keep, nint)
    deallocate(det_keep, coef_keep)
  end subroutine ccik_vec_threshold


  subroutine ccik_vec_project_support(dst, src, support)
    type(ccik_vector), intent(inout) :: dst
    type(ccik_vector), intent(in) :: src, support
    integer(ccik_bit_kind), allocatable :: det_keep(:,:,:)
    double precision, allocatable :: coef_keep(:)
    integer :: isrc, isup, nkeep, nint

    call ccik_vec_free(dst)
    if (src%n <= 0 .or. support%n <= 0) return

    nint = size(src%det, 1)
    allocate(det_keep(nint,2,min(src%n, support%n)), coef_keep(min(src%n, support%n)))
    nkeep = 0
    isrc = 1
    isup = 1
    do while (isrc <= src%n .and. isup <= support%n)
      if (ccik_det_eq(src%det(:,:,isrc), support%det(:,:,isup), nint)) then
        nkeep = nkeep + 1
        det_keep(:,:,nkeep) = src%det(:,:,isrc)
        coef_keep(nkeep) = src%coef(isrc)
        isrc = isrc + 1
        isup = isup + 1
      else if (ccik_det_lt(src%det(:,:,isrc), support%det(:,:,isup), nint)) then
        isrc = isrc + 1
      else
        isup = isup + 1
      end if
    end do

    if (nkeep > 0) then
      call ccik_vec_assign(dst, det_keep(:,:,1:nkeep), coef_keep(1:nkeep), nkeep, nint)
    end if
    deallocate(det_keep, coef_keep)
  end subroutine ccik_vec_project_support


  double precision function ccik_vec_inner(a, b)
    type(ccik_vector), intent(in) :: a, b
    integer :: ia, ib, nint

    ccik_vec_inner = 0.d0
    if (a%n <= 0 .or. b%n <= 0) return

    nint = size(a%det,1)
    ia = 1
    ib = 1
    do while (ia <= a%n .and. ib <= b%n)
      if (ccik_det_eq(a%det(:,:,ia), b%det(:,:,ib), nint)) then
        ccik_vec_inner = ccik_vec_inner + a%coef(ia) * b%coef(ib)
        ia = ia + 1
        ib = ib + 1
      else if (ccik_det_lt(a%det(:,:,ia), b%det(:,:,ib), nint)) then
        ia = ia + 1
      else
        ib = ib + 1
      end if
    end do
  end function ccik_vec_inner


  double precision function ccik_vec_norm2(vec)
    type(ccik_vector), intent(in) :: vec
    ccik_vec_norm2 = ccik_vec_inner(vec, vec)
  end function ccik_vec_norm2


  subroutine ccik_vec_scale(vec, alpha)
    type(ccik_vector), intent(inout) :: vec
    double precision, intent(in) :: alpha
    if (vec%n <= 0) return
    vec%coef(:) = alpha * vec%coef(:)
  end subroutine ccik_vec_scale


  subroutine ccik_vec_axpy(y, alpha, x)
    type(ccik_vector), intent(inout) :: y
    double precision, intent(in) :: alpha
    type(ccik_vector), intent(in) :: x
    integer(ccik_bit_kind), allocatable :: det_tmp(:,:,:)
    double precision, allocatable :: coef_tmp(:)
    integer :: i, n_tmp, nint

    if (x%n <= 0 .or. alpha == 0.d0) return
    nint = size(x%det,1)

    if (y%n <= 0) then
      allocate(det_tmp(nint,2,x%n), coef_tmp(x%n))
      det_tmp = x%det
      coef_tmp = alpha * x%coef
      call ccik_vec_assign(y, det_tmp, coef_tmp, x%n, nint)
      deallocate(det_tmp, coef_tmp)
      return
    end if

    n_tmp = y%n + x%n
    allocate(det_tmp(nint,2,n_tmp), coef_tmp(n_tmp))
    det_tmp(:,:,1:y%n) = y%det(:,:,1:y%n)
    coef_tmp(1:y%n) = y%coef(1:y%n)
    do i = 1, x%n
      det_tmp(:,:,y%n+i) = x%det(:,:,i)
      coef_tmp(y%n+i) = alpha * x%coef(i)
    end do

    call ccik_vec_assign(y, det_tmp, coef_tmp, n_tmp, nint)
    deallocate(det_tmp, coef_tmp)
  end subroutine ccik_vec_axpy

end module ccik_sparse
