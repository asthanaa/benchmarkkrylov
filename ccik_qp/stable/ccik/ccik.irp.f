program ccik
  implicit none
  call ccik_banner
  call ccik_print_parameters
  if (ccik_restart_m_cycle > 1 .and. ccik_max_cycle > 1 .and. ccik_restart_nroot > 0) then
    call ccik_thick_restart_driver
  else
    call ccik_baseline_driver
  end if
end
