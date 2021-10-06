PROGRAM READ_TABLE
   implicit none
   character*256 :: J_tab_name, param_tab_name, aio_tab_name
   integer :: fid1, fid3, i, j, k, p, q
   integer :: n_Wee, n_Fr2, n_Eta
   real(kind=8) :: Weemax, Weemin, Fr2max, Fr2min, Etamax, Etamin
   real(kind=8),dimension(:),allocatable :: Wee_lst, Fr2_lst, Eta_lst
   real(kind=8),dimension(:,:,:),  allocatable :: logJ_tab
   fid1 = 111
   fid3 = 333
   J_tab_name = 'Tab_J_ascii.dat'
   param_tab_name = 'Tab_Jpa_ascii.dat'
   open(unit=fid1,file=J_tab_name,status='old',action='read')
   open(unit=fid3,file=param_tab_name,status='old',action='read')

   read(fid3,*) n_Wee, n_Fr2, n_Eta
   read(fid3,*) Weemin, Fr2min, Etamin
   read(fid3,*) Weemax, Fr2max, Etamax
   allocate(Wee_lst(n_Wee)); allocate(Fr2_lst(n_Fr2)); allocate(Eta_lst(n_Eta))
   allocate(logJ_tab(n_Wee,n_Fr2,n_Eta))

   read(fid1,*) (((logJ_tab(i,j,k),k=1,n_Eta),j=1,n_Fr2),i=1,n_Wee)

   read(fid3,*) (Wee_lst(p),p=1,n_Wee)
   read(fid3,*) (Fr2_lst(p),p=1,n_Fr2)
   read(fid3,*) (Eta_lst(p),p=1,n_Eta)
   close(fid1)
   close(fid3)

   write(6,'(5ES20.7)') (log(Wee_lst(p)),p=1,n_Wee)
   
   aio_tab_name = 'J_table_bin.dat'
   open(unit=fid1,file=aio_tab_name,status='replace',action='write',form='unformatted')
   write(fid1) n_Wee, n_Fr2, n_Eta
   write(fid1) Weemin, Weemax, Fr2min, Fr2max, Etamin, Etamax
   write(fid1) (log(Wee_lst(p)),p=1,n_Wee)
   write(fid1) (log(Fr2_lst(p)),p=1,n_Fr2)
   write(fid1) (log(Eta_lst(p)),p=1,n_Eta)
   write(fid1) (((logJ_tab(i,j,k),k=1,n_Eta),j=1,n_Fr2),i=1,n_Wee)
   close(fid1)
END