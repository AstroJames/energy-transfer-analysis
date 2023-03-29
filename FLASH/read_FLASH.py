from h5py import File
import numpy as np
import numba as nb
from numba import njit
from mpi4py import MPI

class Particles:

    def __init__(self,filename) -> None:
        # file attributes
        self.filename       = filename
        # particle attribues 
        self.n_particles    = 0
        self.__set_number_of_particles()
        self.blocks         = np.zeros(self.n_particles, dtype=np.int32)
        self.id             = np.zeros(self.n_particles, dtype=np.int32)
        # particle physics attribues 
        self.dens           = np.zeros(self.n_particles, dtype=np.float32)
        self.posx           = np.zeros(self.n_particles, dtype=np.float32)
        self.posy           = np.zeros(self.n_particles, dtype=np.float32)
        self.posz           = np.zeros(self.n_particles, dtype=np.float32)
        self.velx           = np.zeros(self.n_particles, dtype=np.float32)
        self.vely           = np.zeros(self.n_particles, dtype=np.float32)
        self.velz           = np.zeros(self.n_particles, dtype=np.float32)
        self.magx           = np.zeros(self.n_particles, dtype=np.float32)
        self.magy           = np.zeros(self.n_particles, dtype=np.float32)
        self.magz           = np.zeros(self.n_particles, dtype=np.float32)

    def __set_number_of_particles(self) -> None:
        """
        This function reads the number of particles in the FLASH particle file
        """
        g = File(self.filename, "r")
        self.n_particles = len(g['tracer particles'][:,0])
        print(f"Numer of particles: {self.n_particles}")
        g.close()

    def __reformat_part_str(self,
                            part_str,
                            idx: int) -> str:
        """
        This function reformats the particle string to be in the correct format
        """
        return str(part_str[idx][0]).split("'")[1]

    def read(self,
             part_str) -> None:
        """
        This function reads in the FLASH particle block data,
        i.e., the block membership of each particle
        """

        block_idx_dic = {"block":0,
                         "id"   :6,
                         "dens" :1,
                         "posx" :2,
                         "posy" :3,
                         "posz" :4,
                         "velx" :7,
                         "vely" :8,
                         "velz" :9,
                         "magx" :10,
                         "magy" :11,
                         "magz" :12
        }

        field_lookup_type = {
            "dens"  : "scalar",
            "id"    : "scalar",
            "block" : "scalar",            
            "vel"   : "vector",
            "mag"   : "vector",
            "pos"   : "vector",
            "vort"  : "vector",
            "mpr"   : "vector",
            "tprs"  : "vector"
        }

        if field_lookup_type[part_str] == "scalar":
            # read single component particle data
            g = File(self.filename, 'r')
            print(f"Reading in particle attribute: {self.__reformat_part_str(g['particle names'],block_idx_dic[part_str])}")
            if part_str == "id":
                setattr(self,
                        part_str,
                        g['tracer particles'][:,block_idx_dic[part_str]].astype(int))
            else:
                setattr(self,
                        part_str,
                        g['tracer particles'][:,block_idx_dic[part_str]])
            g.close() 

        elif field_lookup_type[part_str] == "vector":
            # read vector component particle data
            g = File(self.filename, 'r')
            for coord in ["x","y","z"]:
                data_set_str = f"{part_str}{coord}"
                print(f"Reading in particle attribute: {self.__reformat_part_str(g['particle names'],block_idx_dic[data_set_str])}")
                setattr(self,
                        data_set_str,
                        g['tracer particles'][:,block_idx_dic[data_set_str]])
            g.close()      

    def sort_particles(self):
        # sort quick sort particles by id (O n log n)
        print("sort_particles: beginning to sort particles by id.")
        idx = np.argsort(self.id)
        print("sort_particles: finished sorting particles by id.")
        # loop through all of the attributes and sort them
        for attr in self.__dict__:
            if (attr != "filename") and (attr != "n_particles"):
                setattr(self,attr,getattr(self,attr)[idx])



class Fields():

    def __init__(self,
                 filename: str,
                 reformat: bool = False) -> None:
        """
        Initialize a FLASHGridData object by reading in the data from the specified file.

        Parameters
        ----------
        filename : str 
            The name of the file containing the FLASH grid data.
        reformat : bool, optional
            Whether to reformat the data in the file into 3D arrays (True) or keep it in 1D arrays (False).
            Default is False.

        Attributes
        ----------
        filename : str
            The name of the file containing the FLASH grid data.
        reformat : bool
            Whether to reformat the data in the file into 3D arrays (True) or keep it in 1D arrays (False).
        n_cores : int
            The number of cores used in the simulation.
        nxb : int
            The number of blocks in the x direction.
        nyb : int
            The number of blocks in the y direction.
        nzb : int
            The number of blocks in the z direction.
        n_cells : int
            The total number of cells in the simulation.
        int_properties : dict
            A dictionary containing integer simulation properties.
        str_properties : dict
            A dictionary containing string simulation properties.
        logic_properties : dict
            A dictionary containing logical simulation properties.
        dens : numpy array
            The density field.
        velx : numpy array
            The x velocity field.
        vely : numpy array
            The y velocity field.
        velz : numpy array
            The z velocity field.
        magx : numpy array
            The x magnetic field.
        magy : numpy array
            The y magnetic field.
        magz : numpy array
            The z magnetic field.
        tensx : numpy array
            The x magnetic tension field.
        tensy : numpy array
            The y magnetic tension field.
        tensz : numpy array
            The z magnetic tension field.
        vortx : numpy array
            The x vorticity field.
        vorty : numpy array
            The y vorticity field.
        vortz : numpy array
            The z vorticity field.

        """

        # simulation attributes
        self.filename           = filename
        self.reformat           = reformat
        self.n_cores            = 0
        self.nxb                = 0
        self.nyb                = 0
        self.nzb                = 0
        self.n_cells            = 0
        self.int_properties     = {}
        self.str_properties     = {}
        self.logic_properties   = {}  

        # read in the simulation properties
        self.__read_sim_properties()  
        self.__read_sim_cells() 

        # grid data attributes
        # if the data is going to be reformated, preallocate the 3D
        # arrays for the grid data
        if self.reformat:
            init_field = np.zeros((self.nyb*self.int_properties["jprocs"],
                                    self.nxb*self.int_properties["iprocs"],
                                    self.nzb*self.int_properties["kprocs"]), dtype=np.float32)
            fourier_init_field = np.zeros((self.nyb*self.int_properties["jprocs"],
                                    self.nxb*self.int_properties["iprocs"],
                                    self.nzb*self.int_properties["kprocs"]), dtype=np.complex128)
        else:
            # otherwise, preallocate the 1D arrays for the grid data
            init_field = np.zeros(self.n_cells, dtype=np.float32)

        self.dens   = init_field
        self.velx   = init_field
        self.vely   = init_field
        self.velz   = init_field
        self.magx   = init_field
        self.magy   = init_field
        self.magz   = init_field
        self.tensx  = init_field
        self.tensy  = init_field
        self.tensz  = init_field
        self.vortx  = init_field
        self.vorty  = init_field
        self.vortz  = init_field 

        # spectral transfer variables
        self.dens_k  = fourier_init_field
        self.velx_k  = fourier_init_field
        self.vely_k  = fourier_init_field
        self.velz_k  = fourier_init_field   

    def __read_sim_cells(self) -> None:
        """
        This function reads in the number of cores from the FLASH file.
        """
        g = File(self.filename, 'r')
        self.n_cores    = g['dens'].shape[0]
        self.nxb        = g['dens'].shape[3]
        self.nyb        = g['dens'].shape[1]
        self.nzb        = g['dens'].shape[2]
        self.n_cells    = self.n_cores*self.nxb*self.nyb*self.nzb
        #print(f"Number of cores: {self.n_cells}")
        g.close()

    def __read_sim_properties(self) -> None:
        """
        This function reads in the FLASH field properties.
        """
        g = File(self.filename, 'r')
        self.int_properties     = {str(key).split("'")[1].strip(): value for key, value in g["integer runtime parameters"]}
        self.str_properties     = {str(key).split("'")[1].strip(): str(value).split("'")[1].strip() for key, value in g["string runtime parameters"]}
        self.logic_properties   = {str(key).split("'")[1].strip(): value for key, value in g["logical runtime parameters"]}
        g.close()

    def set_reformat(self,
                     reformat: bool) -> None:
        """
        This function sets the reformat flag for the FLASH field data.
        """
        self.reformat = reformat

    def read(self,
             field_str: str,
             debug: bool = False) -> None:
        """
        This function reads in the FLASH grid data
        """

        field_lookup_type = {
            "dens": "scalar",
            "dvvl": "scalar",
            "vel" : "vector",
            "mag" : "vector",
            "tens": "vector",
            "vort": "vector",
            "mpr" : "vector",
            "tprs": "vector"
        }

        if field_lookup_type[field_str] == "scalar":
            g = File(self.filename, 'r')
            print(f"Reading in grid attribute: {field_str}")
            if self.reformat:
                print(f"Reading in reformatted grid attribute: {field_str}")
                setattr(self, field_str, 
                        reformat_FLASH_field(g[field_str][:,:,:,:],
                                            self.nxb,
                                            self.nyb,
                                            self.nzb,
                                            self.int_properties["iprocs"],
                                            self.int_properties["jprocs"],
                                            self.int_properties["kprocs"],
                                            debug))
            else:
                setattr(self, field_str, g[field_str][:,:,:,:])
            g.close()
        
        elif field_lookup_type[field_str] == "vector":
            g = File(self.filename, 'r')
            for coord in ["x","y","z"]:
                print(f"Reading in grid attribute: {field_str}{coord}")
                if self.reformat:
                    print(f"Reading in reformatted grid attribute: {field_str}{coord}")
                    setattr(self, f"{field_str}{coord}",   
                            reformat_FLASH_field(g[f"{field_str}{coord}"][:,:,:,:],
                                                self.nxb,
                                                self.nyb,
                                                self.nzb,
                                                self.int_properties["iprocs"],
                                                self.int_properties["jprocs"],
                                                self.int_properties["kprocs"],
                                                debug))
                else:
                    setattr(self, f"{field_str}{coord}", g[f"{field_str}{coord}"][:,:,:,:])
            g.close()

    def fourier_transform(self,
                          field_label):
        """
        This function performs a Fourier transform on the grid data.
        """
        if not self.set_reformat:
            return print("First reformat the fields before Fourier transforming.")
        
        field_labels = ["dens","vel"]

        if field_label == "dens":
            print("Fourier transforming: dens")
            self.dens_k = np.fft.fftn(self.dens,norm='forward')
        if field_label =="vel":
            print("Fourier transforming: vel")
            self.velx_k = np.fft.fftn(self.velx,norm='forward')
            self.vely_k = np.fft.fftn(self.vely,norm='forward')
            self.velz_k = np.fft.fftn(self.velz,norm='forward')
        else:
            print(f'Choose one of: {[lab for lab in field_labels]}')




@njit(nb.float32[:,:,:](nb.float32[:,:,:,:], 
      nb.int64, nb.int64, nb.int64, 
      nb.int64, nb.int64, nb.int64, 
      nb.boolean))
def reformat_FLASH_field(field  : np.ndarray,
                         nxb    : int,
                         nyb    : int,
                         nzb    : int,
                         iprocs : int,
                         jprocs : int,
                         kprocs : int,
                         debug) -> np.ndarray:
    """
    This function reformats the FLASH block / core format into
    (x,y,z) format for processing in real-space coordinates utilising
    numba's jit compiler, producing roughly a two-orders of magnitude
    speedup compared to the pure python version.

    INPUTS:
    field   - the FLASH field in (core,block_x,block_y,block_z) coordinates
    iprocs  - the number of cores in the x-direction
    jprocs  - the number of cores in the y-direction
    kprocs  - the number of cores in the z-direction
    debug   - flag to print debug information


    OUTPUTs:
    field_sorted - the organised 3D field in (x,y,z) coordinates

    """

    # The block counter for looping through blocks
    block_counter: nb.int32 = 0

    if debug:
        print(f"reformat_FLASH_field: nxb = {nxb}")
        print(f"reformat_FLASH_field: nyb = {nyb}")
        print(f"reformat_FLASH_field: nzb = {nzb}")
        print(f"reformat_FLASH_field: iprocs = {iprocs}")
        print(f"reformat_FLASH_field: jprocs = {jprocs}")
        print(f"reformat_FLASH_field: kprocs = {kprocs}")

    # Initialise an empty (x,y,z) field
    # has to be the same dtype as input field (single precision)
    field_sorted = np.zeros((nyb*jprocs,
                             nzb*kprocs,
                             nxb*iprocs),
                             dtype=np.float32)

    # Sort the unsorted field
    if debug:
        print("reformat_FLASH_field: Beginning to sort field.")
    for j in range(jprocs):
        for k in range(kprocs):
            for i in range(iprocs):
                field_sorted[j*nyb:(j+1)*nyb,
                             k*nzb:(k+1)*nzb,
                             i*nxb:(i+1)*nxb, 
                             ] = field[block_counter, :, :, :]
                block_counter += 1
    if debug:
        print("reformat_FLASH_field: Sorting complete.")
    return field_sorted